#!/usr/bin/env python3
"""
behavior_manager_node.py  (v2 — Data → Decision → Action)
==========================================================
Единое «Вечное дерево» @ 10 Гц.
Blackboard — единственный источник истины.

Данные о мире текут из ROS-топиков → Blackboard → BT принимает решения.

Blackboard ключи (BehaviorManagerNode заполняет из подписок):
  /robot/sleep             bool  — активный спящий режим
  /robot/sleep_requested   bool  — запрошен переход в сон (от LLM robot_control)
  /robot/sleep_text        str   — прощальная фраза перед сном
  /robot/command           dict  — pending физ. команда {action, ...} от LLM
  /llm/text                str   — текст ответа LLM для озвучки ('' при streaming)
  /llm/voice_style         str   — инструкция голоса (CosyVoice instruct)
  /llm/emotion             str   — желаемая эмоция лица во время речи
  /llm/has_content         bool  — True когда есть ответ для DialogueBranch (text или streamed)
  /social/person_present   bool  — человек в кадре
  /social/name             str   — имя человека
  /social/emotion          str   — текущая эмоция человека
  /social/should_greet     bool  — нужно поздороваться
  /social/greet_text       str   — текст приветствия
  /social/farewell_pending bool  — человек только что ушёл
  /social/farewell_text    str   — текст прощания
  /social/introducing      bool  — идёт сбор имени
  /social/introduce_pending bool — IdentityManager просит произнести фразу знакомства
  /social/introduce_text   str   — текст для произнесения в режиме знакомства
  /search/query            str   — запрос для поиска
  /search/result           dict  — результат поиска

Дерево (Selector, no-memory — каждый тик с начала):
  Root
  ├── SleepTransition  — если запрошен сон: прощание → сон
  ├── SleepActive      — если уже спит: блокирует всё ниже
  ├── RobotCommand     — pending физическая команда от LLM (move/arm/head)
  ├── WebSearch        — pending поисковый запрос от LLM
  ├── SocialBranch     — если человек в кадре:
  │     IntroducingBlock / GreetBranch / DialogueBranch / IdleGaze
  ├── FarewellBranch   — если человек только что ушёл: прощание
  ├── SoundScanBranch  — wake word: поворот КОРПУСА на голос (/sound_direction) до /human_detected
  ├── PIRScanBranch    — чистое PIR-движение (без голоса): голова влево→вправо→центр
  └── GlobalIdle       — ничего не происходит (моргание)
"""

import json
import math
import random
import threading
import time
import requests

import rclpy
from rclpy.node import Node
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist

import py_trees

from inmoov_msgs.action import Speak
from inmoov_msgs.msg import SoundDirection


# ══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ЛИСТЬЯ
# ══════════════════════════════════════════════════════════════════════════════

class CheckBB(py_trees.behaviour.Behaviour):
    """Condition leaf: SUCCESS если check_fn(bb[key]) истинно."""

    def __init__(self, name: str, key: str, check_fn=None):
        super().__init__(name)
        self._attr  = key.strip('/').replace('/', '.')
        self._check = check_fn if check_fn is not None else bool
        self._bb    = py_trees.blackboard.Client(name=f'Chk:{name}')
        self._bb.register_key(key=key, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        try:
            val = self._bb
            for part in self._attr.split('.'):
                val = getattr(val, part)
            ok = self._check(val)
        except KeyError:
            ok = False
        return (py_trees.common.Status.SUCCESS if ok
                else py_trees.common.Status.FAILURE)


class SetBB(py_trees.behaviour.Behaviour):
    """Action leaf: записывает значение в BB. Всегда SUCCESS."""

    def __init__(self, name: str, key: str, value):
        super().__init__(name)
        self._parts = key.strip('/').replace('/', '.').split('.')
        self._value = value
        self._bb    = py_trees.blackboard.Client(name=f'Set:{name}')
        self._bb.register_key(key=key, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        obj = self._bb
        for part in self._parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, self._parts[-1], self._value)
        return py_trees.common.Status.SUCCESS


class AlwaysRunning(py_trees.behaviour.Behaviour):
    """Блокирует поддерево — возвращает RUNNING бесконечно."""

    def __init__(self, name: str = 'AlwaysRunning'):
        super().__init__(name)

    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.RUNNING


# ══════════════════════════════════════════════════════════════════════════════
# ДЕЙСТВИЯ
# ══════════════════════════════════════════════════════════════════════════════

class ExecuteRobotCommand(py_trees.behaviour.Behaviour):
    """Читает /robot/command из BB, публикует физическую команду, очищает ключ."""

    # Абсолютные углы покоя головы (должны совпадать с arduino_left_node и vision_head_tracker)
    _REST_ROTHEAD = 90.0
    _REST_NECK    = 40.0

    def __init__(self, node: Node):
        super().__init__('ExecuteRobotCommand')
        self._node       = node
        self._pub_vel    = node.create_publisher(Twist,      'cmd_vel',          10)
        self._pub_arm    = node.create_publisher(String,     'arm_command',      10)
        self._pub_joint  = node.create_publisher(JointState, '/joint_command',   10)
        self._pub_stat   = node.create_publisher(String,     'status_request',   10)
        self._timer      = None   # таймер остановки движения
        self._head_timer = None   # таймер возврата head_tracker после ручной команды

        self._bb = py_trees.blackboard.Client(name='ExecCmd')
        self._bb.register_key(key='/robot/command',
                               access=py_trees.common.Access.READ)
        self._bb.register_key(key='/robot/command',
                               access=py_trees.common.Access.WRITE)

    def initialise(self):
        cmd    = self._bb.robot.command
        action = cmd.get('action', '')

        if action == 'move':
            self._do_move(cmd)
        elif action == 'arm':
            msg = String()
            msg.data = json.dumps(
                {'command': cmd.get('command', 'home'), 'target': cmd.get('target', '')},
                ensure_ascii=False,
            )
            self._pub_arm.publish(msg)
            self._node.get_logger().info(f'Arm: {msg.data}')
        elif action == 'head':
            self._do_head(cmd)
        elif action == 'status':
            msg = String()
            msg.data = cmd.get('query', 'all')
            self._pub_stat.publish(msg)
        else:
            self._node.get_logger().warn(f'ExecuteRobotCommand: неизвестный action={action}')

        self._bb.robot.command = {}

    def _do_head(self, cmd: dict):
        pan  = float(cmd.get('pan',  0))
        tilt = float(cmd.get('tilt', 0))

        rothead = max(30.0, min(140.0, self._REST_ROTHEAD + pan))
        neck    = max(0.0,  min(100.0, self._REST_NECK    + tilt))

        msg = JointState()
        msg.name     = ['rothead', 'neck']
        msg.position = [
            (rothead - 90.0) * math.pi / 180.0,
            (neck    - 90.0) * math.pi / 180.0,
        ]

        # Выключаем head_tracker ПЕРВЫМ — он опубликует REST(rothead=90,neck=40) на /joint_command.
        # Публикуем нашу команду через 200мс: REST гарантированно придёт в arduino раньше,
        # наша команда придёт позже и перезапишет его. Без задержки REST перезаписывает нас.
        self._node.enable_head_tracker(False)
        self._node.get_logger().info(
            f'Head: pan={pan:+.0f}° tilt={tilt:+.0f}° → rothead={rothead:.0f}° neck={neck:.0f}°')

        if self._head_timer:
            self._head_timer.cancel()
        self._head_timer = threading.Timer(0.2, self._send_head_cmd, args=[msg])
        self._head_timer.daemon = True
        self._head_timer.start()

    def _send_head_cmd(self, msg: JointState):
        msg.header.stamp = self._node.get_clock().now().to_msg()
        self._pub_joint.publish(msg)
        # Через 5с возвращаем управление head_tracker (возобновит слежение за лицом)
        self._head_timer = threading.Timer(5.0, self._resume_head_tracker)
        self._head_timer.daemon = True
        self._head_timer.start()

    def _resume_head_tracker(self):
        self._node.enable_head_tracker(True)
        self._head_timer = None
        self._node.get_logger().info('Head: возврат управления head_tracker')

    def _do_move(self, cmd: dict):
        direction = cmd.get('direction', 'stop')
        speed     = float(cmd.get('speed', 0.3))
        duration  = float(cmd.get('duration', 2.0))
        twist     = Twist()
        mapping   = {
            'forward':  ('linear.x',   speed),
            'backward': ('linear.x',  -speed),
            'left':     ('angular.z',  speed),
            'right':    ('angular.z', -speed),
        }
        if direction in mapping:
            attr, val = mapping[direction]
            obj, field = attr.split('.')
            setattr(getattr(twist, obj), field, val)
        self._pub_vel.publish(twist)
        self._node.get_logger().info(f'Move: {direction} speed={speed}')
        if direction != 'stop' and duration > 0:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(duration, self._stop_vel)
            self._timer.start()

    def _stop_vel(self):
        self._pub_vel.publish(Twist())
        self._timer = None

    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        if self._timer:
            self._timer.cancel()
            self._timer = None


class ExpressEmotion(py_trees.behaviour.Behaviour):
    """Публикует эмоцию на /face_expression. Фиксированная строка или из BB."""

    VALID = {
        'neutral', 'angry', 'wink', 'disgust', 'fear', 'happy', 'smile',
        'sad', 'sigh', 'sorry', 'suspicious', 'thinking', 'unamused',
        'surprise', 'sleeping', 'anger', 'surprised', 'contempt',
        'anxiety', 'disappointment', 'frown', 'gasp', 'excited',
        'chuckle', 'grin', 'helplessness',
    }

    def __init__(self, node: Node, emotion: str | None = None,
                 bb_key: str | None = None):
        super().__init__(f'ExpressEmotion({emotion or bb_key})')
        self._node    = node
        self._emotion = emotion
        self._pub     = node.create_publisher(String, '/face_expression', 10)
        self._bb      = None
        if bb_key:
            self._bb      = py_trees.blackboard.Client(name=f'Expr:{bb_key}')
            self._bb_attr = bb_key.strip('/').replace('/', '.')
            self._bb.register_key(key=bb_key, access=py_trees.common.Access.READ)

    def initialise(self):
        if self._bb is not None:
            try:
                val = self._bb
                for part in self._bb_attr.split('.'):
                    val = getattr(val, part)
                emotion = str(val).lower().strip()
            except Exception:
                emotion = 'neutral'
        else:
            emotion = (self._emotion or 'neutral').lower()

        if emotion not in self.VALID:
            emotion = 'neutral'
        msg = String()
        msg.data = emotion
        self._pub.publish(msg)
        self._node.get_logger().info(f'Эмоция лица: {emotion}')

    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS


class SpeakBehaviour(py_trees.behaviour.Behaviour):
    """
    TTS leaf: RUNNING пока идёт речь.
    Поддерживает preemption через terminate().
    Текст — фиксированная строка или из BB (bb_key).
    """

    def __init__(self, node: Node, text: str | None = None,
                 bb_key: str | None = None, voice_bb_key: str | None = None):
        super().__init__('Speak')
        self._node          = node
        self._text          = text
        self._bb_key        = bb_key
        self._voice_bb_key  = voice_bb_key
        self._client        = ActionClient(node, Speak, 'speak')
        self._goal_handle   = None
        self._done          = threading.Event()
        self._success       = False
        self._bb            = None
        if bb_key or voice_bb_key:
            self._bb = py_trees.blackboard.Client(name=f'Speak:{bb_key or voice_bb_key}')
            if bb_key:
                self._bb_attr = bb_key.strip('/').replace('/', '.')
                self._bb.register_key(key=bb_key, access=py_trees.common.Access.READ)
            if voice_bb_key:
                self._voice_bb_attr = voice_bb_key.strip('/').replace('/', '.')
                self._bb.register_key(key=voice_bb_key, access=py_trees.common.Access.READ)

    def initialise(self):
        self._done.clear()
        self._success     = False
        self._goal_handle = None

        text  = self._text
        voice = ''
        if self._bb is not None:
            try:
                if self._bb_key:
                    val = self._bb
                    for part in self._bb_attr.split('.'):
                        val = getattr(val, part)
                    text = str(val)
            except Exception as e:
                self._node.get_logger().error(f'SpeakBehaviour: BB text ошибка: {e}')
                self._done.set()
                return
            try:
                if self._voice_bb_key:
                    val = self._bb
                    for part in self._voice_bb_attr.split('.'):
                        val = getattr(val, part)
                    voice = str(val)
            except Exception:
                voice = ''

        if not text or not text.strip():
            self._success = True
            self._done.set()
            return

        if not self._client.wait_for_server(timeout_sec=1.0):
            self._node.get_logger().error('SpeakBehaviour: TTS Action Server недоступен')
            self._done.set()
            return

        goal = Speak.Goal()
        goal.text  = text
        goal.voice = voice
        future = self._client.send_goal_async(goal)
        future.add_done_callback(self._goal_accepted_cb)

    def _goal_accepted_cb(self, future):
        gh = future.result()
        if not gh.accepted:
            self._node.get_logger().warn('SpeakBehaviour: goal отклонён')
            self._done.set()
            return
        self._goal_handle = gh
        gh.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future):
        self._success = future.result().result.success
        self._done.set()

    def update(self) -> py_trees.common.Status:
        if not self._done.is_set():
            return py_trees.common.Status.RUNNING
        return (py_trees.common.Status.SUCCESS if self._success
                else py_trees.common.Status.FAILURE)

    def terminate(self, new_status: py_trees.common.Status):
        if (new_status == py_trees.common.Status.INVALID
                and self._goal_handle is not None):
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None


class SetSleepMode(py_trees.behaviour.Behaviour):
    """Активирует спящий режим: публикует /robot_sleep True (latched)."""

    def __init__(self, node: Node):
        super().__init__('SetSleepMode')
        self._node = node
        lqos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._pub = node.create_publisher(Bool, '/robot_sleep', lqos)
        self._bb  = py_trees.blackboard.Client(name='SetSleep')
        self._bb.register_key(key='/robot/sleep',
                               access=py_trees.common.Access.WRITE)
        self._bb.register_key(key='/robot/sleep_requested',
                               access=py_trees.common.Access.WRITE)

    def initialise(self):
        self._node.enable_face_detection(False)
        self._node.enable_head_tracker(False)
        msg = Bool()
        msg.data = True
        self._pub.publish(msg)
        self._bb.robot.sleep           = True
        self._bb.robot.sleep_requested = False
        self._node.get_logger().info('Спящий режим активирован — face_detection и head_tracker выключены')

    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS


class WebSearchBehaviour(py_trees.behaviour.Behaviour):
    """Асинхронный поиск через Tavily. RUNNING до завершения."""

    def __init__(self, node: Node, api_key: str):
        super().__init__('WebSearch')
        self._node       = node
        self._api_key    = api_key
        self._result_pub = node.create_publisher(String, 'search_result', 10)
        self._bb         = py_trees.blackboard.Client(name='WebSearch')
        self._bb.register_key(key='/search/query',
                               access=py_trees.common.Access.READ)
        self._bb.register_key(key='/search/query',
                               access=py_trees.common.Access.WRITE)
        self._bb.register_key(key='/search/result',
                               access=py_trees.common.Access.WRITE)
        self._done    = threading.Event()
        self._snippet = ''
        self._success = False

    def initialise(self):
        self._done.clear()
        self._snippet = ''
        self._success = False
        query = self._bb.search.query
        self._bb.search.query = ''  # очищаем сразу — предотвращает повторный запуск
        threading.Thread(target=self._search, args=(query,), daemon=True).start()

    def _search(self, query: str):
        try:
            self._node.get_logger().info(f'Поиск (Tavily): "{query}"')
            r = requests.post(
                'https://api.tavily.com/search',
                json={
                    'api_key':        self._api_key,
                    'query':          query,
                    'search_depth':   'basic',
                    'max_results':    3,
                    'include_answer': True,
                },
                timeout=20.0,
            )
            r.raise_for_status()
            data    = r.json()
            answer  = data.get('answer', '')
            results = data.get('results', [])
            self._snippet = (
                answer.strip() if answer else
                (results[0]['content'].strip()[:500] if results else 'Ничего не найдено')
            )
            self._success = True
            # Публикуем сразу из потока — не ждём следующего тика BT
            self._bb.search.result = {'snippet': self._snippet}
            msg = String()
            msg.data = self._snippet
            self._result_pub.publish(msg)
            self._node.get_logger().info(
                f'Поиск завершён: {len(self._snippet)} символов → опубликовано')
        except Exception as e:
            self._node.get_logger().error(f'Ошибка поиска Tavily: {e}')
        finally:
            self._done.set()

    def update(self) -> py_trees.common.Status:
        if not self._done.is_set():
            return py_trees.common.Status.RUNNING
        return py_trees.common.Status.SUCCESS if self._success else py_trees.common.Status.FAILURE


class GesticulationAction(py_trees.behaviour.Behaviour):
    """Параллельный жест во время речи. Публикует arm_command на основе эмоции/контекста.

    Пока реализован как stub — всегда SUCCESS.
    Будущая версия: анализировать /llm/emotion и выбирать подходящий жест.
    """

    def __init__(self, node: Node):
        super().__init__('Gesticulate')
        self._node    = node
        self._pub_arm = node.create_publisher(String, 'arm_command', 10)
        self._bb      = py_trees.blackboard.Client(name='Gesticulate')
        self._bb.register_key(key='/llm/emotion', access=py_trees.common.Access.READ)

    def initialise(self):
        try:
            emotion = self._bb.llm.emotion
        except KeyError:
            emotion = 'neutral'
        # TODO: маппинг эмоции → конкретный жест (grab/extend/wave/etc.)
        # Пока — нейтральная "домашняя" поза
        if emotion in ('happy', 'excited', 'smile'):
            cmd = {'command': 'wave', 'target': ''}
        else:
            cmd = {'command': 'home', 'target': ''}
        msg = String()
        msg.data = json.dumps(cmd, ensure_ascii=False)
        self._pub_arm.publish(msg)

    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS


class PIRScanBehaviour(py_trees.behaviour.Behaviour):
    """
    Поворот головы влево→вправо→центр при срабатывании PIR.
    BT — единственный оркестратор: включает face_detection для поиска.

    Если лицо найдено — SocialBranch (более высокий приоритет в Selector) прерывает
    скан через terminate(INVALID), который включает head_tracker (передаёт ему управление).
    Если лицо не найдено — _done() выключает face_detection и возвращает SUCCESS.
    """

    # rothead: rest=90, min=30, max=140 (arduino); rest_rothead=90 (vision_head_tracker)
    _CENTER   = 90.0   # смотрит вперёд (hardware rest arduino_left_node: rothead rest=90)
    _LEFT     = 120.0  # крайнее левое
    _RIGHT    = 60.0   # крайнее правое
    _NECK     = 40.0   # neck hardware rest (arduino_left_node: neck rest=40)

    # vel=1.0 rad/s → deg_per_sec_to_step(57°/s) = step=3 → ~50°/s
    # CENTER(90)→LEFT(120): 30°/50°/s≈0.6s, LEFT→RIGHT(60): 60°/50°/s≈1.2s, RIGHT→CENTER: 30°/50°/s≈0.6s
    _SCAN_VEL = 1.0

    # (name, target_rothead | None=dwell, duration_sec)
    _PHASES = (
        ('go_left',      _LEFT,   0.6),
        ('dwell_left',   None,    1.5),   # было 0.8
        ('go_right',     _RIGHT,  1.2),
        ('dwell_right',  None,    1.5),   # было 0.8
        ('go_center',    _CENTER, 0.6),
        ('dwell_center', None,    1.2),   # новый: зависание по центру
    )

    def __init__(self, node: Node):
        super().__init__('PIRScan')
        self._node      = node
        self._head_pub  = node.create_publisher(JointState, '/joint_command', 10)
        self._bb = py_trees.blackboard.Client(name='PIRScan')
        self._bb.register_key(key='/pir/scan_active',
                               access=py_trees.common.Access.WRITE)
        self._phase      = 0
        self._phase_end  = 0.0
        self._completed  = False   # True когда _done() вызван (скан завершён без лица)

    def _head_cmd(self, rothead: float) -> None:
        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name     = ['rothead', 'neck']
        msg.position = [
            (rothead     - 90.0) * math.pi / 180.0,
            (self._NECK  - 90.0) * math.pi / 180.0,
        ]
        msg.velocity = [self._SCAN_VEL, self._SCAN_VEL]
        self._head_pub.publish(msg)

    def initialise(self) -> None:
        self._phase     = 0
        self._completed = False
        _, target, dur  = self._PHASES[0]
        self._phase_end = time.monotonic() + dur
        self._node.enable_face_detection(True)
        self._head_cmd(target)
        self._node.get_logger().info('PIRScan: старт — поворот влево, face_detection включена')

    def update(self) -> py_trees.common.Status:
        if time.monotonic() < self._phase_end:
            return py_trees.common.Status.RUNNING

        self._phase += 1
        if self._phase >= len(self._PHASES):
            return self._done()

        name, target, dur = self._PHASES[self._phase]
        self._phase_end = time.monotonic() + dur
        if target is not None:
            self._head_cmd(target)
            self._node.get_logger().info(f'PIRScan: {name}')

        return py_trees.common.Status.RUNNING

    def _done(self) -> py_trees.common.Status:
        """Скан завершён — лицо не найдено. Выключаем детекцию."""
        self._completed = True
        self._node.enable_face_detection(False)
        self._bb.pir.scan_active = False
        self._node.get_logger().info('PIRScan: завершён — лицо не обнаружено, face_detection выключена')
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status == py_trees.common.Status.INVALID:
            self._bb.pir.scan_active = False
            if self._completed:
                # py_trees вызывает terminate(INVALID) при перетикивании Sequence(memory=False)
                # сразу после нашего SUCCESS — игнорируем, скан уже завершён нормально.
                return
            # Настоящий preempt: SocialBranch перехватил управление (лицо найдено).
            # Передаём голову vision_head_tracker, face_detection остаётся включённой.
            self._node.enable_head_tracker(True)
            self._node.get_logger().info('PIRScan: прерван (лицо найдено) — head_tracker включён')


class SoundScanBehaviour(py_trees.behaviour.Behaviour):
    """
    По wake word — поворот КОРПУСА (midstom), а не головы, в сторону,
    откуда пришёл голос (по /sound_direction — TDOA sign-vote, см.
    sound_localization_node / project_sound_localization_gcc_phat.md).
    Заменяет старый PIRScan-скан головой конкретно для случая "услышали
    голос" (PIR-движение без голоса по-прежнему использует PIRScanBehaviour
    — там нет направления, которое можно было бы использовать).

    Останавливается РОВНО ТАМ, где стоит, как только OAK-D увидел
    человека (/human_detected — быстрее и грубее, чем полное распознавание
    лица /social/person_present) — корпус НЕ возвращается в центр, и
    управление передаётся head_tracker (который читает face_tracker) для
    точной визуальной доводки. Если человек не найден за DWELL_TIMEOUT —
    возврат корпуса в центр, face_detection выключается, отказ (как у
    старого PIRScan).

    Знак midstom проверен руками 2026-08-22: 60°=влево, 120°=вправо —
    ОБРАТНАЯ конвенция относительно rothead (там 120=влево, 60=вправо).
    Не путать при будущих правках.
    """

    _CENTER = 90.0   # midstom rest (arduino_left_node: midstom rest=90, min=60, max=120)
    _LEFT   = 60.0    # проверено руками 2026-08-22: midstom=60° — корпус полностью ВЛЕВО
    _RIGHT  = 120.0   # проверено руками 2026-08-22: midstom=120° — корпус полностью ВПРАВО (обратная конвенция относительно rothead!)
    _TURN_VEL = 1.0

    _GO_DURATION    = 0.6   # ~30° при ~50°/с (см. _SCAN_VEL расчёт в PIRScanBehaviour)
    _DWELL_TIMEOUT  = 8.0   # сколько ждать /human_detected после поворота, прежде чем сдаться
    _MIN_CONFIDENCE = 0.15  # ниже — направление "неизвестно", не поворачиваем (остаёмся по центру)
    _MIN_ANGLE_DEG  = 15.0  # |angle_deg| меньше — тоже "около центра", не поворачиваем

    def __init__(self, node: Node):
        super().__init__('SoundScan')
        self._node      = node
        self._torso_pub = node.create_publisher(JointState, '/joint_command', 10)
        self._bb = py_trees.blackboard.Client(name='SoundScan')
        self._bb.register_key(key='/sound/scan_active',
                               access=py_trees.common.Access.WRITE)
        self._phase      = 0     # 0=едем к цели, 1=ждём human_detected, 2=возврат в центр
        self._phase_end  = 0.0
        self._completed  = False
        self._target     = self._CENTER

    def _torso_cmd(self, midstom: float) -> None:
        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name     = ['midstom']
        msg.position = [(midstom - 90.0) * math.pi / 180.0]
        msg.velocity = [self._TURN_VEL]
        self._torso_pub.publish(msg)

    def initialise(self) -> None:
        self._completed = False
        self._phase      = 0

        angle      = getattr(self._node, '_last_sound_angle', 0.0)
        confidence = getattr(self._node, '_last_sound_confidence', 0.0)

        if confidence < self._MIN_CONFIDENCE or abs(angle) < self._MIN_ANGLE_DEG:
            self._target = self._CENTER
            self._node.get_logger().info(
                f'SoundScan: направление неуверенное (angle={angle:.0f}° '
                f'conf={confidence:.2f}) — остаюсь по центру')
        elif angle > 0:
            self._target = self._RIGHT
            self._node.get_logger().info(
                f'SoundScan: голос справа (angle={angle:.0f}° conf={confidence:.2f}) '
                f'— поворот корпуса вправо')
        else:
            self._target = self._LEFT
            self._node.get_logger().info(
                f'SoundScan: голос слева (angle={angle:.0f}° conf={confidence:.2f}) '
                f'— поворот корпуса влево')

        self._node.enable_face_detection(True)
        self._torso_cmd(self._target)
        self._phase_end = time.monotonic() + self._GO_DURATION

    def update(self) -> py_trees.common.Status:
        if getattr(self._node, '_human_detected', False):
            return self._found()

        now = time.monotonic()
        if self._phase == 0:
            if now < self._phase_end:
                return py_trees.common.Status.RUNNING
            self._phase     = 1
            self._phase_end = now + self._DWELL_TIMEOUT
            self._node.get_logger().info('SoundScan: держу позицию, жду /human_detected')
            return py_trees.common.Status.RUNNING
        elif self._phase == 1:
            if now < self._phase_end:
                return py_trees.common.Status.RUNNING
            self._phase = 2
            self._torso_cmd(self._CENTER)
            self._phase_end = now + self._GO_DURATION
            self._node.get_logger().info('SoundScan: человек не найден за таймаут — возврат в центр')
            return py_trees.common.Status.RUNNING
        else:
            if now < self._phase_end:
                return py_trees.common.Status.RUNNING
            return self._done()

    def _found(self) -> py_trees.common.Status:
        self._completed = True
        self._bb.sound.scan_active = False
        self._node.enable_head_tracker(True)
        self._node.get_logger().info(
            'SoundScan: OAK-D увидел человека — останавливаюсь, передаю управление head_tracker')
        return py_trees.common.Status.SUCCESS

    def _done(self) -> py_trees.common.Status:
        self._completed = True
        self._node.enable_face_detection(False)
        self._bb.sound.scan_active = False
        self._node.get_logger().info('SoundScan: завершён — человек не найден, face_detection выключена')
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status == py_trees.common.Status.INVALID:
            self._bb.sound.scan_active = False
            if self._completed:
                # py_trees вызывает terminate(INVALID) при перетикивании Sequence(memory=False)
                # сразу после нашего SUCCESS — игнорируем, уже обработано в _found()/_done().
                return
            # Настоящий preempt (например, /social/person_present стал True другим путём,
            # SocialBranch перехватил раньше, чем мы сами увидели /human_detected).
            self._node.enable_head_tracker(True)
            self._node.get_logger().info('SoundScan: прерван (человек найден) — head_tracker включён')


class IdleBlinkBehaviour(py_trees.behaviour.Behaviour):
    """Моргание глаз в состоянии ожидания. Всегда RUNNING."""

    _CLOSED = {
        'eyelid_L_Upper': 70, 'eyelid_L_Lower': 75,
        'eyelid_R_Upper': 65, 'eyelid_R_Lower': 70,
    }
    _OPEN = {
        'eyelid_L_Upper': 85, 'eyelid_L_Lower': 85,
        'eyelid_R_Upper': 85, 'eyelid_R_Lower': 85,
    }
    # Firmware: step=2, SMOOTH_INTERVAL_MS=60 → default 33°/s (too slow for blink).
    # We send vel=3.0 rad/s → deg_per_sec_to_step(171°/s) = step=10 → ~170°/s.
    # Worst-case travel: 20° (R_Upper 85→65) / 170°/s ≈ 120ms to fully close.
    _BLINK_VEL_RADS = 3.0   # rad/s → firmware step≈10 via CMD_SET_SPEEDS
    _BLINK_DURATION = 0.12  # seconds — matches 20° travel at step=10 (2 firmware ticks)
    _INTERVAL_MIN   = 3.0   # мин. пауза между морганиями
    _INTERVAL_MAX   = 7.0   # макс. пауза между морганиями

    def __init__(self, node: Node, name: str = 'IdleBlink'):
        super().__init__(name)
        self._node       = node
        self._pub        = node.create_publisher(JointState, '/face_command', 10)
        self._next_blink = time.monotonic() + random.uniform(self._INTERVAL_MIN, self._INTERVAL_MAX)
        self._open_at    = None

    def _send(self, positions: dict, vel: float = 0.0) -> None:
        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name     = list(positions.keys())
        msg.position = [(float(v) - 90.0) * math.pi / 180.0 for v in positions.values()]
        if vel != 0.0:
            msg.velocity = [vel] * len(positions)
        self._pub.publish(msg)

    def initialise(self) -> None:
        self._next_blink = time.monotonic() + random.uniform(self._INTERVAL_MIN, self._INTERVAL_MAX)
        self._open_at    = None

    def update(self) -> py_trees.common.Status:
        now = time.monotonic()
        if self._open_at is not None:
            if now >= self._open_at:
                self._send(self._OPEN, vel=self._BLINK_VEL_RADS)
                self._open_at    = None
                self._next_blink = now + random.uniform(self._INTERVAL_MIN, self._INTERVAL_MAX)
        elif now >= self._next_blink:
            self._send(self._CLOSED, vel=self._BLINK_VEL_RADS)
            self._open_at = now + self._BLINK_DURATION
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status == py_trees.common.Status.INVALID and self._open_at is not None:
            self._send(self._OPEN, vel=self._BLINK_VEL_RADS)
            self._open_at    = None
            self._next_blink = time.monotonic() + random.uniform(self._INTERVAL_MIN, self._INTERVAL_MAX)


# ══════════════════════════════════════════════════════════════════════════════
# ПОСТРОЕНИЕ ВЕЧНОГО ДЕРЕВА
# ══════════════════════════════════════════════════════════════════════════════

def build_tree(node: Node, tavily_key: str) -> py_trees.behaviour.Behaviour:
    """
    Строит единое вечное дерево приоритетов.

    Root (Selector, no-memory) — каждый тик проверяет с начала:
      1. SleepTransition  — переход в сон по запросу
      2. SleepActive      — блокировка пока спит
      3. RobotCommand     — выполнение физ. команды от LLM
      4. WebSearch        — поиск в интернете
      5. SocialBranch     — социальное взаимодействие (gate: person_present)
      6. FarewellBranch   — прощание когда человек ушёл
      7a. SoundScanBranch — wake word: поворот корпуса (midstom) на голос до /human_detected
      7b. PIRScanBranch   — чистое PIR-движение (без голоса): скан головой
      8. GlobalIdle       — ожидание

    Interrupt Buffer: SocialBranch — Sequence(no-memory):
      CheckPersonPresent → FAILURE если человек ушёл → Sequence прерывается
      → terminate(INVALID) вызывается на SpeakBehaviour → отменяет TTS goal.
    """

    # ── 1. Sleep transition ───────────────────────────────────────────────
    sleep_transition = py_trees.composites.Sequence(
        'SleepTransition', memory=False, children=[
            CheckBB('IsSleepRequested', '/robot/sleep_requested',
                    check_fn=lambda v: v is True),
            py_trees.composites.Sequence('FarewellAndSleep', memory=True, children=[
                ExpressEmotion(node, emotion='sleeping'),
                SpeakBehaviour(node, bb_key='/robot/sleep_text'),
                SetSleepMode(node),
            ]),
        ]
    )

    # ── 2. Sleep active block ─────────────────────────────────────────────
    sleep_active = py_trees.composites.Sequence(
        'SleepActive', memory=False, children=[
            CheckBB('IsAsleep', '/robot/sleep', check_fn=lambda v: v is True),
            AlwaysRunning('SleepBlock'),
        ]
    )

    # ── 3. Robot command (move/arm/head/status) ───────────────────────────
    robot_command = py_trees.composites.Sequence(
        'RobotCommand', memory=False, children=[
            CheckBB('HasCommand', '/robot/command',
                    check_fn=lambda v: bool(v) and bool(v.get('action'))),
            ExecuteRobotCommand(node),
        ]
    )

    # ── 4. Web search ─────────────────────────────────────────────────────
    web_search = py_trees.composites.Sequence(
        'WebSearch', memory=False, children=[
            CheckBB('HasQuery', '/search/query', check_fn=lambda v: bool(v)),
            WebSearchBehaviour(node, tavily_key),
        ]
    )

    # ── 5. Social branch ──────────────────────────────────────────────────
    #
    # Структура знакомства с новым человеком:
    #   IntroducingBlock (no-memory Sequence):
    #     - Проверяет флаг introducing (режим сбора имени)
    #     - SpeakOrSkip: если есть pending фраза — произносит её (memory=True inner),
    #       иначе пропускает (Success fallback)
    #     - AlwaysRunning: блокирует пока introducing=True
    #
    # Когда introducing становится False → IsIntroducing FAILURE → Sequence FAILURE
    # → SocialSelector переходит к следующей ветке.

    speak_if_pending = py_trees.composites.Sequence(
        'SpeakIntroIfPending', memory=True, children=[
            CheckBB('HasIntroText', '/social/introduce_pending',
                    check_fn=lambda v: v is True),
            SetBB('ClearIntroPending', '/social/introduce_pending', False),
            SpeakBehaviour(node, bb_key='/social/introduce_text'),
        ]
    )
    speak_or_skip = py_trees.composites.Selector(
        'SpeakOrSkip', memory=False, children=[
            speak_if_pending,
            py_trees.behaviours.Success(name='NoPendingIntro'),
        ]
    )
    introducing_block = py_trees.composites.Sequence(
        'IntroducingBlock', memory=False, children=[
            CheckBB('IsIntroducing', '/social/introducing',
                    check_fn=lambda v: v is True),
            speak_or_skip,
            AlwaysRunning('WaitIntroduce'),
        ]
    )

    # Приветствие: только один раз (should_greet → True → BT приветствует → сбрасывает флаг)
    greet_branch = py_trees.composites.Sequence(
        'GreetBranch', memory=True, children=[
            CheckBB('ShouldGreet', '/social/should_greet',
                    check_fn=lambda v: v is True),
            py_trees.composites.Parallel(
                'GreetParallel',
                policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
                children=[
                    SpeakBehaviour(node, bb_key='/social/greet_text'),
                    ExpressEmotion(node, emotion='happy'),
                ]
            ),
            SetBB('ClearGreet', '/social/should_greet', False),
        ]
    )

    # Диалоговая ветка: BT оркеструет речь + мимику + жест параллельно.
    # Запускается когда LLM положил текст в BB через /llm_response.
    # Прерывается автоматически если человек уйдёт (SocialBranch gate выше).
    dialogue_branch = py_trees.composites.Sequence(
        'DialogueBranch', memory=True, children=[
            # has_content=True при обычном ответе (text непустой) и при стриминге (text='')
            CheckBB('HasLLMContent', '/llm/has_content', check_fn=lambda v: v is True),
            py_trees.composites.Parallel(
                'DialogueParallel',
                policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
                children=[
                    # При streamed=True text='' → SpeakBehaviour сразу SUCCESS (без TTS-цели)
                    SpeakBehaviour(node,
                                   bb_key='/llm/text',
                                   voice_bb_key='/llm/voice_style'),
                    ExpressEmotion(node, bb_key='/llm/emotion'),
                    GesticulationAction(node),
                ]
            ),
            # Очищаем BB после завершения — чтобы не повторять ответ
            SetBB('ClearLLMText',       '/llm/text',        ''),
            SetBB('ClearLLMEmo',        '/llm/emotion',     'neutral'),
            SetBB('ClearLLMVoice',      '/llm/voice_style', ''),
            SetBB('ClearLLMContent',    '/llm/has_content', False),
        ]
    )

    social_selector = py_trees.composites.Selector(
        'SocialSelector', memory=False, children=[
            introducing_block,
            greet_branch,
            dialogue_branch,
            IdleBlinkBehaviour(node, 'IdleGaze'),  # слежение глазами + моргание
        ]
    )

    # Gate: Sequence(no-memory) — если человек уходит, CheckPersonPresent FAILURE
    # → вся ветка прерывается → terminate(INVALID) на RUNNING SpeakBehaviour → отмена TTS
    social_branch = py_trees.composites.Sequence(
        'SocialBranch', memory=False, children=[
            CheckBB('IsPersonPresent', '/social/person_present',
                    check_fn=lambda v: v is True),
            social_selector,
        ]
    )

    # ── 6. Farewell branch (человек только что ушёл) ─────────────────────
    farewell_branch = py_trees.composites.Sequence(
        'FarewellBranch', memory=True, children=[
            CheckBB('FarewellPending', '/social/farewell_pending',
                    check_fn=lambda v: v is True),
            SetBB('ClearFarewell', '/social/farewell_pending', False),
            py_trees.composites.Parallel(
                'FarewellParallel',
                policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
                children=[
                    SpeakBehaviour(node, bb_key='/social/farewell_text'),
                    ExpressEmotion(node, emotion='neutral'),
                ]
            ),
        ]
    )

    # ── 7a. Sound scan (wake word → поворот КОРПУСА на голос) ─────────────
    sound_scan = py_trees.composites.Sequence(
        'SoundScanBranch', memory=False, children=[
            CheckBB('IsSoundScanActive', '/sound/scan_active',
                    check_fn=lambda v: v is True),
            SoundScanBehaviour(node),
        ]
    )

    # ── 7b. PIR scan (чистое движение без голоса — поиск лица головой) ────
    pir_scan = py_trees.composites.Sequence(
        'PIRScanBranch', memory=False, children=[
            CheckBB('IsPIRScanActive', '/pir/scan_active',
                    check_fn=lambda v: v is True),
            PIRScanBehaviour(node),
        ]
    )

    # ── 8. Global idle ────────────────────────────────────────────────────
    global_idle = IdleBlinkBehaviour(node, 'GlobalIdle')

    # ── Root ──────────────────────────────────────────────────────────────
    root = py_trees.composites.Selector(
        'Root', memory=False, children=[
            sleep_transition,
            sleep_active,
            robot_command,
            web_search,
            social_branch,
            farewell_branch,
            sound_scan,
            pir_scan,
            global_idle,
        ]
    )
    return root


# ══════════════════════════════════════════════════════════════════════════════
# BEHAVIOR MANAGER НОДА
# ══════════════════════════════════════════════════════════════════════════════

class BehaviorManagerNode(LifecycleNode):
    def __init__(self):
        super().__init__('behavior_manager_node')

        self._pir_cooldown     = 20.0  # будет перезаписан в on_configure
        self._pir_next_scan_at = 0.0
        self._pir_prev_state   = False

        # Кэш последнего /sound_direction и /human_detected для SoundScanBehaviour
        # (читает через getattr(node, ...), не через topic-подписку в самом behaviour)
        self._last_sound_angle      = 0.0
        self._last_sound_confidence = 0.0
        self._human_detected        = False

        # ── Blackboard инициализация ──────────────────────────────────────
        self._bb = py_trees.blackboard.Client(name='BehaviorManager')
        _bb_defaults = {
            '/robot/sleep':             False,
            '/robot/sleep_requested':   False,
            '/robot/sleep_text':        '',
            '/robot/command':           {},
            '/llm/text':                '',
            '/llm/voice_style':         '',
            '/llm/emotion':             'neutral',
            '/llm/has_content':         False,
            '/social/person_present':   False,
            '/social/name':             '',
            '/social/emotion':          'neutral',
            '/social/should_greet':     False,
            '/social/greet_text':       '',
            '/social/farewell_pending': False,
            '/social/farewell_text':    '',
            '/social/introducing':      False,
            '/social/introduce_pending': False,
            '/social/introduce_text':   '',
            '/search/query':            '',
            '/search/result':           {},
            '/pir/scan_active':         False,
            '/sound/scan_active':       False,
            '/scene/person_count':      0,
            '/scene/objects_summary':   '',
            '/scene/location':          '',
        }
        for key in _bb_defaults:
            self._bb.register_key(key=key, access=py_trees.common.Access.WRITE)
        self._bb.robot.sleep             = False
        self._bb.robot.sleep_requested   = False
        self._bb.robot.sleep_text        = ''
        self._bb.robot.command           = {}
        self._bb.llm.text                = ''
        self._bb.llm.voice_style         = ''
        self._bb.llm.emotion             = 'neutral'
        self._bb.llm.has_content         = False
        self._bb.social.person_present   = False
        self._bb.social.name             = ''
        self._bb.social.emotion          = 'neutral'
        self._bb.social.should_greet     = False
        self._bb.social.greet_text       = ''
        self._bb.social.farewell_pending = False
        self._bb.social.farewell_text    = ''
        self._bb.social.introducing      = False
        self._bb.social.introduce_pending = False
        self._bb.social.introduce_text   = ''
        self._bb.search.query            = ''
        self._bb.search.result           = {}
        self._bb.pir.scan_active         = False
        self._bb.sound.scan_active       = False
        self._bb.scene.person_count      = 0
        self._bb.scene.objects_summary   = ''
        self._bb.scene.location          = ''

        # Для определения перехода person_present True→False
        self._person_was_present = False

        # Таймер задержки прощания — даём TTS договорить перед person_present=False
        self._farewell_timer: threading.Timer | None = None
        self._farewell_delay_sec = 8.0  # макс. время ожидания окончания TTS

        # После say_goodbye: _social_ctx_cb НЕ должен перезаписывать person_present=False
        # из identity_manager (он в IDLE), иначе BT-gate убьёт TTS прощания.
        # _finalize_farewell() сам выставит False после таймера.
        self._suppress_social_present_until: float = 0.0  # time.monotonic()

        self._tree       = None
        self._tick_timer = None

    # ── Vision control (BT — единственный оркестратор) ───────────────────

    def enable_face_detection(self, enabled: bool) -> None:
        msg = Bool()
        msg.data = enabled
        self._face_det_pub.publish(msg)

    def enable_head_tracker(self, enabled: bool) -> None:
        msg = Bool()
        msg.data = enabled
        self._head_tracker_pub.publish(msg)

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _llm_response_cb(self, msg: String):
        """Ответ LLM → Blackboard. BT DialogueBranch увидит текст и оркеструет речь."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Невалидный /llm_response JSON: {e}')
            return

        streamed = data.get('streamed', False)
        via_telegram = data.get('telegram', False)
        self._bb.llm.text        = '' if streamed else data.get('text', '')
        self._bb.llm.voice_style = data.get('voice_instruct', '')
        self._bb.llm.emotion     = data.get('emotion', 'neutral')
        # has_content=True триггерит BT DialogueBranch для эмоции/жеста
        # (при streamed=True текст пустой, но SpeakBehaviour сразу успешен → Emotion+Gesture бегут)
        self._bb.llm.has_content = bool(
            self._bb.llm.text or streamed
        )
        if self._bb.llm.has_content and not via_telegram:
            # Голос = подтверждение присутствия: разрешаем DialogueBranch даже без лица.
            # Telegram-запросы не ставят person_present — человек физически не присутствует.
            self._bb.social.person_present = True
            self.get_logger().info('LLM ответ → person_present=True (голосовое присутствие)')
        self.get_logger().debug(
            f'LLM→BB: "{self._bb.llm.text[:60]}" '
            f'[{self._bb.llm.emotion}]'
        )

    def _event_cb(self, msg: String):
        """Robot events от LLM tool calls (move/arm/head/sleep/search)."""
        try:
            event = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Невалидный JSON: {e}')
            return

        action = event.get('action', '')
        self.get_logger().info(f'Event: {action}')

        if action in ('move', 'arm', 'head', 'status'):
            self._bb.robot.command = event
        elif action == 'sleep':
            self._bb.robot.sleep_requested = True
            self._bb.robot.sleep_text = event.get(
                'text', 'Спокойной ночи! Скажи «Эй Лёня» чтобы разбудить меня.')
        elif action == 'goodbye':
            # LLM явно попрощался. speak_text уже озвучивается TTS через llm_node —
            # FarewellBranch не нужен (иначе прощание прозвучит дважды).
            # Просто переводим IM в IDLE и запускаем таймер на выключение vision.
            self._bb.social.should_greet      = False
            self._bb.social.introducing       = False
            self._bb.social.introduce_pending = False
            # Переводим IdentityManager в IDLE (разрываем сессию немедленно)
            go_idle_msg = Bool()
            go_idle_msg.data = True
            self._go_idle_pub.publish(go_idle_msg)
            # Выключаем face_detection и head_tracker немедленно — человек прощается,
            # нельзя допустить повторного приветствия пока TTS ещё не закончил.
            self.enable_face_detection(False)
            self.enable_head_tracker(False)
            # Блокируем перезапись person_present из social_ctx до конца прощального TTS.
            # Без этого identity_manager (в IDLE) каждые 0.5с шлёт person_present=False,
            # BT-gate убивает SpeakBehaviour через terminate(INVALID) ещё до слова.
            self._suppress_social_present_until = (
                time.monotonic() + self._farewell_delay_sec + 2.0)
            # Таймер на финальную очистку после TTS
            if self._farewell_timer is not None:
                self._farewell_timer.cancel()
            self._farewell_timer = threading.Timer(
                self._farewell_delay_sec, self._finalize_farewell)
            self._farewell_timer.daemon = True
            self._farewell_timer.start()
            self.get_logger().info(
                f'say_goodbye → IDLE, face_detection выключена немедленно, '
                f'финальная очистка через {self._farewell_delay_sec:.0f}с')
        elif action == 'search':
            self._bb.search.query = event.get('query', '')
        else:
            self.get_logger().warn(f'Неизвестный action в event_cb: "{action}"')

    def _social_ctx_cb(self, msg: String):
        """Расширенный социальный контекст от IdentityManager → Blackboard."""
        try:
            ctx = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        # После say_goodbye: identity_manager в IDLE и шлёт person_present=False @ 2Гц.
        # Если записать это в BB, BT-gate убьёт TTS прощания через terminate(INVALID).
        # _finalize_farewell() сам выставит False после таймера.
        if time.monotonic() < self._suppress_social_present_until:
            pass  # не обновляем person_present — пусть остаётся True от _llm_response_cb
        else:
            self._bb.social.person_present = ctx.get('person_present', False)
        self._bb.social.name           = ctx.get('name', '')
        self._bb.social.emotion        = ctx.get('emotion', 'neutral')
        self._bb.social.introducing    = ctx.get('introducing', False)

        # should_greet — одноразовый сигнал: устанавливаем только когда IM так говорит
        if ctx.get('should_greet', False):
            self._bb.social.should_greet = True
            self._bb.social.greet_text   = ctx.get('greet_text', '')

        # introduce_pending — одноразовый сигнал: IM хочет произнести фразу знакомства
        if ctx.get('introduce_pending', False):
            self._bb.social.introduce_pending = True
            self._bb.social.introduce_text    = ctx.get('introduce_text', '')

    def _scene_ctx_cb(self, msg: String):
        """Сводка сцены (объекты + люди) от scene_manager_node → Blackboard."""
        try:
            ctx = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        self._bb.scene.person_count    = ctx.get('person_count', 0)
        self._bb.scene.location        = ctx.get('location', '')
        self._bb.scene.objects_summary = ', '.join(
            f"{o.get('label')}:{o.get('count')}" for o in ctx.get('objects', [])
        )

    def _person_present_cb(self, msg: Bool):
        """Человек ушёл → даём TTS договорить, потом прощаемся."""
        was_present = self._person_was_present
        now_present = msg.data
        self._person_was_present = now_present

        if now_present and not was_present:
            # Человек вернулся — отменяем отложенное прощание
            if self._farewell_timer is not None:
                self._farewell_timer.cancel()
                self._farewell_timer = None
            self._bb.social.person_present   = True
            self._bb.social.farewell_pending = False
            self._bb.social.farewell_text    = ''

        elif was_present and not now_present:
            # Тихий уход (таймаут / OakD вето) — прощание НЕ произносим.
            # Прощание только в ответ на явное "пока" через say_goodbye tool call.
            name = self._bb.social.name
            self._bb.social.should_greet      = False
            self._bb.social.introducing       = False
            self._bb.social.introduce_pending = False
            self.get_logger().info(
                f'Человек ушёл ({name}) — тихий IDLE (без прощания)')

            # Через farewell_delay_sec выключаем vision и завершаем диалог
            if self._farewell_timer is not None:
                self._farewell_timer.cancel()
            self._farewell_timer = threading.Timer(
                self._farewell_delay_sec, self._finalize_farewell)
            self._farewell_timer.daemon = True
            self._farewell_timer.start()

    def _finalize_farewell(self):
        """Вызывается таймером: закрываем диалог и выключаем vision."""
        self._farewell_timer = None
        self._bb.social.person_present = False
        self.enable_head_tracker(False)
        self.enable_face_detection(False)
        # Сбрасываем pending LLM — говорить уже некому
        self._bb.llm.text        = ''
        self._bb.llm.emotion     = 'neutral'
        self._bb.llm.voice_style = ''
        self.get_logger().info('Прощание: person_present=False, vision выключен')

    def _robot_sleep_cb(self, msg: Bool):
        """Синхронизируем спящий режим из latched топика."""
        self._bb.robot.sleep = msg.data
        if not msg.data:
            self._bb.robot.sleep_requested = False
            self._pir_next_scan_at = 0.0  # сбрасываем cooldown — wake-word важнее
            # Пробуждение всегда идёт через wake word (voice_detector публикует
            # /robot_sleep False по нему) — значит есть направление, используем
            # SoundScan (поворот корпуса), не старый PIR-скан головой.
            self._bb.sound.scan_active = True
            self.get_logger().info('Пробуждение (по wake word) — запуск SoundScan')

    def _sound_direction_cb(self, msg: SoundDirection):
        """Кэш последнего /sound_direction для SoundScanBehaviour."""
        self._last_sound_angle      = msg.angle_deg
        self._last_sound_confidence = msg.confidence

    def _human_detected_cb(self, msg: Bool):
        """Кэш последнего /human_detected (OAK-D) для SoundScanBehaviour."""
        self._human_detected = msg.data

    def _wake_detected_cb(self, msg: Bool):
        """Wake word в IDLE → запускаем SoundScan (поворот корпуса на голос)."""
        if not msg.data:
            return
        if self._bb.robot.sleep:
            return  # спящий режим обрабатывается через /robot_sleep False
        try:
            person_present = self._bb.social.person_present
        except Exception:
            person_present = False

        if person_present:
            # Пользователь сказал wake word пока робот приветствовал — прерывание.
            # Сбрасываем should_greet чтобы BT не зациклился на повторном приветствии.
            try:
                if self._bb.social.should_greet:
                    self._bb.social.should_greet = False
                    self._bb.social.greet_text   = ''
                    self.get_logger().info(
                        'Wake word прервал приветствие → сброс should_greet')
            except Exception:
                pass
            return

        if self._bb.pir.scan_active or self._bb.sound.scan_active:
            return
        self._pir_next_scan_at = 0.0   # wake word важнее cooldown
        self._bb.sound.scan_active = True
        self.get_logger().info('Wake word в IDLE → запуск SoundScan (поворот корпуса на голос)')

    def _pir_cb(self, msg: Bool):
        """PIR сигнал: реагируем только на передний фронт (False→True).
        Arduino шлёт heartbeat каждые 5с — повторные True при застрявшем HIGH игнорируются.
        """
        prev             = self._pir_prev_state
        self._pir_prev_state = msg.data

        if not msg.data:
            return
        if prev:
            return  # heartbeat или sustained HIGH — не новое событие движения

        # Передний фронт: новое движение обнаружено
        if self._bb.robot.sleep:
            return
        try:
            person_present = self._bb.social.person_present
        except Exception:
            person_present = False
        if person_present or self._bb.pir.scan_active:
            return
        now = time.monotonic()
        if now < self._pir_next_scan_at:
            self.get_logger().debug('PIR: движение (cooldown активен, игнорируем)')
            return
        self._bb.pir.scan_active = True
        self._pir_next_scan_at   = now + self._pir_cooldown
        self.get_logger().info('PIR: движение — запуск сканирования головой')

    # ── Тик ──────────────────────────────────────────────────────────────

    def _tick(self):
        if self._tree is not None:
            self._tree.tick_once()


    # ── Lifecycle callbacks ────────────────────────────────────────────────

    def _dp(self, name, default=None):
        """Безопасный declare_parameter: игнорирует повторное объявление при re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('tavily_api_key',        '')
        self._dp('bt_tick_rate_hz',       10.0)
        self._dp('pir_scan_cooldown_sec', 20.0)

        tavily_key          = self.get_parameter('tavily_api_key').value
        self._tick_rate     = self.get_parameter('bt_tick_rate_hz').value
        self._pir_cooldown  = self.get_parameter('pir_scan_cooldown_sec').value

        self._tree = build_tree(self, tavily_key)
        self._tree.setup_with_descendants()

        lqos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._face_det_pub     = self.create_lifecycle_publisher(Bool, '/face_detection/enable', lqos)
        self._head_tracker_pub = self.create_lifecycle_publisher(Bool, '/head_tracker/enable',   lqos)
        self._go_idle_pub      = self.create_lifecycle_publisher(Bool, '/go_idle', 10)

        self.create_subscription(String, '/llm_response',   self._llm_response_cb,   10)
        self.create_subscription(String, 'robot_events',    self._event_cb,           10)
        self.create_subscription(String, '/social_context', self._social_ctx_cb,      10)
        self.create_subscription(String, '/scene/objects',  self._scene_ctx_cb,       10)
        self.create_subscription(Bool,   '/person_present', self._person_present_cb,  10)
        self.create_subscription(Bool,   '/pir_state',      self._pir_cb,             10)
        self.create_subscription(Bool,   'wake_detected',   self._wake_detected_cb,   10)
        self.create_subscription(Bool,   '/robot_sleep',    self._robot_sleep_cb,     lqos)
        self.create_subscription(SoundDirection, '/sound_direction', self._sound_direction_cb, 10)
        self.create_subscription(Bool,   '/human_detected', self._human_detected_cb,  10)

        self.get_logger().info('BehaviorManager настроен')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._face_det_pub.on_activate(state)
        self._head_tracker_pub.on_activate(state)
        self._go_idle_pub.on_activate(state)
        self.enable_face_detection(False)
        self.enable_head_tracker(False)
        self._tick_timer = self.create_timer(1.0 / self._tick_rate, self._tick)
        self.get_logger().info(
            f'BehaviorManager v2 готов (вечное дерево @ {self._tick_rate:.0f} Гц). '
            f'Tavily: {"настроен" if self.get_parameter("tavily_api_key").value else "не настроен"}'
        )
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        if self._tick_timer:
            self.destroy_timer(self._tick_timer)
            self._tick_timer = None
        self._face_det_pub.on_deactivate(state)
        self._head_tracker_pub.on_deactivate(state)
        self._go_idle_pub.on_deactivate(state)
        self._bb.robot.command       = {}
        self._bb.llm.has_content     = False
        self._bb.social.person_present = False
        self._bb.pir.scan_active     = False
        self._bb.sound.scan_active   = False
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        if self._tick_timer:
            self.destroy_timer(self._tick_timer)
            self._tick_timer = None
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        if self._tick_timer:
            self.destroy_timer(self._tick_timer)
            self._tick_timer = None
        return TransitionCallbackReturn.SUCCESS


def main():
    rclpy.init()
    node = BehaviorManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
