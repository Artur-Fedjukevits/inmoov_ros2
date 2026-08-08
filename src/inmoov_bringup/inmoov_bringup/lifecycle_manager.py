#!/usr/bin/env python3
"""
InmoovLifecycleManager — тир-based lifecycle orchestrator для InMoov Robot.

Поднимает ноды по тирам (0..6) последовательно, внутри тира параллельно.
При отказе: 3 попытки, потом degraded (если нода не критична) или ABORT.

Watchdog: периодически опрашивает get_state всех нод. При смерти процесса:
  - помечает degraded_reason='watchdog'
  - если нода критична — каскадно деактивирует все тиры выше
  - при respawn (ros2 launch respawn=True) — автоматически re-активирует ноду
    и каскадно поднимает зависимые тиры обратно

Управление через топик /lifecycle/command:
  ACTIVATE        — поднять всю систему (автоматически при старте)
  DEACTIVATE      — деактивировать всё кроме Foundation
  SLEEP           — деактивировать vision + processing, оставить voice/LLM
  WAKE            — реактивировать после SLEEP
  SHUTDOWN        — shutdown всех нод
  RESTART_TIER N  — перезапустить конкретный тир

Статус публикуется в /lifecycle/status (JSON), включая degraded_nodes[].
"""

import json
import time
import threading
import yaml
from typing import Optional

import rclpy
from rclpy.node import Node
from lifecycle_msgs.msg import Transition, State
from lifecycle_msgs.srv import ChangeState, GetState
from std_msgs.msg import String, Bool
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy


# Переходы lifecycle_msgs
_CONFIGURE  = Transition.TRANSITION_CONFIGURE
_ACTIVATE   = Transition.TRANSITION_ACTIVATE
_DEACTIVATE = Transition.TRANSITION_DEACTIVATE
_CLEANUP    = Transition.TRANSITION_CLEANUP
_SHUTDOWN   = Transition.TRANSITION_UNCONFIGURED_SHUTDOWN

# Ожидаемые состояния
_STATE_INACTIVE     = State.PRIMARY_STATE_INACTIVE
_STATE_ACTIVE       = State.PRIMARY_STATE_ACTIVE
_STATE_UNCONFIGURED = State.PRIMARY_STATE_UNCONFIGURED

# Ноды деактивируемые при SLEEP (vision + VAD)
_SLEEP_DEACTIVATE = {
    'face_capture_node', 'face_detection_node_left', 'face_detection_node_right',
    'face_tracker_node_left', 'face_tracker_node_right',
    'face_recognition_node', 'face_gallery_node',
    'emotion_recognition_node', 'vision_head_tracker_node',
    'human_detection_node', 'oak_node',
    'voice_detector_node',
}

# Причины деградации
_REASON_ACTIVATION = 'activation'   # провал при первичной активации
_REASON_WATCHDOG   = 'watchdog'     # процесс умер (SIGKILL, OOM, segfault)
_REASON_CASCADE    = 'cascade'      # каскадная деактивация из-за гибели ноды ниже


class ManagedNode:
    """Состояние одной управляемой ноды."""

    def __init__(self, name: str, critical: bool):
        self.name = name
        self.critical = critical
        self.degraded = False
        self.degraded_reason = ''   # _REASON_* константа
        self.respawn_count = 0      # сколько раз watchdog поднимал ноду
        self.recovering = False     # recovery сейчас выполняется
        self.attempts = 0
        self._change_state_cli = None
        self._get_state_cli = None

    def __repr__(self):
        status = f'DEGRADED({self.degraded_reason})' if self.degraded else 'ok'
        crit = 'CRIT' if self.critical else 'opt'
        return f'{self.name}[{crit},{status}]'


class InmoovLifecycleManager(Node):

    def __init__(self):
        super().__init__('lifecycle_manager')

        # Параметры
        self.declare_parameter('retry_count',                3)
        self.declare_parameter('retry_interval_sec',         10.0)
        self.declare_parameter('transition_timeout_sec',     30.0)
        self.declare_parameter('tier_advance_timeout_sec',   90.0)
        self.declare_parameter('config_file',                '')
        self.declare_parameter('autostart_delay_sec',        5.0)
        self.declare_parameter('watchdog_interval_sec',      5.0)
        self.declare_parameter('watchdog_startup_delay_sec', 15.0)
        self.declare_parameter('max_respawn_count',          5)

        self._retry_count             = self.get_parameter('retry_count').value
        self._retry_interval          = self.get_parameter('retry_interval_sec').value
        self._trans_timeout           = self.get_parameter('transition_timeout_sec').value
        self._tier_timeout            = self.get_parameter('tier_advance_timeout_sec').value
        self._watchdog_interval       = self.get_parameter('watchdog_interval_sec').value
        self._watchdog_startup_delay  = self.get_parameter('watchdog_startup_delay_sec').value
        self._max_respawn_count       = self.get_parameter('max_respawn_count').value
        autostart_delay               = self.get_parameter('autostart_delay_sec').value

        # Публикация статуса
        self._status_pub = self.create_publisher(String, '/lifecycle/status', 10)

        # Команды управления
        self._cmd_sub = self.create_subscription(
            String, '/lifecycle/command', self._command_cb, 10)

        # Подписка на robot_sleep (latched)
        _latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._sleep_sub = self.create_subscription(
            Bool, '/robot_sleep', self._sleep_cb, _latched)

        # Состояние системы
        self._tiers: list[list[ManagedNode]] = []
        self._all_nodes: dict[str, ManagedNode] = {}
        self._system_state = 'starting'
        self._sleep_mode = False
        self._lock = threading.Lock()
        self._startup_timer = None
        self._watchdog_stop = threading.Event()

        # Таймер публикации статуса
        self._status_timer = self.create_timer(2.0, self._publish_status)

        config_file = self.get_parameter('config_file').value
        if config_file:
            self._load_config_from_yaml(config_file)

        if self._tiers and autostart_delay > 0:
            self._startup_timer = self.create_timer(autostart_delay, self._autostart_cb)
            self.get_logger().info(
                f'InmoovLifecycleManager ready — автозапуск через {autostart_delay:.0f}с '
                f'({len(self._tiers)} тиров, {len(self._all_nodes)} нод)')
        else:
            self.get_logger().info('InmoovLifecycleManager ready — ожидаю конфигурацию тиров')

    # ──────────────────────────────────────────────────────────────────────────
    # Загрузка конфига

    def _load_config_from_yaml(self, config_file: str):
        try:
            with open(config_file) as f:
                cfg = yaml.safe_load(f)
            params = cfg.get('lifecycle_manager', {}).get('ros__parameters', {})

            for key in ('retry_count', 'retry_interval_sec', 'transition_timeout_sec',
                        'tier_advance_timeout_sec', 'watchdog_interval_sec',
                        'watchdog_startup_delay_sec', 'max_respawn_count'):
                if key in params:
                    setattr(self, f'_{key}', params[key])

            tiers_cfg = params.get('tiers', [])
            self.configure_tiers(tiers_cfg)
            self.get_logger().info(
                f'Конфиг загружен из {config_file}: '
                f'{len(self._tiers)} тиров, {len(self._all_nodes)} нод')
        except Exception as e:
            self.get_logger().error(f'Ошибка загрузки конфига {config_file}: {e}')

    def _autostart_cb(self):
        self.destroy_timer(self._startup_timer)
        self._startup_timer = None
        self.start_activation()

    # ──────────────────────────────────────────────────────────────────────────
    # Публичный API

    def configure_tiers(self, tiers_cfg: list[dict]):
        """Инициализировать тиры из конфига. Сразу создаёт все сервисные клиенты."""
        for tier_cfg in tiers_cfg:
            tier_nodes = []
            critical_set = set(tier_cfg.get('critical', []))
            for node_name in tier_cfg['nodes']:
                mn = ManagedNode(node_name, critical=(node_name in critical_set))
                self._all_nodes[node_name] = mn
                tier_nodes.append(mn)
            self._tiers.append(tier_nodes)

        # Предсоздаём клиенты из главного потока — thread-safe для create_client
        for mn in self._all_nodes.values():
            self._get_change_state_client(mn.name)
            self._get_get_state_client(mn.name)

        self.get_logger().info(
            f'Конфигурация тиров: {len(self._tiers)} тиров, {len(self._all_nodes)} нод')

    def start_activation(self):
        threading.Thread(target=self._activate_all, daemon=True).start()

    # ──────────────────────────────────────────────────────────────────────────
    # Сервисные клиенты

    def _get_change_state_client(self, node_name: str):
        mn = self._all_nodes[node_name]
        if mn._change_state_cli is None:
            mn._change_state_cli = self.create_client(
                ChangeState, f'/{node_name}/change_state')
        return mn._change_state_cli

    def _get_get_state_client(self, node_name: str):
        mn = self._all_nodes[node_name]
        if mn._get_state_cli is None:
            mn._get_state_cli = self.create_client(
                GetState, f'/{node_name}/get_state')
        return mn._get_state_cli

    # ──────────────────────────────────────────────────────────────────────────
    # Переходы

    def _send_transition(self, node_name: str, transition_id: int) -> bool:
        cli = self._get_change_state_client(node_name)
        if not cli.wait_for_service(timeout_sec=self._trans_timeout):
            self.get_logger().error(f'{node_name}: сервис change_state недоступен')
            return False
        req = ChangeState.Request()
        req.transition.id = transition_id
        future = cli.call_async(req)
        deadline = time.monotonic() + self._trans_timeout
        while not future.done():
            time.sleep(0.05)
            if time.monotonic() > deadline:
                self.get_logger().error(f'{node_name}: таймаут перехода {transition_id}')
                return False
        return future.result() is not None and future.result().success

    def _get_state(self, node_name: str, timeout_sec: float = 5.0) -> Optional[int]:
        cli = self._get_get_state_client(node_name)
        if not cli.wait_for_service(timeout_sec=timeout_sec):
            return None
        future = cli.call_async(GetState.Request())
        deadline = time.monotonic() + timeout_sec
        while not future.done():
            time.sleep(0.05)
            if time.monotonic() > deadline:
                return None
        if future.result() is None:
            return None
        return future.result().current_state.id

    def _configure_and_activate(self, mn: ManagedNode) -> bool:
        state = self._get_state(mn.name)
        if state == _STATE_ACTIVE:
            return True
        if state != _STATE_INACTIVE:
            if not self._send_transition(mn.name, _CONFIGURE):
                return False
        if not self._send_transition(mn.name, _ACTIVATE):
            return False
        return self._get_state(mn.name) == _STATE_ACTIVE

    def _activate_node_with_retry(self, mn: ManagedNode) -> bool:
        for attempt in range(1, self._retry_count + 1):
            mn.attempts = attempt
            self.get_logger().info(f'{mn.name}: попытка {attempt}/{self._retry_count}...')
            if self._configure_and_activate(mn):
                self.get_logger().info(f'{mn.name}: ACTIVE ✓')
                return True
            if attempt < self._retry_count:
                wait = self._retry_interval * attempt
                self.get_logger().warn(f'{mn.name}: провал, жду {wait:.0f}с...')
                time.sleep(wait)
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # Активация системы

    def _activate_all(self):
        self._system_state = 'starting'
        self._publish_status()

        for tier_idx, tier in enumerate(self._tiers):
            self.get_logger().info(f'═══ Тир {tier_idx}: активация {len(tier)} нод ═══')
            if not self._activate_tier(tier_idx, tier):
                self.get_logger().error(
                    f'Тир {tier_idx}: критическая нода провалила активацию — ABORT')
                self._system_state = 'fault'
                self._publish_status()
                return

        self._system_state = 'active'
        self.get_logger().info('══ Система полностью активирована ══')
        self._publish_status()

        # Запускаем watchdog после полной активации + startup_delay
        if self._watchdog_interval > 0:
            threading.Thread(target=self._watchdog_start, daemon=True).start()

    def _activate_tier(self, tier_idx: int, tier: list[ManagedNode]) -> bool:
        """Параллельная активация нод тира. False = critical failure."""
        threads = []

        def activate_one(mn):
            try:
                ok = self._activate_node_with_retry(mn)
            except Exception as e:
                self.get_logger().error(f'{mn.name}: исключение при активации: {e}')
                ok = False
            if not ok:
                mn.degraded = True
                mn.degraded_reason = _REASON_ACTIVATION

        for mn in tier:
            t = threading.Thread(target=activate_one, args=(mn,), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=self._tier_timeout)

        degraded = [mn for mn in tier if mn.degraded]
        active   = [mn for mn in tier if not mn.degraded]
        if degraded:
            self.get_logger().warn(f'Тир {tier_idx} DEGRADED: {[mn.name for mn in degraded]}')
        if active:
            self.get_logger().info(f'Тир {tier_idx}: {len(active)}/{len(tier)} нод активны')
        for mn in degraded:
            if mn.critical:
                self.get_logger().error(f'CRITICAL нода провалилась: {mn.name}')
                self._publish_status()
                return False
        self._publish_status()
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Watchdog

    def _watchdog_start(self):
        """Ждёт startup_delay, затем запускает watchdog-цикл."""
        self.get_logger().info(
            f'Watchdog стартует через {self._watchdog_startup_delay:.0f}с '
            f'(интервал {self._watchdog_interval:.0f}с, max_respawn={self._max_respawn_count})')
        self._watchdog_stop.wait(timeout=self._watchdog_startup_delay)
        if self._watchdog_stop.is_set():
            return
        self.get_logger().info('Watchdog активен')
        while not self._watchdog_stop.is_set():
            self._watchdog_check_all()
            self._watchdog_stop.wait(timeout=self._watchdog_interval)

    def _watchdog_check_all(self):
        """Опрашивает все ноды; реагирует на смерть и воскрешение.

        Детектируем два сценария:
          A. state is None        → процесс умер, respawn ещё не случился
          B. state == UNCONFIGURED → процесс умер + ros2 launch уже respawn'ул,
                                     нода жива но не активирована (быстрый respawn_delay)
        """
        for tier_idx, tier in enumerate(self._tiers):
            for mn in tier:
                # Не трогаем ноды деградировавшие по другим причинам
                if mn.degraded and mn.degraded_reason != _REASON_WATCHDOG:
                    continue

                state = self._get_state(mn.name, timeout_sec=2.0)

                if not mn.degraded:
                    # Нода должна быть ACTIVE — любое другое состояние подозрительно
                    if state is None or state == _STATE_UNCONFIGURED:
                        cause = 'не отвечает' if state is None else 'в UNCONFIGURED (crash+respawn)'
                        self.get_logger().error(f'Watchdog: {mn.name} {cause} — degraded!')
                        mn.degraded = True
                        mn.degraded_reason = _REASON_WATCHDOG
                        self._publish_status()
                        if mn.critical:
                            threading.Thread(
                                target=self._cascade_deactivate,
                                args=(tier_idx,), daemon=True).start()
                        # Сценарий B: процесс уже жив → сразу запускаем recovery
                        if state == _STATE_UNCONFIGURED and not mn.recovering:
                            self._start_recovery(mn, tier_idx)

                elif mn.degraded_reason == _REASON_WATCHDOG and not mn.recovering:
                    # Сценарий A: ждали respawn — проверяем что нода наконец появилась
                    if state is not None:
                        self._start_recovery(mn, tier_idx)

    def _start_recovery(self, mn: ManagedNode, tier_idx: int):
        """Запускает recovery в фоне если лимит не исчерпан."""
        if mn.respawn_count >= self._max_respawn_count:
            self.get_logger().error(
                f'Watchdog: {mn.name} достиг max_respawn_count={self._max_respawn_count}')
            return
        mn.respawn_count += 1
        self.get_logger().info(
            f'Watchdog: {mn.name} запускаю recovery '
            f'(попытка {mn.respawn_count}/{self._max_respawn_count})')
        threading.Thread(
            target=self._recover_node, args=(mn, tier_idx), daemon=True).start()

    def _cascade_deactivate(self, failed_tier_idx: int):
        """Деактивирует все тиры выше failed_tier_idx (они зависели от упавшей ноды)."""
        self.get_logger().warn(
            f'Каскадная деактивация: тиры {failed_tier_idx+1}…{len(self._tiers)-1}')
        for tier_idx in range(len(self._tiers) - 1, failed_tier_idx, -1):
            tier = self._tiers[tier_idx]
            active = [mn for mn in tier if not mn.degraded]
            if not active:
                continue
            threads = [
                threading.Thread(
                    target=self._send_transition,
                    args=(mn.name, _DEACTIVATE),
                    daemon=True,
                )
                for mn in active
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=self._trans_timeout)
            for mn in active:
                mn.degraded = True
                mn.degraded_reason = f'{_REASON_CASCADE}_{failed_tier_idx}'
        self._system_state = 'degraded'
        self._publish_status()

    def _recover_node(self, mn: ManagedNode, tier_idx: int):
        """Re-активирует воскресшую ноду; при успехе — каскадно поднимает зависимые тиры."""
        mn.recovering = True
        mn.degraded = True  # удерживаем degraded на время попытки
        try:
            ok = self._activate_node_with_retry(mn)
        finally:
            mn.recovering = False
        if ok:
            mn.degraded = False
            mn.degraded_reason = ''
            self.get_logger().info(f'Watchdog: {mn.name} успешно восстановлен ✓')
            if mn.critical:
                # Ждём завершения _cascade_deactivate (он стартует параллельно).
                # Без паузы _cascade_recover может запуститься до того, как все
                # тиры помечены 'cascade_N', и пропустит их через ранний break.
                time.sleep(3.0)
                self._cascade_recover(tier_idx)
        else:
            mn.degraded = True
            mn.degraded_reason = _REASON_WATCHDOG
            self.get_logger().error(f'Watchdog: {mn.name} не удалось восстановить')
        self._publish_status()

    def _cascade_recover(self, recovered_tier_idx: int):
        """После восстановления критической ноды — поднимает каскадно деградировавшие тиры."""
        cascade_prefix = f'{_REASON_CASCADE}_{recovered_tier_idx}'
        for tier_idx in range(recovered_tier_idx + 1, len(self._tiers)):
            tier = self._tiers[tier_idx]
            to_recover = [mn for mn in tier if mn.degraded_reason == cascade_prefix]
            if not to_recover:
                # Этот тир не был каскадно деградирован — пропускаем, но идём дальше.
                # (не break — из-за race с cascade_deactivate некоторые тиры могут
                #  быть помечены cascade_N позже, чем мы проверяем первый тир)
                continue
            self.get_logger().info(
                f'Каскадное восстановление тира {tier_idx} ({len(to_recover)} нод)...')
            for mn in to_recover:
                mn.degraded = False
                mn.degraded_reason = ''
                mn.attempts = 0
            self._activate_tier(tier_idx, to_recover)
            self._publish_status()

        # Если все ноды восстановлены — возвращаем system_state в 'active'
        all_ok = all(not mn.degraded for tier in self._tiers for mn in tier)
        if all_ok:
            self._system_state = 'active'
            self.get_logger().info('Watchdog: полное восстановление системы ✓')
            self._publish_status()

    # ──────────────────────────────────────────────────────────────────────────
    # SLEEP / WAKE

    def _sleep_cb(self, msg: Bool):
        if msg.data and not self._sleep_mode:
            self.get_logger().info('robot_sleep → SLEEP: деактивирую vision')
            self._sleep_mode = True
            threading.Thread(target=self._do_sleep, daemon=True).start()
        elif not msg.data and self._sleep_mode:
            self.get_logger().info('robot_sleep → WAKE: реактивирую vision')
            self._sleep_mode = False
            threading.Thread(target=self._do_wake, daemon=True).start()

    def _do_sleep(self):
        self._system_state = 'sleep'
        for name in _SLEEP_DEACTIVATE:
            if name in self._all_nodes and not self._all_nodes[name].degraded:
                self._send_transition(name, _DEACTIVATE)
        self._publish_status()

    def _do_wake(self):
        self._system_state = 'waking'
        for name in _SLEEP_DEACTIVATE:
            if name in self._all_nodes and not self._all_nodes[name].degraded:
                self._configure_and_activate(self._all_nodes[name])
        self._system_state = 'active'
        self._publish_status()

    # ──────────────────────────────────────────────────────────────────────────
    # Команды управления

    def _command_cb(self, msg: String):
        cmd = msg.data.strip().upper()
        self.get_logger().info(f'/lifecycle/command: {cmd}')

        if cmd == 'SHUTDOWN':
            threading.Thread(target=self._do_shutdown, daemon=True).start()
        elif cmd == 'SLEEP':
            self._sleep_mode = True
            threading.Thread(target=self._do_sleep, daemon=True).start()
        elif cmd == 'WAKE':
            self._sleep_mode = False
            threading.Thread(target=self._do_wake, daemon=True).start()
        elif cmd.startswith('RESTART_TIER '):
            try:
                tier_idx = int(cmd.split()[1])
                threading.Thread(
                    target=self._restart_tier, args=(tier_idx,), daemon=True).start()
            except (IndexError, ValueError):
                self.get_logger().error(f'Неверный формат: {cmd}')
        else:
            self.get_logger().warn(f'Неизвестная команда: {cmd}')

    def _restart_tier(self, tier_idx: int):
        if tier_idx >= len(self._tiers):
            self.get_logger().error(f'Тир {tier_idx} не существует')
            return
        tier = self._tiers[tier_idx]
        for mn in tier:
            mn.degraded = False
            mn.degraded_reason = ''
            mn.attempts = 0
        self.get_logger().info(f'Перезапуск тира {tier_idx}...')
        self._activate_tier(tier_idx, tier)

    def _do_shutdown(self):
        # Останавливаем watchdog перед shutdown
        self._watchdog_stop.set()
        self._system_state = 'shutdown'
        self._publish_status()
        for tier in reversed(self._tiers):
            self._shutdown_tier(tier)

    def _shutdown_tier(self, tier: list[ManagedNode]):
        """Параллельно DEACTIVATE, потом CLEANUP+SHUTDOWN для всех нод тира."""
        active_nodes = [mn for mn in tier if not mn.degraded]
        if not active_nodes:
            return

        threads = [
            threading.Thread(
                target=self._send_transition, args=(mn.name, _DEACTIVATE), daemon=True)
            for mn in active_nodes
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=self._trans_timeout)

        def cleanup_shutdown(mn: ManagedNode):
            self._send_transition(mn.name, _CLEANUP)
            self._send_transition(mn.name, _SHUTDOWN)

        threads = [
            threading.Thread(target=cleanup_shutdown, args=(mn,), daemon=True)
            for mn in active_nodes
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=self._trans_timeout)

    # ──────────────────────────────────────────────────────────────────────────
    # Статус

    def _publish_status(self):
        degraded_nodes   = [mn.name for tier in self._tiers for mn in tier if mn.degraded]
        recovering_nodes = [mn.name for tier in self._tiers for mn in tier if mn.recovering]
        status = {
            'system':          self._system_state,
            'sleep_mode':      self._sleep_mode,
            'degraded_nodes':  degraded_nodes,
            'recovering_nodes': recovering_nodes,
            'tiers': [
                {
                    'id': tier_idx,
                    'nodes': {
                        mn.name: {
                            'critical':        mn.critical,
                            'degraded':        mn.degraded,
                            'degraded_reason': mn.degraded_reason,
                            'respawn_count':   mn.respawn_count,
                            'recovering':      mn.recovering,
                            'attempts':        mn.attempts,
                        }
                        for mn in tier
                    },
                }
                for tier_idx, tier in enumerate(self._tiers)
            ],
        }
        self._status_pub.publish(String(data=json.dumps(status, ensure_ascii=False)))


def main(args=None):
    rclpy.init(args=args)
    node = InmoovLifecycleManager()
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=16)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
