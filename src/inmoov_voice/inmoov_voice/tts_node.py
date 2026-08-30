#!/usr/bin/env python3
"""
tts_node.py — ROS2 Action Server для синтеза и воспроизведения речи.

Action: /speak (inmoov_msgs/action/Speak)
  Goal:     text, voice, rate
  Feedback: status, progress, bytes_played
  Result:   success, message, audio_sec

Особенности:
  - Preemption: новый goal мгновенно прерывает текущее воспроизведение
  - Fallback: при недоступности основного TTS сервера — локальный
  - Публикует tts_speaking (Bool) для voice_detector (обратная совместимость)

Клиенты:
  - llm_node      — текстовые ответы на голосовые команды
  - inmoov_cognition — речь как часть поведения (жесты + речь)
"""

import math
import os
import threading
import time

import numpy as np
import requests
import sounddevice as sd

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

from inmoov_msgs.action import Speak


class TTSNode(LifecycleNode):
    # Инцидент 2026-08-07: при обрыве PipeWire/ALSA (Jabra) stream.write()
    # у PortAudio уходит во внутренний C-level busy-retry XRun-цикл
    # (AlsaRestart/PaAlsaStream_HandleXrun) без сна между попытками —
    # Python-флаги (my_abort) внутри такого write() не проверяются, поэтому
    # штатно прервать это нельзя. За секунды это забивает CPU и заваливает
    # journald (наблюдалось 386к сообщений/с), из-за чего виснет вся машина.
    # Watchdog ниже — единственный надёжный выход: если после начала
    # воспроизведения нет прогресса дольше _STALL_TIMEOUT_SEC, считаем
    # процесс безнадёжно зависшим и убиваем его целиком; respawn=True
    # в launch-файле поднимет чистый процесс через respawn_delay секунд.
    _STALL_TIMEOUT_SEC = 8.0

    def __init__(self):
        super().__init__('tts_node')

        # Threading state остаётся в __init__ — не зависит от lifecycle
        self._execute_lock       = threading.Lock()
        self._abort_lock         = threading.Lock()
        self._abort_event        = threading.Event()
        self._goals_pending      = 0
        self._goals_pending_lock = threading.Lock()
        self._cancel_queued      = threading.Event()
        # Эмоция лица, показанная для текущей речи (см. _execute_speak) —
        # держится на /face_expression_hold пока не завершится ПОСЛЕДНЯЯ
        # ожидающая цель (тот же паттерн, что и tts_speaking ниже), защищена
        # тем же _goals_pending_lock.
        self._active_face_emotion: str | None = None
        self._FACE_EMOTIONS = frozenset(('neutral', 'happy', 'sad', 'surprise'))

        # Stall-watchdog (см. комментарий у _STALL_TIMEOUT_SEC)
        self._playback_active      = threading.Event()
        self._last_write_ts        = 0.0
        # Инцидент 2026-08-15: watchdog сравнивал "сейчас" с моментом ПОСЛЕДНЕГО
        # УСПЕШНОГО write() — это путает две разные вещи. Между HTTP-чанками
        # /tts/stream есть законные паузы (сервер ещё генерирует следующий
        # кусок аудио) — тогда write() вообще не вызывается, writes_in_progress=0.
        # Раньше watchdog принимал такую паузу за "write() завис" и убивал
        # процесс через 8с, хотя ALSA/PortAudio были ни при чём. Теперь watchdog
        # смотрит только на время ВНУТРИ самого вызова write() — т.е. реальный
        # симптом busy-retry XRun-петли (инцидент 2026-08-07), а не на паузы
        # между приходом данных по сети.
        self._write_in_progress_since = 0.0
        self._stall_watchdog_ready = False

        # Заглушки — заполняются в on_configure / on_activate
        self._session       = None
        self._speaking_pub  = None
        self._jaw_pub       = None
        self._face_expr_pub = None
        self._action_server = None
        self._active_url    = None
        self._output_device = None

    # ── Lifecycle: Phase 2 ─────────────────────────────────────────────────

    def _dp(self, name, default=None):
        """Безопасный declare_parameter: игнорирует повторное объявление при re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('tts_server_url',       'http://192.168.10.118:8000')
        self._dp('tts_fallback_url',     'http://localhost:8000')
        self._dp('chunk_size',           4096)
        self._dp('connect_timeout_sec',  5.0)
        self._dp('timeout_sec',          30.0)
        self._dp('output_device_name',   '')
        self._dp('jaw_closed',           10)
        self._dp('jaw_open',             90)
        self._dp('jaw_rms_threshold',    300.0)
        self._dp('jaw_rms_max',          6000.0)
        self._dp('jaw_speed_deg_per_sec', 400.0)

        self.primary_url     = self.get_parameter('tts_server_url').value
        self.fallback_url    = self.get_parameter('tts_fallback_url').value
        self.chunk_size      = self.get_parameter('chunk_size').value
        self.connect_timeout = self.get_parameter('connect_timeout_sec').value
        self.timeout_sec     = self.get_parameter('timeout_sec').value
        self._active_url     = self.primary_url
        self._jaw_closed     = self.get_parameter('jaw_closed').value
        self._jaw_open       = self.get_parameter('jaw_open').value
        self._jaw_rms_threshold = self.get_parameter('jaw_rms_threshold').value
        self._jaw_rms_max       = self.get_parameter('jaw_rms_max').value
        self._jaw_speed_rad_s   = math.radians(
            self.get_parameter('jaw_speed_deg_per_sec').value)

        self._session = requests.Session()

        self.create_subscription(Bool,   '/tts_cancel_queue',  self._cancel_queue_cb, 10)
        self._speaking_pub = self.create_lifecycle_publisher(Bool, 'tts_speaking', 10)
        self._jaw_pub      = self.create_lifecycle_publisher(JointState, '/face_command', 10)
        # Held-мимика на время речи (см. _execute_speak) — отдельно от
        # анимированного одноразового /face_expression (greet/farewell/BT).
        self._face_expr_pub = self.create_lifecycle_publisher(
            String, '/face_expression_hold', 10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._speaking_pub.on_activate(state)
        self._jaw_pub.on_activate(state)
        self._face_expr_pub.on_activate(state)

        # Инцидент 2026-08-15: при аварийном os._exit(1) из _stall_watchdog_loop
        # процесс убивается посреди speaking=True — finally-блок с
        # self._publish_speaking(False) не успевает выполниться. После рестарта
        # voice_detector_node навсегда остаётся уверен, что TTS говорит, и
        # игнорирует весь микрофонный ввод (voice_detector_node._audio_callback:
        # `if self.tts_speaking: return`). Публикуем False сразу при активации —
        # при штатном старте это no-op (получатели уже инициализированы как
        # False), но снимает залипание после аварийного рестарта.
        self._publish_speaking(False)

        # Находим output device (нужен живой PipeWire)
        self._output_device = self._find_output_device(
            self.get_parameter('output_device_name').value)

        # Проверяем серверы — WARN если недоступны, но не FAILURE (fallback есть)
        self._check_servers()

        if not self._stall_watchdog_ready:
            self._stall_watchdog_ready = True
            threading.Thread(
                target=self._stall_watchdog_loop, daemon=True,
                name='tts_stall_watchdog').start()

        self._action_server = ActionServer(
            self,
            Speak,
            'speak',
            execute_callback=self._execute_speak,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info(f'TTS Action Server готов. Сервер: {self._active_url}')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        # Прерываем текущее воспроизведение
        with self._abort_lock:
            self._abort_event.set()

        if self._action_server is not None:
            self._action_server.destroy()
            self._action_server = None

        self._speaking_pub.on_deactivate(state)
        self._jaw_pub.on_deactivate(state)
        self._face_expr_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        if self._session:
            self._session.close()
            self._session = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        with self._abort_lock:
            self._abort_event.set()
        if self._session:
            self._session.close()
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        if self._session:
            self._session.close()
        return TransitionCallbackReturn.SUCCESS

    # ── Stall-watchdog (см. комментарий у _STALL_TIMEOUT_SEC) ─────────────

    def _stall_watchdog_loop(self):
        """Живёт всё время работы процесса (не привязан к lifecycle-состоянию:
        застрять в C-петле можно и во время on_deactivate).

        Смотрит ТОЛЬКО на время внутри активного вызова stream.write() —
        см. комментарий у self._write_in_progress_since в __init__. Паузы
        между чанками (ждём данные по сети/WS) сюда не попадают."""
        while True:
            time.sleep(1.0)
            if not self._playback_active.is_set():
                continue
            started = self._write_in_progress_since
            if started <= 0.0:
                continue  # сейчас не внутри write() — ждём данные, это нормально
            stalled = time.time() - started
            if stalled > self._STALL_TIMEOUT_SEC:
                self.get_logger().fatal(
                    f'TTS: stream.write() не прогрессирует {stalled:.1f}с — '
                    f'похоже на зависшую ALSA/PipeWire XRun-петлю внутри '
                    f'PortAudio. Аварийный перезапуск процесса.'
                )
                import sys
                sys.stderr.flush()
                os._exit(1)  # SIGKILL-подобный выход: рвём C-уровень немедленно

    def _write_chunk(self, stream, data: bytes):
        """stream.write() с heartbeat для stall-watchdog.
        _write_in_progress_since отмечает окно РЕАЛЬНОГО вызова write() —
        watchdog реагирует только пока мы внутри него."""
        self._write_in_progress_since = time.time()
        try:
            stream.write(data)
        finally:
            self._write_in_progress_since = 0.0
        self._last_write_ts = time.time()

    # ── Проверка серверов ──────────────────────────────────────────────────

    def _check_servers(self):
        if self._probe_server(self.primary_url):
            self._active_url = self.primary_url
            self.get_logger().info(f'TTS: основной сервер доступен ({self.primary_url})')
        elif self._probe_server(self.fallback_url):
            self._active_url = self.fallback_url
            self.get_logger().warn(
                f'Основной TTS недоступен! Резервный: {self.fallback_url}')
        else:
            self.get_logger().error('Оба TTS сервера недоступны!')

    def _probe_server(self, url: str) -> bool:
        try:
            r = self._session.get(f'{url}/health', timeout=self.connect_timeout)
            info = r.json()
            self.get_logger().info(
                f'  {url}: GPU={info.get("gpu", "CPU")}, '
                f'VRAM={info.get("vram_used_mb", "?")}/'
                f'{info.get("vram_total_mb", "?")} MB, '
                f'SR={info.get("sample_rate", "?")}'
            )
            return True
        except Exception:
            return False

    # ── Сброс очереди TTS (preemption при пользовательском прерывании) ────

    def _cancel_queue_cb(self, msg: Bool):
        """Отменяет текущий goal и помечает все ожидающие для отклонения."""
        if not msg.data:
            return
        with self._goals_pending_lock:
            if self._goals_pending == 0:
                return  # action server не играет — нечего отменять
            self._cancel_queued.set()
        with self._abort_lock:
            self._abort_event.set()  # прерывает текущее воспроизведение
        self.get_logger().info('TTS: сброс очереди (пользовательское прерывание)')

    # ── Action callbacks ───────────────────────────────────────────────────

    def _goal_callback(self, goal_request):
        """Принимаем все goals. Preemption происходит внутри execute."""
        text_preview = (goal_request.text[:40] + '...') if len(goal_request.text) > 40 else goal_request.text
        style_info = ''
        if goal_request.voice:
            style_info += f' emotion="{goal_request.voice}"'
        if goal_request.rate not in (0.0, 1.0):
            style_info += f' speed={goal_request.rate:.2f}'
        self.get_logger().info(f'Новый Speak goal: "{text_preview}"{style_info}')
        with self._goals_pending_lock:
            self._goals_pending += 1
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        """Принимаем запросы на отмену от клиентов."""
        self.get_logger().info('Запрос отмены TTS goal')
        return CancelResponse.ACCEPT

    # ── Execute: основная логика ───────────────────────────────────────────

    def _execute_speak(self, goal_handle):
        """
        Выполняется в отдельном потоке (ReentrantCallbackGroup).

        Preemption: при старте создаём новый abort_event, сигнализируем старому.
        Старый execute loop проверяет свой event и выходит.
        """
        # ── Ждём завершения предыдущего goal (очередь, не preemption) ────────
        # Таймаут: если предыдущий завис — пропускаем через 35с
        if not self._execute_lock.acquire(timeout=35.0):
            self.get_logger().warn('TTS: предыдущий goal завис — пропускаю')
            goal_handle.abort()
            result = Speak.Result()
            result.success = False
            result.message = 'Таймаут очереди'
            with self._goals_pending_lock:
                self._goals_pending -= 1
            return result

        my_abort = threading.Event()
        with self._abort_lock:
            self._abort_event = my_abort

        text    = goal_handle.request.text.strip()
        emotion = goal_handle.request.voice.strip()
        if not text:
            goal_handle.abort()
            result = Speak.Result()
            result.success = False
            result.message = 'Пустой текст'
            with self._goals_pending_lock:
                self._goals_pending -= 1
            self._execute_lock.release()
            return result

        # Если очередь была сброшена (пользователь перебил) — отклоняем этот чанк
        if self._cancel_queued.is_set():
            goal_handle.abort()
            result = Speak.Result()
            result.success = False
            result.message = 'Очередь сброшена'
            with self._goals_pending_lock:
                self._goals_pending -= 1
                if self._goals_pending == 0:
                    self._cancel_queued.clear()
            self._execute_lock.release()
            return result

        feedback      = Speak.Feedback()
        bytes_played  = 0
        start_time    = time.time()

        def _fb(status: str, progress: float = -1.0):
            feedback.status      = status
            feedback.progress    = progress
            feedback.bytes_played = bytes_played
            goal_handle.publish_feedback(feedback)

        _fb('connecting')

        # ── Выбор URL с fallback ───────────────────────────────────────────
        urls = [self._active_url]
        other = self.fallback_url if self._active_url == self.primary_url else self.primary_url
        if other != self._active_url:
            urls.append(other)

        success   = False
        error_msg = ''

        self._publish_speaking(True)
        # Мимика лица синхронно с голосом на всё время этой фразы (см.
        # set_voice_style в llm_node). Пусто (greet/farewell) — лицо не трогаем.
        face_emotion = emotion.lower()
        if face_emotion in self._FACE_EMOTIONS:
            with self._goals_pending_lock:
                self._active_face_emotion = face_emotion
            self._publish_face_emotion(face_emotion)
        try:
            for url in urls:
                if my_abort.is_set() or goal_handle.is_cancel_requested:
                    break
                try:
                    success, bytes_played, error_msg = self._stream_and_play(
                        url, text, emotion, goal_handle, my_abort, _fb
                    )
                    if success or error_msg not in ('connection_error',):
                        if url != self._active_url:
                            self.get_logger().warn(f'Переключился на резервный TTS: {url}')
                            self._active_url = url
                        break
                except Exception as e:
                    error_msg = str(e)
                    self.get_logger().error(f'TTS ошибка [{url}]: {e}')
        finally:
            revert_face = False
            with self._goals_pending_lock:
                self._goals_pending -= 1
                if self._goals_pending == 0:
                    self._cancel_queued.clear()
                    # Последняя ожидающая цель — если для неё была показана
                    # не-нейтральная эмоция, возвращаем лицо в neutral. Работает
                    # и при обычном завершении, и при cancel/abort (оба пути
                    # приходят сюда же).
                    if self._active_face_emotion not in (None, 'neutral'):
                        revert_face = True
                    self._active_face_emotion = None
            if revert_face:
                self._publish_face_emotion('neutral')
            self._publish_speaking(False)
            self._execute_lock.release()

        # ── Результат ─────────────────────────────────────────────────────
        audio_sec = time.time() - start_time

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result = Speak.Result()
            result.success   = False
            result.message   = 'Отменено клиентом'
            result.audio_sec = audio_sec
            return result

        if success:
            self.get_logger().info(f'TTS воспроизведён за {audio_sec:.1f}с')
            goal_handle.succeed()
        else:
            goal_handle.abort()

        result = Speak.Result()
        result.success   = success
        result.message   = '' if success else error_msg
        result.audio_sec = audio_sec
        return result

    def _stream_and_play(
        self, url: str, text: str, emotion: str,
        goal_handle, my_abort: threading.Event, fb
    ) -> tuple[bool, int, str]:
        """
        Стримит аудио с TTS сервера и воспроизводит.
        Возвращает (success, bytes_played, error_msg).

        emotion — имя голосового пресета сервера (OmniVoice/audio.cpp,
        server.json → voice_presets): "neutral"/"happy"/"sad"/"surprise".
        Клонирование голоса доминирует над текстовыми instruct-инструкциями
        (наследие CosyVoice3), поэтому только предустановленные пресеты.
        Пусто/неизвестное имя = сервер откатывает на "neutral".
        """
        body: dict = {'text': text}
        if emotion:
            body['emotion'] = emotion

        # TTFA (Time To First Audio) — от отправки POST до первого полученного
        # аудио-байта: коннект + время сервера до начала выдачи потока.
        # Именно эта задержка определяет ощущаемую "отзывчивость" TTS —
        # см. обсуждение миграции CosyVoice3 → OmniVoice (MIGRATION_NOTES.md).
        t_req_start = time.time()
        ttfa_logged = False

        try:
            with self._session.post(
                f'{url}/tts/stream',
                json=body,
                stream=True,
                timeout=(self.connect_timeout, self.timeout_sec),
            ) as resp:
                resp.raise_for_status()
                fb('synthesizing')

                sample_rate    = self._parse_sample_rate(resp.headers)
                content_length = int(resp.headers.get('Content-Length', -1))
                bytes_played   = 0

                with sd.RawOutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype='int16',
                    device=self._output_device,
                ) as stream:
                    fb('playing')
                    self._last_write_ts = time.time()
                    self._playback_active.set()
                    try:
                        for chunk in resp.iter_content(chunk_size=self.chunk_size):
                            # Preemption / cancellation check между чанками
                            if my_abort.is_set() or goal_handle.is_cancel_requested:
                                stream.abort()
                                resp.close()
                                self._publish_jaw(self._jaw_closed)
                                return False, bytes_played, 'aborted'

                            if chunk:
                                if not ttfa_logged:
                                    self.get_logger().info(
                                        f'TTFA: {time.time() - t_req_start:.2f}с '
                                        f'[{url}]')
                                    ttfa_logged = True
                                self._write_chunk(stream, chunk)
                                bytes_played += len(chunk)
                                progress = (bytes_played / content_length
                                            if content_length > 0 else -1.0)
                                fb('playing', progress)
                                self._publish_jaw(self._chunk_to_jaw(chunk))
                    finally:
                        self._playback_active.clear()

                self._publish_jaw(self._jaw_closed)

                if bytes_played == 0:
                    return False, 0, 'Пустой поток от TTS сервера'

                return True, bytes_played, ''

        except requests.exceptions.ConnectionError as e:
            self.get_logger().warn(f'TTS ConnectionError [{url}]: {e}')
            return False, 0, 'connection_error'
        except requests.exceptions.Timeout:
            return False, 0, f'Таймаут ({self.timeout_sec}с)'
        except requests.exceptions.HTTPError as e:
            return False, 0, f'HTTP ошибка: {e}'

    # ── Вспомогательные методы ─────────────────────────────────────────────

    def _chunk_to_jaw(self, chunk: bytes) -> int:
        """Вычисляет RMS чанка PCM int16 и маппит в позицию челюсти."""
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < self._jaw_rms_threshold:
            return self._jaw_closed
        t = min(1.0, (rms - self._jaw_rms_threshold) /
                     (self._jaw_rms_max - self._jaw_rms_threshold))
        return int(self._jaw_closed + t * (self._jaw_open - self._jaw_closed))

    def _publish_jaw(self, position: int):
        """Публикует позицию челюсти через /face_command (JointState).
        position — градусы [jaw_closed..jaw_open], конвертируется в радианы
        по формуле (deg - 90) * π/180 (center_deg=90, как везде в face protocol).
        velocity  — скорость в rad/s; arduino_left_node конвертирует в step
                    и отправляет CMD_SET_SPEEDS только при изменении.
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = ['jaw']
        msg.position = [(float(position) - 90.0) * math.pi / 180.0]
        msg.velocity = [self._jaw_speed_rad_s]
        self._jaw_pub.publish(msg)

    def _publish_speaking(self, speaking: bool):
        msg = Bool()
        msg.data = speaking
        self._speaking_pub.publish(msg)

    def _publish_face_emotion(self, name: str):
        """Held-мимика на время речи, см. _execute_speak. Отдельно от
        анимированного /face_expression — face_expressions_node применяет
        позу статично, без встроенного авто-возврата."""
        msg = String()
        msg.data = name
        self._face_expr_pub.publish(msg)

    def _find_output_device(self, name: str):
        if not name:
            return None
        name_lower = name.lower()
        # Prefer PipeWire/pulse virtual device (no "(hw:" in name).
        # Direct hw: access conflicts with PipeWire exclusive ownership → silent playback.
        for i, d in enumerate(sd.query_devices()):
            if '(hw:' not in d['name'] and d['max_output_channels'] > 0 \
                    and name_lower in d['name'].lower():
                self.get_logger().info(f'Аудио вывод: [{i}] {d["name"]}')
                return i
        # hw: device found but PipeWire manages it → fall back to system default
        for i, d in enumerate(sd.query_devices()):
            if '(hw:' in d['name'] and d['max_output_channels'] > 0 \
                    and name_lower in d['name'].lower():
                self.get_logger().warn(
                    f'Устройство "{d["name"]}" доступно только через raw ALSA — '
                    f'использую системное (PipeWire)')
                return None
        self.get_logger().warn(f'Устройство вывода "{name}" не найдено, использую системное')
        return None

    @staticmethod
    def _parse_sample_rate(headers) -> int:
        content_type = headers.get('Content-Type', '')
        for part in content_type.split(';'):
            part = part.strip()
            if part.startswith('rate='):
                try:
                    return int(part[5:])
                except ValueError:
                    pass
        return int(headers.get('X-Sample-Rate', 24000))




def main():
    rclpy.init()
    node = TTSNode()
    # MultiThreadedExecutor нужен для ReentrantCallbackGroup:
    # execute callback нового goal должен стартовать пока старый ещё работает
    executor = MultiThreadedExecutor(num_threads=4)
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
