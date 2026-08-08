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

import json
import math
import os
import queue
import threading
import time

import numpy as np
import requests
import sounddevice as sd
import websocket as _ws_lib  # websocket-client (sync)

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

        # Stall-watchdog (см. комментарий у _STALL_TIMEOUT_SEC)
        self._playback_active      = threading.Event()
        self._last_write_ts        = 0.0
        self._stall_watchdog_ready = False

        # Bistream state — один WS на весь ответ LLM
        self._bs_text_q    = None        # queue.Queue текстовых кусков
        self._bs_abort     = threading.Event()
        self._bs_thread    = None        # поток _bs_worker
        self._bs_ws        = None        # открытый WS (для внешнего close)
        self._bs_lock      = threading.Lock()  # защита _bs_ws / _bs_abort
        self._bs_sample_rate = 24000     # обновляется из /health при probe
        self._bs_text_chars  = 0         # суммарный объём текста сессии (для лимита длины)
        self._bs_full_text   = ''        # полный накопленный текст сессии (для HTTP-fallback)

        # Заглушки — заполняются в on_configure / on_activate
        self._session       = None
        self._speaking_pub  = None
        self._jaw_pub       = None
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
        self.create_subscription(String, '/tts/stream_ctrl',   self._bs_ctrl_cb,      50)
        self._speaking_pub = self.create_lifecycle_publisher(Bool, 'tts_speaking', 10)
        self._jaw_pub      = self.create_lifecycle_publisher(JointState, '/face_command', 10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._speaking_pub.on_activate(state)
        self._jaw_pub.on_activate(state)

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
        # Прерываем текущее воспроизведение (action server + bistream)
        with self._abort_lock:
            self._abort_event.set()
        self._bs_cancel()

        if self._action_server is not None:
            self._action_server.destroy()
            self._action_server = None

        self._speaking_pub.on_deactivate(state)
        self._jaw_pub.on_deactivate(state)
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
        застрять в C-петле можно и во время on_deactivate)."""
        while True:
            time.sleep(1.0)
            if not self._playback_active.is_set():
                continue
            stalled = time.time() - self._last_write_ts
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
        """stream.write() с heartbeat для stall-watchdog."""
        stream.write(data)
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
            sr = info.get('sample_rate')
            if sr and url == self.primary_url:
                self._bs_sample_rate = int(sr)
            self.get_logger().info(
                f'  {url}: GPU={info.get("gpu", "CPU")}, '
                f'VRAM={info.get("vram_used_mb", "?")}/'
                f'{info.get("vram_total_mb", "?")} MB, '
                f'SR={self._bs_sample_rate}'
            )
            return True
        except Exception:
            return False

    # ── Сброс очереди TTS (preemption при пользовательском прерывании) ────

    def _cancel_queue_cb(self, msg: Bool):
        """Отменяет текущий goal и помечает все ожидающие для отклонения."""
        if not msg.data:
            return
        # Отменяем bistream независимо от _goals_pending
        self._bs_cancel()
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
            style_info += f' instruct="{goal_request.voice}"'
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

        text     = goal_handle.request.text.strip()
        instruct = goal_handle.request.voice.strip()
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
        try:
            for url in urls:
                if my_abort.is_set() or goal_handle.is_cancel_requested:
                    break
                try:
                    success, bytes_played, error_msg = self._stream_and_play(
                        url, text, instruct, goal_handle, my_abort, _fb
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
            with self._goals_pending_lock:
                self._goals_pending -= 1
                if self._goals_pending == 0:
                    self._cancel_queued.clear()
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
        self, url: str, text: str, instruct: str,
        goal_handle, my_abort: threading.Event, fb
    ) -> tuple[bool, int, str]:
        """
        Стримит аудио с TTS сервера и воспроизводит.
        Возвращает (success, bytes_played, error_msg).
        """
        body: dict = {'text': text}
        if instruct:
            # CosyVoice3 требует токен <|endofprompt|> в конце инструкции
            body['instruct'] = instruct if instruct.endswith('<|endofprompt|>') \
                               else instruct + '<|endofprompt|>'

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

    # ── Bistream: WS-клиент для LLM-ответов ───────────────────────────────────

    def _bs_ctrl_cb(self, msg: String):
        """Управление bistream сессией через единый топик (FIFO-порядок гарантирован).
        Протокол: 'start:<instruct>', 'text:<chunk>', 'end', 'cancel'."""
        cmd = msg.data
        if cmd.startswith('start:'):
            instruct = cmd[6:]
            # Отменяем предыдущую сессию (если была)
            self._bs_cancel()
            # Отменяем и action server (если говорит) — bistream приоритетнее
            with self._abort_lock:
                self._abort_event.set()
            with self._bs_lock:
                self._bs_abort.clear()
                self._bs_text_q = queue.Queue()
                self._bs_text_chars = 0
                self._bs_full_text  = ''
            self._bs_thread = threading.Thread(
                target=self._bs_worker, args=(instruct,), daemon=True)
            self._bs_thread.start()
        elif cmd.startswith('text:'):
            text = cmd[5:]
            with self._bs_lock:
                if self._bs_text_q is not None and not self._bs_abort.is_set():
                    self._bs_text_q.put(text)
                    self._bs_text_chars += len(text)
                    self._bs_full_text  += (' ' if self._bs_full_text else '') + text
        elif cmd == 'end':
            with self._bs_lock:
                if self._bs_text_q is not None:
                    self._bs_text_q.put(None)  # sentinel → генератор завершится
        elif cmd == 'cancel':
            self._bs_cancel()

    def _bs_cancel(self):
        """Прерывает текущую bistream сессию (потокобезопасно)."""
        with self._bs_lock:
            self._bs_abort.set()
            ws = self._bs_ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def _bs_worker(self, instruct: str):
        """
        Поток: подключается к /tts/bistream, пересылает текст, воспроизводит аудио.
        Использует _execute_lock чтобы сериализоваться с action server.
        При ошибке соединения до первых аудио-данных пробует fallback URL.
        """
        if not self._execute_lock.acquire(timeout=self.timeout_sec):
            self.get_logger().warn('Bistream: не могу получить execute_lock')
            return
        my_abort = threading.Event()
        with self._abort_lock:
            self._abort_event = my_abort
        try:
            urls = [self._active_url]
            other = self.fallback_url if self._active_url == self.primary_url else self.primary_url
            if other != self._active_url:
                urls.append(other)
            loop_fallback_triggered = False
            for url in urls:
                if my_abort.is_set() or self._bs_abort.is_set():
                    break
                ws_url = (url.replace('http://', 'ws://')
                             .replace('https://', 'wss://') + '/tts/bistream')
                played, loop_detected = self._bs_run(ws_url, instruct, my_abort)
                if played:
                    if url != self._active_url:
                        self.get_logger().warn(f'Bistream: переключился на резервный TTS: {url}')
                        self._active_url = url
                    if loop_detected and not my_abort.is_set() and not self._bs_abort.is_set():
                        # Bistream зациклился: повторяем через HTTP /tts/stream (надёжный)
                        with self._bs_lock:
                            fallback_text = self._bs_full_text
                        if fallback_text:
                            self.get_logger().info(
                                f'Bistream: HTTP-fallback для "{fallback_text[:60]}..."')
                            loop_fallback_triggered = True
                            self._bs_http_fallback(url, fallback_text, instruct, my_abort)
                    break
                if len(urls) > 1 and url == urls[0]:
                    self.get_logger().warn(
                        f'Bistream: {url} недоступен, пробуем {urls[1]}')
        finally:
            with self._goals_pending_lock:
                if self._goals_pending == 0:
                    self._cancel_queued.clear()
            self._execute_lock.release()

    # Сколько байт аудио ожидаем на символ текста (эмпирически, 24kHz int16)
    # ~0.08с/символ * 48000 байт/с = 3840 байт/символ; множитель 4x — запас
    _BS_BYTES_PER_CHAR      = 3840
    _BS_DURATION_MULTIPLIER = 4.0   # допускаем до 4x от ожидаемого
    _BS_MIN_MAX_BYTES       = 48000 * 10  # минимальный лимит: 10с (короткие фразы)
    # Детектор петли: окно 2с, порог схожести, минимум 3с аудио до первой проверки
    _BS_LOOP_WINDOW_BYTES   = 48000 * 2   # 2с при 24kHz int16
    _BS_LOOP_SIM_THRESHOLD  = 0.98        # выше → все блоки идентичны → петля
    _BS_LOOP_MIN_BYTES      = 48000 * 3   # не проверяем до 3с аудио
    _BS_LOOP_CONSECUTIVE    = 2           # 2 срабатывания подряд → аборт

    def _bs_run(self, ws_url: str, instruct: str, my_abort: threading.Event) -> tuple[bool, bool]:
        """Основная логика bistream: WS + sounddevice.
        Возвращает (played_any, loop_detected):
          played_any    — True если хоть какой-то аудио был воспроизведён
          loop_detected — True если сработал детектор петли (нужен HTTP-fallback)
        """
        bytes_played  = 0
        loop_detected = False
        self.get_logger().info(f'Bistream: подключение к {ws_url}')
        try:
            ws = _ws_lib.create_connection(ws_url, timeout=self.connect_timeout)
            # Короткий timeout для recv_data: позволяет регулярно проверять флаги отмены
            # и защищает от вечного зависания если сервер завис не закрыв соединение.
            ws.settimeout(2.0)
            with self._bs_lock:
                self._bs_ws = ws

            ws.send(json.dumps({'type': 'start', 'instruct': instruct}))

            # Читаем суммарный объём текста до отправки end (может ещё прибавляться)
            with self._bs_lock:
                text_chars_snapshot = self._bs_text_chars

            # Отдельный поток: очередь → WS (не блокирует recv ниже)
            def _sender():
                try:
                    while True:
                        with self._bs_lock:
                            q = self._bs_text_q
                        if q is None:
                            break
                        try:
                            chunk = q.get(timeout=1.0)
                        except queue.Empty:
                            if self._bs_abort.is_set():
                                break
                            continue
                        if chunk is None:
                            ws.send(json.dumps({'type': 'end'}))
                            return
                        ws.send(json.dumps({'type': 'text', 'text': chunk}))
                except Exception as e:
                    self.get_logger().debug(f'Bistream sender: {e}')

            threading.Thread(target=_sender, daemon=True).start()

            self._publish_speaking(True)
            _last_data_ts = time.time()
            _bs_silence_max = 60.0  # сервер молчит >60с после последних данных → обрываем

            # Лимит длительности по объёму текста (защита от бесконечной петли генерации)
            with self._bs_lock:
                total_chars = max(self._bs_text_chars, text_chars_snapshot)
            max_bytes = max(
                int(total_chars * self._BS_BYTES_PER_CHAR * self._BS_DURATION_MULTIPLIER),
                self._BS_MIN_MAX_BYTES,
            )
            max_sec = max_bytes / (self._bs_sample_rate * 2)
            self.get_logger().debug(
                f'Bistream: лимит {max_sec:.1f}с ({total_chars} символов текста)')

            # Детектор петли: скользящий буфер последних 2с аудио
            _loop_buf: list[bytes] = []
            _loop_buf_bytes        = 0
            _loop_stuck_count      = 0
            _loop_next_check       = self._BS_LOOP_WINDOW_BYTES  # проверяем после первого окна

            with sd.RawOutputStream(
                samplerate=self._bs_sample_rate,
                channels=1,
                dtype='int16',
                device=self._output_device,
            ) as stream:
              self._last_write_ts = time.time()
              self._playback_active.set()
              try:
                while True:
                    if my_abort.is_set() or self._bs_abort.is_set():
                        stream.abort()
                        break
                    try:
                        opcode, data = ws.recv_data()
                    except Exception as _e:
                        # poll timeout — проверяем watchdog и флаги, не выходим сразу
                        if isinstance(_e, (_ws_lib.WebSocketTimeoutException, TimeoutError)):
                            if time.time() - _last_data_ts > _bs_silence_max:
                                self.get_logger().warn(
                                    f'Bistream: сервер молчит >{_bs_silence_max:.0f}с — обрываем')
                                break
                            continue
                        break
                    _last_data_ts = time.time()
                    if not data:
                        break
                    if isinstance(data, bytes) and data:
                        self._write_chunk(stream, data)
                        bytes_played += len(data)
                        self._publish_jaw(self._chunk_to_jaw(data))

                        # ── Лимит по длительности ──────────────────────────────
                        if bytes_played > max_bytes:
                            self.get_logger().warn(
                                f'Bistream: превышен лимит {max_sec:.1f}с '
                                f'(текст={total_chars} симв.) — прерываем')
                            stream.abort()
                            break

                        # ── Детектор петли (спектральная монотонность) ─────────
                        if bytes_played >= self._BS_LOOP_MIN_BYTES:
                            _loop_buf.append(data)
                            _loop_buf_bytes += len(data)
                            # Обрезаем буфер до последних _BS_LOOP_WINDOW_BYTES
                            while _loop_buf_bytes > self._BS_LOOP_WINDOW_BYTES and _loop_buf:
                                removed = _loop_buf.pop(0)
                                _loop_buf_bytes -= len(removed)

                            if bytes_played >= _loop_next_check:
                                _loop_next_check = bytes_played + self._BS_LOOP_WINDOW_BYTES // 2
                                sim = self._bs_loop_similarity(_loop_buf)
                                if sim > self._BS_LOOP_SIM_THRESHOLD:
                                    _loop_stuck_count += 1
                                    self.get_logger().warn(
                                        f'Bistream: подозрение на петлю '
                                        f'(sim={sim:.3f}, #{_loop_stuck_count})')
                                    if _loop_stuck_count >= self._BS_LOOP_CONSECUTIVE:
                                        self.get_logger().error(
                                            'Bistream: петля подтверждена — прерываем! '
                                            'Попробуем повторить через HTTP.')
                                        loop_detected = True
                                        stream.abort()
                                        break
                                else:
                                    _loop_stuck_count = 0
              finally:
                self._playback_active.clear()

            self._publish_jaw(self._jaw_closed)
            self.get_logger().info('Bistream: воспроизведение завершено')

        except _ws_lib.WebSocketException as e:
            self.get_logger().warn(f'Bistream WS error: {e}')
        except Exception as e:
            self.get_logger().error(f'Bistream error: {e}')
        finally:
            self._publish_speaking(False)
            with self._bs_lock:
                self._bs_ws = None
        return bytes_played > 0, loop_detected

    # ── Вспомогательные методы ─────────────────────────────────────────────────

    def _bs_http_fallback(self, url: str, text: str, instruct: str,
                          my_abort: threading.Event):
        """HTTP /tts/stream fallback после обнаружения bistream-петли.
        Воспроизводит весь текст через надёжный batch-эндпоинт.
        """
        body: dict = {'text': text}
        if instruct:
            body['instruct'] = instruct if instruct.endswith('<|endofprompt|>') \
                               else instruct + '<|endofprompt|>'
        try:
            self._publish_speaking(True)
            with self._session.post(
                f'{url}/tts/stream',
                json=body,
                stream=True,
                timeout=(self.connect_timeout, self.timeout_sec),
            ) as resp:
                resp.raise_for_status()
                sample_rate = self._parse_sample_rate(resp.headers)
                with sd.RawOutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype='int16',
                    device=self._output_device,
                ) as stream:
                    self._last_write_ts = time.time()
                    self._playback_active.set()
                    try:
                        for chunk in resp.iter_content(chunk_size=self.chunk_size):
                            if my_abort.is_set() or self._bs_abort.is_set():
                                stream.abort()
                                resp.close()
                                return
                            if chunk:
                                self._write_chunk(stream, chunk)
                                self._publish_jaw(self._chunk_to_jaw(chunk))
                    finally:
                        self._playback_active.clear()
            self._publish_jaw(self._jaw_closed)
            self.get_logger().info('Bistream HTTP-fallback: воспроизведение завершено')
        except Exception as e:
            self.get_logger().error(f'Bistream HTTP-fallback ошибка: {e}')
        finally:
            self._publish_speaking(False)

    def _bs_loop_similarity(self, buf: list[bytes]) -> float:
        """Средняя попарная спектральная схожесть сегментов в буфере.
        Значение >0.97 означает монотонный повторяющийся звук (петля модели).
        Нормальная речь: 0.5–0.85. Петля: 0.95–1.0.
        """
        raw = b''.join(buf)
        if len(raw) < 4096:
            return 0.0
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        sr  = self._bs_sample_rate
        seg_len = sr // 4  # 250ms сегменты
        n = len(pcm) // seg_len
        if n < 4:
            return 0.0
        feats = []
        for i in range(min(n, 8)):
            seg = pcm[i * seg_len:(i + 1) * seg_len]
            f   = np.abs(np.fft.rfft(seg))
            # 8 log-energy bands — грубые, но достаточны для детекции петли
            nb  = 8
            b   = np.array([np.mean(f[j * len(f) // nb:(j + 1) * len(f) // nb])
                             for j in range(nb)])
            norm = np.linalg.norm(b)
            if norm > 0:
                feats.append(b / norm)
        if len(feats) < 4:
            return 0.0
        sims = [float(np.dot(feats[i], feats[j]))
                for i in range(len(feats))
                for j in range(i + 1, len(feats))]
        return float(np.mean(sims))

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
