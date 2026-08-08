#!/usr/bin/env python3

"""
audio_source_node.py
====================
Единственная нода, которая открывает микрофон.
Публикует нормализованные float32 чанки в топик 'raw_audio'.

Lifecycle:
  on_configure — declare params, create publisher
  on_activate  — открыть аудио поток; FAILURE если PipeWire не готов (→ retry)
  on_deactivate — закрыть поток, остановить таймеры

Параметры:
  sample_rate  (int)   — 16000
  chunk_size   (int)   — 512  (32 мс; оптимально для Silero VAD и OWW)
  device_index (int)   — -1 = системное устройство по умолчанию
  device_name  (str)   — поиск по подстроке имени
  watchdog_sec (float) — перезапуск если нет аудио N секунд (default 5.0)
"""

import os
import queue
import re
import subprocess
import threading
import time
from collections import deque

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import pyaudio
import numpy as np


class AudioSourceNode(LifecycleNode):
    # Инцидент 2026-08-07: при сбое PipeWire/ALSA reader-тред может зависнуть
    # внутри blocking stream.read() в C-level busy-retry XRun-петле PortAudio
    # (та же природа, что и у tts_node — см. комментарий там). _restart_stream()
    # ниже НЕ убивает такой тред (см. комментарий в _restart_stream) — он
    # намеренно "бросается" в надежде, что PipeWire восстановится и read()
    # сам вернёт OSError. Если это не происходит, тред жрёт ядро CPU вечно
    # и накапливается с каждым новым рестартом. Единственная защита —
    # обнаружить частые рестарты подряд (симптом застрявшего reader'а,
    # который watchdog не может вылечить) и убить процесс целиком: respawn=True
    # в launch-файле поднимет чистый процесс через respawn_delay секунд.
    _RESTART_STORM_COUNT      = 5     # рестартов...
    _RESTART_STORM_WINDOW_SEC = 30.0  # ...за это окно → аварийный выход

    def __init__(self):
        super().__init__('audio_source_node')
        self._restart_times = deque(maxlen=self._RESTART_STORM_COUNT)
        self._pa                  = None
        self._stream              = None
        self._last_ok             = 0.0
        self._restart_count       = 0
        self._timers              = []
        self._pub                 = None
        self._audio_queue         = queue.Queue(maxsize=5)
        self._stream_running      = False
        self._read_thread         = None
        self._last_wp_restart     = 0.0
        self._recovery_lock       = threading.Lock()
        self._restart_in_progress = False
        # Exponential backoff for persistent PipeWire failures
        self._retry_backoff_sec   = 5.0
        self._next_retry_time     = 0.0
        self._MAX_BACKOFF_SEC     = 120.0

    # ── Lifecycle callbacks ────────────────────────────────────────────────

    def _dp(self, name, default=None):
        """Безопасный declare_parameter: игнорирует повторное объявление при re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('sample_rate',              16000)
        self._dp('chunk_size',               512)
        self._dp('device_index',             -1)
        self._dp('device_name',              '')
        self._dp('watchdog_sec',             5.0)
        self._dp('pa_source_check',          '')   # подстрока в pactl get-default-source; пусто = не проверять
        self._dp('jabra_card_name',          'Jabra_Speak2')  # подстрока для поиска карты в pactl list cards
        self._dp('jabra_profile',            'output:analog-stereo+input:mono-fallback')
        self._dp('zero_wp_check_count',      500)  # чанков нулей (~15с) → проверить PipeWire
        self._dp('wp_restart_cooldown_sec',  90.0)

        self.rate                    = self.get_parameter('sample_rate').value
        self.chunk_size              = self.get_parameter('chunk_size').value
        self._device_index           = self.get_parameter('device_index').value
        self._device_name            = self.get_parameter('device_name').value
        self._watchdog_sec           = self.get_parameter('watchdog_sec').value
        self._pa_source_check        = self.get_parameter('pa_source_check').value
        self._jabra_card_name        = self.get_parameter('jabra_card_name').value
        self._jabra_profile          = self.get_parameter('jabra_profile').value
        self._zero_wp_check_count    = self.get_parameter('zero_wp_check_count').value
        self._wp_restart_cooldown    = self.get_parameter('wp_restart_cooldown_sec').value

        self._pub = self.create_lifecycle_publisher(Float32MultiArray, 'raw_audio', 20)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._pub.on_activate(state)

        if not self._open_stream():
            self._pub.on_deactivate(state)
            return TransitionCallbackReturn.FAILURE

        self._stream_running = True
        self._read_thread = threading.Thread(
            target=self._stream_reader, daemon=True, name='audio_reader')
        self._read_thread.start()

        timer_period = (self.chunk_size / self.rate) * 0.9
        self._timers.append(self.create_timer(timer_period, self._drain_and_publish))
        self._timers.append(self.create_timer(self._watchdog_sec, self._watchdog))
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self._stream_running = False
        for t in self._timers:
            self.destroy_timer(t)
        self._timers.clear()
        # Сигналим reader-треду через self._stream = None ПЕРЕД вызовом pa.terminate().
        # Без этого pa.terminate() при живом заблокированном read() → SIGABRT.
        self._safe_close_stream()
        self._pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self._safe_close_stream()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self._safe_close_stream()
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        self._safe_close_stream()
        return TransitionCallbackReturn.SUCCESS

    # ── Открытие потока ───────────────────────────────────────────────────

    def _check_pa_source(self) -> bool:
        """Проверяет что PulseAudio default source содержит self._pa_source_check.
        Возвращает False если устройство не найдено — не открываем поток."""
        if not self._pa_source_check:
            return True
        try:
            out = subprocess.check_output(
                ['pactl', 'get-default-source'], timeout=3, text=True
            ).strip()
            if self._pa_source_check.lower() not in out.lower():
                self.get_logger().error(
                    f'pa_source_check: default source "{out}" не содержит '
                    f'"{self._pa_source_check}" — Jabra не подключена, поток не открываю'
                )
                return False
            return True
        except Exception as e:
            self.get_logger().error(f'pa_source_check: pactl завершился с ошибкой: {e}')
            return False

    def _open_stream(self) -> bool:
        self._close_stream()

        if not self._check_pa_source():
            return False

        try:
            self._pa = pyaudio.PyAudio()

            device_index = self._device_index
            if self._device_name:
                device_index = self._find_device_by_name(self._device_name)

            open_kwargs = dict(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )
            if device_index >= 0:
                open_kwargs['input_device_index'] = device_index
                dev_name = self._pa.get_device_info_by_index(device_index).get('name', '?')
                self.get_logger().info(f'Устройство: [{device_index}] {dev_name}')
            else:
                self.get_logger().info('Устройство: системное по умолчанию')

            self._stream = self._pa.open(**open_kwargs)
            self._last_ok = time.time()
            self.get_logger().info(
                f'AudioSource запущен: {self.rate} Гц, chunk={self.chunk_size} '
                f'({1000 * self.chunk_size / self.rate:.0f} мс)'
            )
            return True

        except Exception as e:
            self.get_logger().error(f'Не удалось открыть аудио поток: {e}')
            self._close_stream()
            return False

    def _safe_close_stream(self):
        """Закрывает поток безопасно: обнуляет self._stream ПЕРЕД terminate(),
        чтобы reader-тред (возможно заблокированный в read()) не устроил SIGABRT."""
        stream, pa = self._stream, self._pa
        self._stream = None  # reader-тред видит None и выходит
        self._pa = None
        try:
            if stream is not None:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
        except Exception:
            pass
        try:
            if pa is not None:
                pa.terminate()
        except Exception:
            pass

    def _close_stream(self):
        try:
            if self._stream is not None:
                if self._stream.is_active():
                    self._stream.stop_stream()
                self._stream.close()
        except Exception:
            pass
        finally:
            self._stream = None

        try:
            if self._pa is not None:
                self._pa.terminate()
        except Exception:
            pass
        finally:
            self._pa = None

    def _find_device_by_name(self, name: str) -> int:
        name_lower = name.lower()
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if name_lower in info['name'].lower():
                return i
        self.get_logger().warn(f'Устройство "{name}" не найдено, использую системное по умолчанию')
        return -1

    # ── Чтение в фоновом треде (не блокирует executor) ───────────────────

    def _stream_reader(self):
        """Фоновый тред: читает аудио из PyAudio в очередь.
        Блокирующий stream.read() изолирован от ROS2 executor — watchdog может сработать."""
        while self._stream_running:
            if self._stream is None:
                time.sleep(0.05)
                continue
            try:
                data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                try:
                    self._audio_queue.put_nowait(data)
                except queue.Full:
                    pass
            except OSError as e:
                self.get_logger().warn(f'Ошибка чтения аудио: {e}')
                break  # Watchdog обнаружит паузу и перезапустит поток

    # Счётчик подряд нулевых чанков — для детекции аппаратного Mute
    _zero_streak: int = 0
    _ZERO_WARN_AFTER: int = 100   # ~3с при chunk=512, rate=16000

    def _drain_and_publish(self):
        """Таймер-колбэк: забирает чанки из очереди и публикует (не блокирует)."""
        try:
            data = self._audio_queue.get_nowait()
        except queue.Empty:
            return

        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # Детекция аппаратного Mute (кнопка на Jabra): устройство возвращает
        # строго нулевые данные. Не прекращаем публикацию, но предупреждаем.
        if audio.max() == 0.0 and audio.min() == 0.0:
            self._zero_streak += 1
            if self._zero_streak == self._ZERO_WARN_AFTER:
                self.get_logger().warn(
                    'AudioSource: микрофон возвращает нули — '
                    'проверьте кнопку Mute на Jabra (красный индикатор)')
            elif self._zero_streak > self._ZERO_WARN_AFTER and self._zero_streak % 300 == 0:
                self.get_logger().warn('AudioSource: микрофон всё ещё замьючен (кнопка Mute)')
            if self._zero_streak == self._zero_wp_check_count:
                threading.Thread(
                    target=self._check_and_maybe_restart_wireplumber,
                    daemon=True, name='wp_check').start()
        else:
            if self._zero_streak >= self._ZERO_WARN_AFTER:
                self.get_logger().info('AudioSource: микрофон разблокирован — сигнал появился')
            self._zero_streak = 0

        msg = Float32MultiArray()
        dim = MultiArrayDimension()
        dim.label  = 'sample_rate'
        dim.size   = len(audio)
        dim.stride = self.rate
        msg.layout.dim = [dim]
        msg.data = audio.tolist()

        self._pub.publish(msg)
        self._last_ok = time.time()

    # ── Watchdog — runtime recovery (поток умер в процессе работы) ────────

    def _watchdog(self):
        if self._stream is None:
            now = time.time()
            if now < self._next_retry_time:
                return  # still in backoff
            self.get_logger().warn(
                f'Watchdog: поток не открыт — перезапускаю (backoff={self._retry_backoff_sec:.0f}с)')
            self._next_retry_time = now + self._retry_backoff_sec
            self._retry_backoff_sec = min(self._retry_backoff_sec * 2, self._MAX_BACKOFF_SEC)
            self._restart_stream()
            return

        elapsed = time.time() - self._last_ok
        if elapsed > self._watchdog_sec:
            self.get_logger().warn(
                f'Watchdog: нет аудио {elapsed:.1f}с — перезапускаю поток'
            )
            self._restart_stream()

    def _restart_stream(self):
        if self._restart_in_progress:
            return
        self._restart_in_progress = True
        try:
            self._restart_count += 1

            now = time.time()
            self._restart_times.append(now)
            if len(self._restart_times) >= self._RESTART_STORM_COUNT and \
                    now - self._restart_times[0] < self._RESTART_STORM_WINDOW_SEC:
                self.get_logger().fatal(
                    f'AudioSource: {len(self._restart_times)} рестартов за '
                    f'{now - self._restart_times[0]:.1f}с — похоже на зависший '
                    f'reader-тред в busy-retry петле (не лечится внутри процесса). '
                    f'Аварийный перезапуск процесса.'
                )
                import sys
                sys.stderr.flush()
                os._exit(1)

            self.get_logger().info(f'Перезапуск аудио потока (попытка #{self._restart_count})...')
            # Сигнализируем reader-треду остановиться и НЕМЕДЛЕННО забываем старый stream/pa.
            # НЕ вызываем stop_stream()/close()/terminate() — если тред заблокирован в read(),
            # любое обращение к PA контексту из другого треда вызовет SIGABRT.
            # Старый тред (daemon) умрёт сам когда PipeWire восстановится и вернёт OSError.
            self._stream_running = False
            self._stream = None   # Reader увидит None и выйдет из следующего цикла
            self._pa     = None   # GC уберёт без явного terminate()
            self._read_thread = None
            # Сбрасываем очередь
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break
            if self._open_stream():
                self._stream_running = True
                self._read_thread = threading.Thread(
                    target=self._stream_reader, daemon=True, name='audio_reader')
                self._read_thread.start()
                self._retry_backoff_sec = 5.0  # reset backoff on success
                self._next_retry_time   = 0.0
                self.get_logger().info('Аудио поток восстановлен')
            else:
                self.get_logger().error('Перезапуск не удался — проверяю PipeWire...')
                threading.Thread(
                    target=self._check_and_maybe_restart_wireplumber,
                    daemon=True, name='wp_check').start()
        finally:
            self._restart_in_progress = False


    # ── PipeWire / WirePlumber recovery ──────────────────────────────────────

    def _pipewire_jabra_state(self):
        """Проверяет состояние Jabra в PipeWire.
        Возвращает (has_card, card_id, profile_ok, source_ok)."""
        if not self._jabra_card_name:
            return True, '', True, True
        has_card, card_id, profile_ok, source_ok = False, '', False, False
        try:
            raw = subprocess.check_output(['pactl', 'list', 'cards'], timeout=5, text=True)
        except Exception as e:
            self.get_logger().warn(f'pactl list cards: {e}')
            return False, '', False, False

        for sec in re.split(r'\n(?=Card #)', raw):
            if self._jabra_card_name.lower() not in sec.lower():
                continue
            has_card = True
            m = re.search(r'Name:\s*(\S+)', sec)
            if m:
                card_id = m.group(1)
            m = re.search(r'Active Profile:\s*(\S+)', sec)
            if m:
                profile_ok = self._jabra_profile.lower() == m.group(1).lower()
            break

        if has_card:
            try:
                src = subprocess.check_output(
                    ['pactl', 'get-default-source'], timeout=3, text=True).strip()
                check = self._pa_source_check or self._jabra_card_name
                source_ok = check.lower() in src.lower()
            except Exception as e:
                self.get_logger().warn(f'pactl get-default-source: {e}')

        return has_card, card_id, profile_ok, source_ok

    def _check_and_maybe_restart_wireplumber(self):
        """Проверяет PipeWire и перезапускает WirePlumber если профиль/source нарушены.
        Запускается в daemon-треде."""
        if not self._jabra_card_name:
            return
        has_card, card_id, profile_ok, source_ok = self._pipewire_jabra_state()
        if has_card and profile_ok and source_ok:
            self.get_logger().info(
                'PipeWire в порядке — причина нулей: кнопка Mute на Jabra')
            return
        reasons = []
        if not has_card:
            reasons.append('карта не найдена')
        elif not profile_ok:
            reasons.append(f'неверный профиль (ожидается {self._jabra_profile})')
        if not source_ok:
            reasons.append('source не активен')
        self.get_logger().error(
            f'PipeWire проблема ({", ".join(reasons)}) — перезапускаю WirePlumber')
        self._restart_wireplumber()

    def _restart_wireplumber(self):
        """Перезапускает WirePlumber, восстанавливает профиль и PCM.
        Вызывается из daemon-треда, защищён cooldown-ом."""
        with self._recovery_lock:
            now = time.time()
            if now - self._last_wp_restart < self._wp_restart_cooldown:
                remaining = self._wp_restart_cooldown - (now - self._last_wp_restart)
                self.get_logger().info(
                    f'WP cooldown: следующий перезапуск через {remaining:.0f}с')
                return
            self._last_wp_restart = now

        self.get_logger().warn('Перезапуск WirePlumber...')
        try:
            subprocess.run(
                ['systemctl', '--user', 'restart', 'wireplumber'],
                timeout=20, check=True)
        except Exception as e:
            self.get_logger().error(f'Не удалось перезапустить WirePlumber: {e}')
            return

        # Ждём появления Jabra source (до 10с)
        card_id = ''
        source_ok = False
        for _ in range(20):
            time.sleep(0.5)
            _, card_id, _, source_ok = self._pipewire_jabra_state()
            if source_ok:
                break
        else:
            self.get_logger().warn('Jabra source не появился за 10с после перезапуска WP')

        # Исправляем профиль если нужно
        _, card_id, profile_ok, _ = self._pipewire_jabra_state()
        if card_id and not profile_ok:
            try:
                subprocess.run(
                    ['pactl', 'set-card-profile', card_id, self._jabra_profile],
                    timeout=5, check=True)
                self.get_logger().info(f'Профиль восстановлен: {self._jabra_profile}')
            except Exception as e:
                self.get_logger().warn(f'Не удалось установить профиль: {e}')

        # PCM=100% (ищем карту по имени чтобы не зашивать индекс)
        try:
            aplay = subprocess.run(
                ['aplay', '-l'], capture_output=True, text=True, timeout=5)
            for line in aplay.stdout.splitlines():
                if 'jabra' in line.lower():
                    m = re.search(r'card (\d+):', line)
                    if m:
                        subprocess.run(
                            ['amixer', '-c', m.group(1), 'set', 'PCM', '100%'],
                            timeout=5)
                        self.get_logger().info(f'PCM card {m.group(1)} → 100%')
                        break
        except Exception as e:
            self.get_logger().warn(f'amixer PCM: {e}')

        self.get_logger().info('WirePlumber восстановлен — перезапускаю аудио поток')
        self._restart_stream()


def main():
    rclpy.init()
    node = AudioSourceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
