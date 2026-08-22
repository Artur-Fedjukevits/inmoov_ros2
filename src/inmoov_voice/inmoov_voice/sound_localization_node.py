#!/usr/bin/env python3

"""
sound_localization_node.py
===========================
Грубое направление на источник звука по паре микрофонов MAX9814 на
звуковой карте CM6206 (USB 0d8c:0102, ALSA-имя "ICUSBAUDIO7D") —
через голосование по ЗНАКУ полосового GCC-PHAT (TDOA), НЕ через
межканальную разницу громкости (ILD) и НЕ через физический угол.

История (см. project_sound_localization_gcc_phat.md за подробностями):
1) Исходный широкополосный GCC-PHAT — чисто работал на открытом столе
   (±90°, шипение), но на установленной в уши голове давал физически
   невозможные задержки: открытый скелет черепа (сервоприводы, провода,
   шея) пропускает звук напрямую между капсюлями, эта внутренняя утечка
   доминирует над прямым путём через воздух снаружи.
2) Пивот на ILD (разница громкости) — рабочий на 5-6см, но на реальной
   дистанции разговора (1-3м, комната ~3x2.5м) деградировал почти до
   случайного знака: отражения от стен перебивают слабую разницу
   громкости от направления (проверено статистически 2026-08-22 —
   несколько подряд идущих замеров дали неверный знак на 1м).
3) 2026-08-22: полосовой фильтр (2-6кГц — где утечка через полость слабее,
   чем на басах) перед GCC-PHAT НЕ убрал избыточную задержку (медианные
   значения всё ещё в разы превышают физический предел для базы между
   ушами), НО её ЗНАК стабильно коррелирует с реальным направлением на
   всех проверенных дистанциях (0.2-2м, шипение и живая речь). Голосование
   большинства по знаку за скользящее окно блоков подтверждено вслепую:
   6/6 верных угадываний в реальном тесте (разные стороны, дистанции,
   45°, центр).

Микрофоны читаются НАПРЯМУЮ через raw ALSA, в обход PipeWire (карта
исключена из WirePlumber правилом device.disabled в
~/.config/wireplumber/main.lua.d/52-cm6206-disable.lua — без этого
PipeWire держит D-Bus резервацию устройства и raw-доступ невозможен).

ВАЖНО — гейн карты НЕ переживает перезагрузку/переподключение (ALSA
alsa-restore срабатывает раньше, чем инициализируется эта USB-карта):
после каждой перезагрузки хоста проверить
`amixer -c ICUSBAUDIO7D contents | grep -A3 "Mic Capture Volume"` —
должно быть **60% (4157/6928, +0.23дБ)**, а не максимум (6928/6928).
Восстановить: `amixer -c ICUSBAUDIO7D sset Mic 60% cap`.
(Пробовали асимметричный гейн по каналам для компенсации разной
акустической связи капсюлей с внешним звуком — не получилось: слишком
чувствительно к точному значению, при перекосе один канал теряет
чувствительность к своему направлению полностью. Симметричный гейн проще
и предсказуемее, конкретное значение (60%) для алгоритма TDOA не так
критично, как было для ILD — это по-прежнему число дБ, симметрия важнее.)

Алгоритм на блок (~85мс @ 48кГц):
  1. RMS блока (оба канала) → energy-гейт (rms_gate_dbfs), как раньше.
  2. Полосовой фильтр 2-6кГц (bandpass_low/high, Butterworth, sosfiltfilt
     — нулевая фазовая задержка, важно для точности TDOA) на оба канала.
  3. GCC-PHAT: Hann-окно → FFT → кросс-спектр XL·conj(XR) → PHAT-нормировка
     (делим на модуль, оставляем только фазу) → IFFT → пик в широком
     окне поиска (±5мс, НЕ ограничен физическим пределом — сам пик всё
     равно окажется вне физических рамок, но его ЗНАК информативен).
  4. Знак сырой задержки (мкс) добавляется в скользящее окно
     (vote_window_sec, по умолчанию 3с) — НЕ усредняем сами задержки
     (величина физически бессмысленна и хаотична), только считаем
     большинство по знаку.
  5. angle_deg = (голосов_право − голосов_лево) / всего_голосов · 90°,
     confidence = |то же соотношение|. НЕ физическая модель.

ВАЖНО — калибровка перед использованием на голове:
Знак зависит от того, какой физический капсюль подключён в какой
ALSA-канал (0=right/FL, 1=left/FR — да, "перевёрнуто" относительно
интуиции, см. память). Проверить руками (говорить/шипеть с известной
стороны, смотреть знак angle_deg) и при необходимости выставить
swap_channels:=false (дефолт true подобран 2026-08-22, тот же маппинг,
что и у ILD-версии — если карту/пайку не трогали после того теста,
менять не нужно).

Практический вывод для потребителей топика: полагаться нужно на
`angle_deg`/`confidence` (уже агрегированы за окно), НЕ на `tdoa_us`
(это медиана за окно чисто для отладки, физически нереалистична).
Низкий confidence (<0.5) означает, что окно ещё не набралось или знак
внутри окна колеблется — стоит подождать ещё немного речи, прежде чем
принимать решение о повороте.

Известные ограничения:
  - Неоднозначность спереди/сзади (общая для любой пары микрофонов) —
    не различает источник спереди и сзади под тем же углом. Разрешать
    через зрение (OAK-D/face_detection) — грубый крен по звуку, точная
    сторона и фронт/зад по камере.
  - Голос заметно шумнее шипения на дистанции >1м (гласные/периодичность
    хуже для PHAT, чем широкополосный шум) — окно голосования сглаживает
    это, но короткие реплики (<1-2с) могут не успеть набрать уверенный
    результат.
  - При долгой тишине окно голосования не сбрасывается само — старое
    направление "подвисает" до следующих голосов. Если после паузы
    заговорил кто-то с другой стороны, первые ~vote_window_sec может
    показывать прошлое направление.
  - Нет on_set_parameters_callback — `ros2 param set` во время работы
    не применяется, только перезапуск с -p.

Топик:
  /sound_direction  (inmoov_msgs/SoundDirection)

Отдельный запуск для теста (без launch-файла):
  ros2 run inmoov_voice sound_localization_node
  ros2 topic echo /sound_direction
"""

import math
import os
import queue
import statistics
import threading
from collections import deque

# Ограничить внутреннюю многопоточность BLAS/numpy ДО импорта numpy/scipy —
# на этом NUC параллельно молотят тяжёлые ноды (face_detection ~60%+ CPU
# на глаз), лишние потоки BLAS только добавляют конкуренцию за ядра и
# затрудняют то, ради чего вообще нужна отдельная очередь (см. ниже).
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
import sounddevice as sd
from scipy.signal import butter, sosfiltfilt

from inmoov_msgs.msg import SoundDirection


class SoundLocalizationNode(Node):

    def __init__(self):
        super().__init__('sound_localization_node')

        self.declare_parameter('device_name', 'ICUSBAUDIO7D')
        self.declare_parameter('sample_rate', 48000)
        self.declare_parameter('block_size', 4096)       # ~85мс @ 48кГц
        self.declare_parameter('bandpass_low_hz', 2000.0)
        self.declare_parameter('bandpass_high_hz', 6000.0)
        self.declare_parameter('mic_distance_m', 0.145)   # только для справки/поиска окна, НЕ используется в angle_deg
        self.declare_parameter('search_window_sec', 0.005)  # ±5мс — заведомо шире физического предела, чтобы не резать сам пик
        self.declare_parameter('vote_window_sec', 3.0)    # скользящее окно голосования по знаку; больше = надёжнее, но медленнее реагирует
        self.declare_parameter('rms_gate_dbfs', -24.0)    # под гейн 60% и реальную дистанцию 1-3м (см. память)
        self.declare_parameter('publish_silence', False)
        self.declare_parameter('swap_channels', True)     # raw ch0=физически правый, ch1=физически левый — см. память
        self.declare_parameter('watchdog_sec', 3.0)

        self._device_name       = self.get_parameter('device_name').value
        self.rate                = self.get_parameter('sample_rate').value
        self.block_size          = self.get_parameter('block_size').value
        self._bandpass_low       = self.get_parameter('bandpass_low_hz').value
        self._bandpass_high      = self.get_parameter('bandpass_high_hz').value
        self._mic_distance_m     = self.get_parameter('mic_distance_m').value
        self._search_window_sec  = self.get_parameter('search_window_sec').value
        self._vote_window_sec    = self.get_parameter('vote_window_sec').value
        self._rms_gate_dbfs      = self.get_parameter('rms_gate_dbfs').value
        self._publish_silence    = self.get_parameter('publish_silence').value
        self._swap_channels      = self.get_parameter('swap_channels').value
        self._watchdog_sec       = self.get_parameter('watchdog_sec').value

        vote_window_blocks = max(1, int(self._vote_window_sec * self.rate / self.block_size))
        self._vote_window = deque(maxlen=vote_window_blocks)

        self._sos = butter(4, [self._bandpass_low, self._bandpass_high],
                            btype='band', fs=self.rate, output='sos')

        self._n_fft = 1
        while self._n_fft < 2 * self.block_size:
            self._n_fft *= 2
        self._hann = np.hanning(self.block_size)

        self._pub = self.create_publisher(SoundDirection, 'sound_direction', 10)

        self._stream = None
        self._last_block_time = 0.0
        self._error_streak = 0

        # Очередь между realtime-колбэком PortAudio и тяжёлой обработкой
        # (полосовой фильтр + FFT/PHAT) — колбэк должен возвращаться быстро,
        # иначе PortAudio сообщает "input overflow" и данные теряются.
        # Обнаружено 2026-08-22: под конкурентной CPU-нагрузкой от других
        # нод (face_detection ~60%+/глаз) даже дешёвая обработка (~0.6мс)
        # внутри колбэка периодически не укладывалась в тайминг — вынесена
        # в отдельный поток, колбэк теперь только копирует блок в очередь.
        self._queue = queue.Queue(maxsize=8)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self._open_stream()
        self._timers = [self.create_timer(self._watchdog_sec, self._watchdog)]

        self.get_logger().info(
            f'SoundLocalization (TDOA sign-vote): rate={self.rate} блок={self.block_size} '
            f'({1000 * self.block_size / self.rate:.0f}мс) '
            f'полоса={self._bandpass_low:.0f}-{self._bandpass_high:.0f}Гц '
            f'окно_голосования={self._vote_window_sec}с ({vote_window_blocks} блоков) '
            f'rms_gate={self._rms_gate_dbfs}дБFS')

    # ── Открытие устройства ────────────────────────────────────────────────

    def _find_device_index(self):
        for i, d in enumerate(sd.query_devices()):
            if self._device_name.lower() in d['name'].lower() and d['max_input_channels'] >= 2:
                return i
        return None

    def _open_stream(self) -> bool:
        self._close_stream()

        idx = self._find_device_index()
        if idx is None:
            self.get_logger().error(
                f'Устройство "{self._device_name}" с 2 входными каналами не найдено. '
                f'Проверь: карта исключена из WirePlumber? (wpctl status не должен её '
                f'показывать); лежит ли она всё ещё на hw:CARD={self._device_name}?')
            return False

        try:
            self.get_logger().info(f'Открываю [{idx}] {sd.query_devices()[idx]["name"]}')
            self._stream = sd.InputStream(
                device=idx,
                channels=2,
                samplerate=self.rate,
                blocksize=self.block_size,
                dtype='float32',
                latency=0.2,  # секунды, ЯВНО числом — строка 'high' у этого драйвера маппится всего на ~35мс (мало!), не помогала; 0.2с эмпирически чисто без overflow (2026-08-22, тест 0.1с ещё ловил overflow, 0.15с+ чисто)
                callback=self._on_audio_block,
            )
            self._stream.start()
            self._last_block_time = self.get_clock().now().nanoseconds / 1e9
            return True
        except Exception as e:
            self.get_logger().error(f'Не удалось открыть аудио поток: {e}')
            self._stream = None
            return False

    def _close_stream(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _watchdog(self):
        now = self.get_clock().now().nanoseconds / 1e9
        if self._stream is None or (now - self._last_block_time) > self._watchdog_sec:
            self.get_logger().warn('Watchdog: нет аудио — переоткрываю поток')
            self._open_stream()

    # ── Callback аудио-потока (реалтайм тред PortAudio, НЕ ROS executor) ────
    # Должен быть максимально быстрым — только копия блока в очередь, вся
    # тяжёлая обработка (полосовой фильтр, FFT/PHAT) в _worker_loop().

    def _on_audio_block(self, indata, frames, time_info, status):
        self._last_block_time = self.get_clock().now().nanoseconds / 1e9
        if status:
            self.get_logger().warn(f'Audio status: {status}')
        try:
            self._queue.put_nowait(indata.copy())
        except queue.Full:
            # Обработка не успевает — выкидываем самый старый блок и кладём
            # новый, чтобы очередь не копила задержку бесконечно.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(indata.copy())
            except (queue.Empty, queue.Full):
                pass

    # ── Рабочий поток: тяжёлая обработка вне realtime-колбэка ───────────────

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                block = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                left, right = block[:, 0], block[:, 1]
                if self._swap_channels:
                    left, right = right, left
                self._process_block(left, right)
                self._error_streak = 0
            except Exception as e:
                self._error_streak += 1
                if self._error_streak <= 3 or self._error_streak % 100 == 0:
                    self.get_logger().error(f'Ошибка обработки блока: {e}')

    def _process_block(self, left: np.ndarray, right: np.ndarray):
        rms_l = float(np.sqrt(np.mean(left.astype(np.float64) ** 2)))
        rms_r = float(np.sqrt(np.mean(right.astype(np.float64) ** 2)))
        rms_avg = math.sqrt((rms_l ** 2 + rms_r ** 2) / 2.0)
        rms_dbfs = 20.0 * math.log10(max(rms_avg, 1e-9))
        voiced = rms_dbfs >= self._rms_gate_dbfs

        if not voiced and not self._publish_silence:
            return

        if voiced:
            tdoa_us = self._gcc_phat_tdoa_us(left, right)
            self._vote_window.append(tdoa_us)

            total = len(self._vote_window)
            pos = sum(1 for x in self._vote_window if x > 0)
            neg = sum(1 for x in self._vote_window if x < 0)
            score = (pos - neg) / total if total > 0 else 0.0
            angle_deg = score * 90.0
            confidence = abs(score)
            tdoa_us_median = statistics.median(self._vote_window)
        else:
            tdoa_us_median, angle_deg, confidence = 0.0, 0.0, 0.0

        msg = SoundDirection()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'head'
        msg.angle_deg = float(angle_deg)
        msg.tdoa_us = float(tdoa_us_median)
        msg.confidence = float(confidence)
        msg.rms_dbfs = float(rms_dbfs)
        msg.voiced = bool(voiced)
        self._pub.publish(msg)

    def _gcc_phat_tdoa_us(self, left: np.ndarray, right: np.ndarray) -> float:
        """Полосовой GCC-PHAT, возвращает знаковую задержку в мкс (сырую,
        физически нереалистичную по величине из-за утечки звука через
        полость черепа — использовать только ЗНАК, см. докстринг модуля)."""
        fl = sosfiltfilt(self._sos, left.astype(np.float64))
        fr = sosfiltfilt(self._sos, right.astype(np.float64))

        fl_w = fl * self._hann
        fr_w = fr * self._hann

        XL = np.fft.rfft(fl_w, n=self._n_fft)
        XR = np.fft.rfft(fr_w, n=self._n_fft)
        R = XL * np.conj(XR)
        R_phat = R / (np.abs(R) + 1e-12)
        r = np.fft.fftshift(np.fft.irfft(R_phat, n=self._n_fft))
        center = self._n_fft // 2

        wide = max(1, int(self._search_window_sec * self.rate))
        lo, hi = center - wide, center + wide
        lag = int(np.argmax(r[lo:hi])) + lo - center
        return lag / self.rate * 1e6

    def destroy_node(self):
        for t in getattr(self, '_timers', []):
            self.destroy_timer(t)
        self._close_stream()
        self._stop_event.set()
        self._worker.join(timeout=2.0)
        super().destroy_node()


def main():
    rclpy.init()
    node = SoundLocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
