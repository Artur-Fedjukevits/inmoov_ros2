#!/usr/bin/env python3

"""
sound_localization_node.py
===========================
Грубое направление на источник звука по паре микрофонов MAX9814 на
звуковой карте CM6206 (USB 0d8c:0102, ALSA-имя "ICUSBAUDIO7D") —
через межканальную разницу громкости (ILD), НЕ через время прихода
(TDOA/GCC-PHAT).

Почему не TDOA — история (2026-08-19..22, см.
project_sound_localization_gcc_phat.md): на открытом столе GCC-PHAT
давал чистые и правильные ±90°. После установки капсюлей в уши робота
результат стабильно уходил в 0° независимо от реального направления —
оба капсюля на реальной голове акустически связаны через общую открытую
полость черепа (сервоприводы, провода, шея — видно на фото сборки).
Герметизация уха (силиконовый герметик вокруг капсюля) заметно снизила,
но не убрала утечку: измеренная задержка регулярно ПРЕВЫШАЛА физически
возможный максимум для базы между ушами — то есть алгоритм мерил не
прямой путь звука, а переотражения внутри черепа. Проверили и
электрическую наводку между каналами (перепаивали провод перед
установкой) — низкая корреляция (0.04-0.06) её исключает, дело именно
в акустике. Разница громкости оказалась единственным сигналом, который
на реальной речи (10с, естественные паузы) стабильно смещался в нужную
сторону (~2.7-3.4дБ) — на неё и переписано.

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

60% подобрано 2026-08-22 под реальную дистанцию разговора (1-3м, не
5-6см калибровочных тестов) — на 70% пик речи почти клиппировал
(-0.6дБFS, меньше 1дБ до потолка), на 50% медиана речи была ниже гейта.
Пол комнаты на 60% гейне ~-30.5дБFS (стабильно, ±1дБ) — важно: подъём
аналогового гейна усиливает и полезный сигнал, и акустический шум
одновременно (SNR относительно акустического шума почти не меняется от
гейна) — гейн двигали не ради SNR, а чтобы уровень речи с реальной
дистанции попадал выше `rms_gate_dbfs` и не тонул в порогах ноды.

Алгоритм на блок:
  1. RMS каждого канала отдельно + средний RMS обоих (для гейта)
  2. ild_db_raw = 20·log10(rms_right / rms_left)
  3. ild_db = ild_db_raw - ild_bias_db  — компенсация систематического
     перекоса: правый капсюль акустически связан с внешним звуком лучше
     левого (плотнее сидит в ухе — левый переклеивали после щели, правый
     не трогали), из-за чего сырой ild_db стабильно смещён в "право"
     примерно на +4.5дБ независимо от реального направления. Замерено
     2026-08-22: белый шум с телефона (5-6см) в ПОЛНОЙ ТИШИНЕ (без
     фонового 3D-принтера, который в первой попытке дал ложную картину —
     см. память) — источник у левого уха даёт ild_db_raw=-4.1дБ, у
     правого +13.1дБ; симметричная компенсация даёт офсет 4.5дБ.
  4. angle_deg = clip(ild_db / max_ild_db, -1, 1) · 90°
     — ЛИНЕЙНОЕ отображение, не физическая модель (в отличие от старой
     asin(tdoa·c/d) формулы GCC-PHAT-версии) — просто разумный масштаб
     для совместимости с потребителями топика. max_ild_db — во сколько
     дБ разницы считать "предельно в одну сторону", подбирается
     эмпирически под конкретную голову.

ВАЖНО — калибровка перед использованием на голове:
Знак ild_db/angle_deg зависит от того, какой физический капсюль
подключён в какой ALSA-канал (0=left/FL, 1=right/FR). Проверить руками
(говорить/шипеть с известной стороны, смотреть знак angle_deg) и при
необходимости выставить swap_channels:=true, а не лезть в код.

Дефолт swap_channels=True подобран 2026-08-22 после перепайки провода
(заделка щели левого уха задела и разводку) — каналы физически оказались
перепутаны. Подтверждено на обоих ушах: левое (без swap) даёт
отрицательный angle_deg на живой речи (-38.7°), правое (с swap) даёт
устойчиво +90° на шипении (65/65 блоков). Если карту/мики/пайку ещё раз
трогали — перекалибровать заново тем же способом, знак может опять
измениться.

Практический вывод для потребителей топика: одиночному блоку не
доверять, громкость речи и так прыгает от слова к слову — агрегировать
(медиана/EMA) по нескольким подряд идущим voiced=true блокам. Нода уже
сглаживает ild_db через EMA (см. параметр ema_alpha) для базовой
стабильности, но резкие скачки на паузах в речи всё равно возможны.

Топик:
  /sound_direction  (inmoov_msgs/SoundDirection)

Отдельный запуск для теста (без launch-файла):
  ros2 run inmoov_voice sound_localization_node --ros-args -p max_ild_db:=6.0
  ros2 topic echo /sound_direction
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
import sounddevice as sd

from inmoov_msgs.msg import SoundDirection


class SoundLocalizationNode(Node):

    def __init__(self):
        super().__init__('sound_localization_node')

        self.declare_parameter('device_name', 'ICUSBAUDIO7D')
        self.declare_parameter('sample_rate', 48000)
        self.declare_parameter('block_size', 4096)     # ~85мс @ 48кГц — ILD не нужна тонкая временная точность GCC-PHAT
        self.declare_parameter('max_ild_db', 6.0)       # дБ разницы, соответствующие ±90°; подбирается под голову
        self.declare_parameter('ild_bias_db', 4.5)      # системный сдвиг показаний в "право" (правый капсюль акустически связан лучше левого — см. память 2026-08-22), вычитается из сырого ild_db
        self.declare_parameter('rms_gate_dbfs', -24.0)  # под гейн 60% и реальную дистанцию разговора 1-3м (см. память 2026-08-22, пол комнаты ~-30.5дБFS на этом гейне)
        self.declare_parameter('publish_silence', False)
        self.declare_parameter('swap_channels', True)   # каналы физически перепутаны после перепайки провода при заделке уха (2026-08-22, см. память)
        self.declare_parameter('ema_alpha', 0.3)        # сглаживание ild_db между блоками, 0..1 (больше = быстрее реакция)
        self.declare_parameter('watchdog_sec', 3.0)

        self._device_name     = self.get_parameter('device_name').value
        self.rate              = self.get_parameter('sample_rate').value
        self.block_size        = self.get_parameter('block_size').value
        self._max_ild_db       = self.get_parameter('max_ild_db').value
        self._ild_bias_db      = self.get_parameter('ild_bias_db').value
        self._rms_gate_dbfs    = self.get_parameter('rms_gate_dbfs').value
        self._publish_silence  = self.get_parameter('publish_silence').value
        self._swap_channels    = self.get_parameter('swap_channels').value
        self._ema_alpha        = self.get_parameter('ema_alpha').value
        self._watchdog_sec     = self.get_parameter('watchdog_sec').value

        self._ild_ema = 0.0
        self._ema_initialized = False

        self._pub = self.create_publisher(SoundDirection, 'sound_direction', 10)

        self._stream = None
        self._last_block_time = 0.0
        self._error_streak = 0

        self._open_stream()
        self._timers = [self.create_timer(self._watchdog_sec, self._watchdog)]

        self.get_logger().info(
            f'SoundLocalization (ILD): rate={self.rate} блок={self.block_size} '
            f'({1000 * self.block_size / self.rate:.0f}мс) '
            f'max_ild_db={self._max_ild_db} rms_gate={self._rms_gate_dbfs}дБFS')

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

    def _on_audio_block(self, indata, frames, time_info, status):
        self._last_block_time = self.get_clock().now().nanoseconds / 1e9
        if status:
            self.get_logger().warn(f'Audio status: {status}')
        try:
            left, right = indata[:, 0], indata[:, 1]
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
            ild_db_raw = 20.0 * math.log10(max(rms_r, 1e-9) / max(rms_l, 1e-9))
            ild_db = ild_db_raw - self._ild_bias_db   # компенсация систематического перекоса каналов (правый капсюль лучше акустически связан — см. память)
            if not self._ema_initialized:
                self._ild_ema = ild_db
                self._ema_initialized = True
            else:
                self._ild_ema = self._ema_alpha * ild_db + (1.0 - self._ema_alpha) * self._ild_ema
            ratio = max(-1.0, min(1.0, self._ild_ema / self._max_ild_db))
            angle_deg = ratio * 90.0
            confidence = max(0.0, min(1.0, (rms_dbfs - self._rms_gate_dbfs) / 20.0))
        else:
            ild_db, angle_deg, confidence = 0.0, 0.0, 0.0

        msg = SoundDirection()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'head'
        msg.angle_deg = float(angle_deg)
        msg.ild_db = float(ild_db if voiced else 0.0)
        msg.confidence = float(confidence)
        msg.rms_dbfs = float(rms_dbfs)
        msg.voiced = bool(voiced)
        self._pub.publish(msg)

    def destroy_node(self):
        for t in getattr(self, '_timers', []):
            self.destroy_timer(t)
        self._close_stream()
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
