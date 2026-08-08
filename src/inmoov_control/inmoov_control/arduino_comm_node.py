"""
arduino_comm_node.py — Base class for InMoov Arduino communication nodes.

Replaces xicro for servo control. Handles:
  TX: /joint_command  (JointState) → batch servo packet over serial
      /face_command   (JointState) → batch face servo packet (Left Arduino only)
  RX: sensor data from Arduino   → individual ROS topics

Subclasses define which joints belong to which Arduino and which topics
to publish for sensor data.
"""

import math
import os
import threading
import time
import serial

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from std_msgs.msg import Int16, Bool
from sensor_msgs.msg import JointState

from .protocol import (
    FrameParser, build_set_servos, build_set_speeds, deg_per_sec_to_step,
    CMD_ULTRASONIC, CMD_PIR, CMD_ACK
)


def deg_to_rad(deg: float, center: float = 90.0) -> float:
    return (deg - center) * math.pi / 180.0


def rad_to_deg(rad: float, center: float = 90.0) -> float:
    return rad * 180.0 / math.pi + center


def clamp_deg(v: float) -> int:
    return max(0, min(180, int(round(v))))


class ArduinoCommNode(LifecycleNode):
    """
    Base class. Subclass and override:
      BODY_JOINTS  — list of (joint_name, center_deg) in packet order
      FACE_JOINTS  — list of (joint_name, center_deg) in packet order, or []
      HAS_ULTRASONIC — publish /ultrasonicXXX_distance (Int16, cm)
      HAS_PIR        — publish /pir_state (Bool)
      ULTRASONIC_TOPIC, PIR_TOPIC — topic names for sensors
    """

    BODY_JOINTS:     list[tuple[str, float, int]] = []   # (joint_name, center_deg, rest_deg)
    FACE_JOINTS:     list[tuple[str, float, int]] = []
    HAS_ULTRASONIC:  bool = False
    HAS_PIR:         bool = False
    ULTRASONIC_TOPIC: str = 'ultrasonic_distance'
    PIR_TOPIC:        str = 'pir_state'

    # Парные суставы: когда приходит команда на ключ — применяем то же значение на значение.
    # Зеркалирование глаз: eye_lr_L ↔ eye_lr_R, eye_ud_L ↔ eye_ud_R.
    EYE_SYNC: dict[str, str] = {
        'eye_lr_L': 'eye_lr_R',
        'eye_lr_R': 'eye_lr_L',
        'eye_ud_L': 'eye_ud_R',
        'eye_ud_R': 'eye_ud_L',
    }

    def __init__(self, node_name: str, serial_port: str, baudrate: int = 115200):
        super().__init__(node_name)

        self._serial_port = serial_port
        self._baudrate    = baudrate

        # Servo state — initialised to rest positions
        self._body_degs    = [rest for _, _, rest in self.BODY_JOINTS]
        self._face_degs    = [rest for _, _, rest in self.FACE_JOINTS]
        self._body_speeds  = [0] * len(self.BODY_JOINTS)
        self._face_speeds  = [0] * len(self.FACE_JOINTS)
        self._speeds_dirty = False

        self._lock = threading.Lock()

        # Joint maps
        self._body_map = {name: i for i, (name, _, _) in enumerate(self.BODY_JOINTS)}
        self._face_map = {name: i for i, (name, _, _) in enumerate(self.FACE_JOINTS)}

        # Serial / RX state
        self._ser        = None
        self._parser     = FrameParser()
        self._running    = False   # RX thread stop flag
        self._rx_thread  = None
        self._tx_timer   = None

        # Publisher handles (set in on_configure)
        self._ultrasonic_pub = None
        self._pir_pub        = None

    # ── Lifecycle callbacks ────────────────────────────────────────────────

    def on_configure(self, state):
        self.create_subscription(JointState, '/joint_command', self._joint_cmd_cb, 10)
        if self.FACE_JOINTS:
            self.create_subscription(JointState, '/face_command', self._face_cmd_cb, 10)

        if self.HAS_ULTRASONIC:
            self._ultrasonic_pub = self.create_lifecycle_publisher(
                Int16, self.ULTRASONIC_TOPIC, 10)
        if self.HAS_PIR:
            self._pir_pub = self.create_lifecycle_publisher(Bool, self.PIR_TOPIC, 10)

        node_name = self.get_name()
        body_names = [n for n, _, _ in self.BODY_JOINTS]
        self.get_logger().info(
            f'{node_name} configured | port={self._serial_port} | '
            f'body={len(self.BODY_JOINTS)}, face={len(self.FACE_JOINTS)} joints')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        if self._ultrasonic_pub:
            self._ultrasonic_pub.on_activate(state)
        if self._pir_pub:
            self._pir_pub.on_activate(state)

        # Проверяем что порт существует в файловой системе
        if not os.path.exists(self._serial_port):
            self.get_logger().error(
                f'Serial port not found: {self._serial_port} → FAILURE (retry)')
            if self._ultrasonic_pub:
                self._ultrasonic_pub.on_deactivate(state)
            if self._pir_pub:
                self._pir_pub.on_deactivate(state)
            return TransitionCallbackReturn.FAILURE

        self._connect_serial()
        if self._ser is None:
            if self._ultrasonic_pub:
                self._ultrasonic_pub.on_deactivate(state)
            if self._pir_pub:
                self._pir_pub.on_deactivate(state)
            return TransitionCallbackReturn.FAILURE

        self._tx_timer = self.create_timer(0.02, self._send_servos)

        self._running   = True
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

        self.get_logger().info(
            f'{self.get_name()} active | port={self._serial_port}')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        # Отправляем сервы в позицию покоя
        self._send_rest_positions()

        # Останавливаем RX поток
        self._running = False
        if self._rx_thread and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=2.0)
        self._rx_thread = None

        # Останавливаем TX таймер
        if self._tx_timer:
            self.destroy_timer(self._tx_timer)
            self._tx_timer = None

        # Закрываем serial
        self._close_serial()

        if self._ultrasonic_pub:
            self._ultrasonic_pub.on_deactivate(state)
        if self._pir_pub:
            self._pir_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self._close_serial()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self._send_rest_positions()
        self._close_serial()
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        self._close_serial()
        return TransitionCallbackReturn.SUCCESS

    def _send_rest_positions(self):
        rest_body = [rest for _, _, rest in self.BODY_JOINTS]
        rest_face = [rest for _, _, rest in self.FACE_JOINTS]
        if self._ser and self._ser.is_open:
            try:
                frame = build_set_servos(rest_body + rest_face)
                self._ser.write(frame)
                self._ser.flush()
                time.sleep(0.15)
            except Exception:
                pass

    def _close_serial(self):
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
        except Exception:
            pass
        finally:
            self._ser = None

    # -----------------------------------------------------------------------
    # Serial connection
    # -----------------------------------------------------------------------

    def _connect_serial(self) -> None:
        try:
            self._ser = serial.Serial(
                self._serial_port, self._baudrate, timeout=0.1
            )
            self.get_logger().info(f'Serial connected: {self._serial_port}')
            time.sleep(2.0)           # wait for Arduino bootloader
            # No reset_input_buffer — FrameParser discards bootloader garbage via SOF check
        except serial.SerialException as e:
            self.get_logger().error(f'Serial open failed: {e}')
            self._ser = None

    # -----------------------------------------------------------------------
    # RX loop (sensor data from Arduino)
    # -----------------------------------------------------------------------

    def _rx_loop(self) -> None:
        try:
            while self._running:
                if self._ser is None or not self._ser.is_open:
                    time.sleep(0.1)
                    continue
                try:
                    chunk = self._ser.read(64)
                except serial.SerialException as e:
                    self.get_logger().error(f'Serial read error: {e}')
                    self._ser = None
                    continue

                if not chunk:
                    continue

                self._parser.push(chunk)
                for cmd, data in self._parser.frames:
                    self._handle_rx(cmd, data)
                self._parser.frames.clear()
        except Exception as e:
            self.get_logger().error(f'RX thread crashed: {e}', exc_info=True)

    def _handle_rx(self, cmd: int, data: bytes) -> None:
        if cmd == CMD_ULTRASONIC and self._ultrasonic_pub and len(data) >= 2:
            dist_cm = (data[0] << 8) | data[1]
            msg = Int16()
            msg.data = dist_cm
            self._ultrasonic_pub.publish(msg)

        elif cmd == CMD_PIR and self._pir_pub and len(data) >= 1:
            msg = Bool()
            msg.data = bool(data[0])
            self._pir_pub.publish(msg)

    # -----------------------------------------------------------------------
    # Command callbacks
    # -----------------------------------------------------------------------

    def _apply_joint(self, name: str, pos_rad: float, vel: float) -> None:
        """Применяет позицию к body или face карте. Вызывается под self._lock."""
        if name in self._body_map:
            i = self._body_map[name]
            _, center, _ = self.BODY_JOINTS[i]
            self._body_degs[i] = clamp_deg(rad_to_deg(pos_rad, center))
            if vel != 0.0:
                step = deg_per_sec_to_step(math.degrees(abs(vel)))
                if step != self._body_speeds[i]:
                    self._body_speeds[i] = step
                    self._speeds_dirty = True
        elif name in self._face_map:
            i = self._face_map[name]
            _, center, _ = self.FACE_JOINTS[i]
            self._face_degs[i] = clamp_deg(rad_to_deg(pos_rad, center))
            if vel != 0.0:
                step = deg_per_sec_to_step(math.degrees(abs(vel)))
                if step != self._face_speeds[i]:
                    self._face_speeds[i] = step
                    self._speeds_dirty = True

    def _joint_cmd_cb(self, msg: JointState) -> None:
        with self._lock:
            vels = msg.velocity
            for idx, (name, pos_rad) in enumerate(zip(msg.name, msg.position)):
                vel = vels[idx] if idx < len(vels) else 0.0
                self._apply_joint(name, pos_rad, vel)
                # Синхронизация парного глаза (eye_lr_L ↔ eye_lr_R, eye_ud_L ↔ eye_ud_R)
                mirror = self.EYE_SYNC.get(name)
                if mirror:
                    self._apply_joint(mirror, pos_rad, vel)

    def _face_cmd_cb(self, msg: JointState) -> None:
        with self._lock:
            vels = msg.velocity
            for idx, (name, pos_rad) in enumerate(zip(msg.name, msg.position)):
                vel = vels[idx] if idx < len(vels) else 0.0
                self._apply_joint(name, pos_rad, vel)
                # Синхронизация парного глаза
                mirror = self.EYE_SYNC.get(name)
                if mirror:
                    self._apply_joint(mirror, pos_rad, vel)

    # -----------------------------------------------------------------------
    # TX: send servo packet at 50 Hz
    # -----------------------------------------------------------------------

    def _send_servos(self) -> None:
        if self._ser is None or not self._ser.is_open:
            return

        with self._lock:
            body = list(self._body_degs)
            face = list(self._face_degs)
            dirty = self._speeds_dirty
            if dirty:
                body_spd = list(self._body_speeds)
                face_spd = list(self._face_speeds)
                self._speeds_dirty = False

        try:
            # Send speed update first (only when something changed)
            if dirty:
                spd_frame = build_set_speeds(body_spd + face_spd)
                self._ser.write(spd_frame)
            # Send servo positions
            frame = build_set_servos(body + face)
            self._ser.write(frame)
            self._ser.flush()
        except serial.SerialException as e:
            self.get_logger().error(f'Serial write error: {e}')
            self._ser = None

    # destroy_node заменён на on_shutdown / on_deactivate (lifecycle)
