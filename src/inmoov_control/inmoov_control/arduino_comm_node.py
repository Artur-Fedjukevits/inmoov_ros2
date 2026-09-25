"""
arduino_comm_node.py — Base class for InMoov Arduino communication nodes.

Replaces xicro for servo control. Handles:
  TX: /joint_cmd      (inmoov_msgs/JointCommand) → arbitrated per joint (JointArbiter:
                      priority + lease, see JointCommand.msg) → batch servo packet
      /joint_command, /face_command (JointState) — legacy, priority 0, no lease
  RX: sensor data from Arduino   → individual ROS topics
  /joint_commanded (JointState) — the commands actually ACCEPTED by the arbiter
      (for nodes that track the current pose: head tracker, BT, joint_state_publisher)
  ~/joint_owners   (String JSON, 1 Hz) — live leases, for debugging

A command without velocity (or 0) moves the joint at its default speed (the
firmware table step, DEFAULT_STEPS) — speed is no longer inherited from
whatever the previous sender set.

Subclasses define which joints belong to which Arduino and which topics
to publish for sensor data.

Link robustness:
  - A serial error (unplugged board, USB reset) closes the port; the RX thread
    then reopens it with a growing back-off (_RECONNECT_MIN_SEC → _RECONNECT_MAX_SEC)
    while the node stays active.
  - The firmware returns to rest by itself if SET_SERVOS frames stop for 1.5 s
    (host-loss failsafe); the 50 Hz TX stream is its heartbeat. Transitions are
    reported as CMD_STATUS and published on ~/failsafe (latched Bool).
  - The sleep flag and the speed table are re-sent after every (re)connect and
    every _STATE_RESEND_SEC — an Arduino that reset on its own (firmware
    defaults: awake, table speeds) is resynchronised without a USB drop.
    Both commands are idempotent in the firmware (speed 0 = keep).

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import json
import math
import os
import threading
import time
import serial

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Int16, Int16MultiArray, Bool, String
from sensor_msgs.msg import JointState
from inmoov_msgs.msg import JointCommand

from .joint_arbiter import JointArbiter

from .protocol import (
    FrameParser, build_set_servos, build_set_speeds, build_sleep, deg_per_sec_to_step,
    CMD_ULTRASONIC, CMD_PIR, CMD_HALL, CMD_STATUS, CMD_ACK
)

_RECONNECT_MIN_SEC = 2.0    # first reopen attempt after a serial error
_RECONNECT_MAX_SEC = 30.0   # back-off cap
_STATE_RESEND_SEC  = 2.0    # periodic re-send of sleep flag + speeds


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
      HAS_HALL       — publish HALL_TOPIC (Int16MultiArray, raw analogRead per finger)
      ULTRASONIC_TOPIC, PIR_TOPIC, HALL_TOPIC — topic names for sensors

    All subclasses subscribe to /robot_sleep (latched Bool) and forward it to
    the Arduino as a CMD_SLEEP frame — while asleep, the firmware stops
    sending CMD_ULTRASONIC/CMD_PIR/CMD_HALL telemetry.
    """

    BODY_JOINTS:     list[tuple[str, float, int]] = []   # (joint_name, center_deg, rest_deg)
    FACE_JOINTS:     list[tuple[str, float, int]] = []
    # Firmware table step per servo, BODY_JOINTS + FACE_JOINTS order (test_servo_tables)
    DEFAULT_STEPS:   list[int] = []
    HAS_ULTRASONIC:  bool = False
    HAS_PIR:         bool = False
    HAS_HALL:        bool = False
    ULTRASONIC_TOPIC: str = 'ultrasonic_distance'
    PIR_TOPIC:        str = 'pir_state'
    HALL_TOPIC:       str = 'hall_raw'

    # Paired joints: when a command arrives for the key joint, apply the same
    # value to the mirrored joint. Eye mirroring: eye_lr_L <-> eye_lr_R, eye_ud_L <-> eye_ud_R.
    EYE_SYNC: dict[str, str] = {
        'eye_lr_L': 'eye_lr_R',
        'eye_lr_R': 'eye_lr_L',
        'eye_ud_L': 'eye_ud_R',
        'eye_ud_R': 'eye_ud_L',
    }
    # A lease on one eye covers its mirror
    _ARBITER_GROUPS = {'eye_lr_L': 'eye_lr', 'eye_lr_R': 'eye_lr',
                       'eye_ud_L': 'eye_ud', 'eye_ud_R': 'eye_ud'}

    def __init__(self, node_name: str, serial_port: str, baudrate: int = 115200):
        super().__init__(node_name)

        self._serial_port = serial_port
        self._baudrate    = baudrate

        # Servo state — initialised to rest positions
        self._body_degs    = [rest for _, _, rest in self.BODY_JOINTS]
        self._face_degs    = [rest for _, _, rest in self.FACE_JOINTS]
        n_body = len(self.BODY_JOINTS)
        steps  = self.DEFAULT_STEPS or [0] * (n_body + len(self.FACE_JOINTS))
        self._default_body_steps = list(steps[:n_body])
        self._default_face_steps = list(steps[n_body:])
        self._body_speeds  = list(self._default_body_steps)
        self._face_speeds  = list(self._default_face_steps)
        self._speeds_dirty = False
        self._arbiter      = JointArbiter(self._ARBITER_GROUPS)

        # Sleep state — forwarded to Arduino as CMD_SLEEP on change
        self._sleeping     = False
        self._sleep_dirty  = False

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
        self._last_state_resend = 0.0

        # Subscription/publisher handles (set in on_configure, destroyed in on_cleanup)
        self._subs           = []
        self._ultrasonic_pub = None
        self._pir_pub        = None
        self._hall_pub       = None
        self._failsafe_pub   = None
        self._commanded_pub  = None
        self._owners_pub     = None
        self._owners_timer   = None

    # ── Lifecycle callbacks ────────────────────────────────────────────────

    def on_configure(self, state):
        self._subs = [
            self.create_subscription(JointCommand, '/joint_cmd', self._arb_cmd_cb, 20),
            self.create_subscription(JointState, '/joint_command', self._legacy_cmd_cb, 10)]
        if self.FACE_JOINTS:
            self._subs.append(
                self.create_subscription(JointState, '/face_command', self._legacy_cmd_cb, 10))

        # /robot_sleep is latched (TRANSIENT_LOCAL) — forward to Arduino as CMD_SLEEP
        sleep_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._subs.append(
            self.create_subscription(Bool, '/robot_sleep', self._sleep_cb, sleep_qos))

        if self.HAS_ULTRASONIC:
            self._ultrasonic_pub = self.create_lifecycle_publisher(
                Int16, self.ULTRASONIC_TOPIC, 10)
        if self.HAS_PIR:
            self._pir_pub = self.create_lifecycle_publisher(Bool, self.PIR_TOPIC, 10)
        if self.HAS_HALL:
            self._hall_pub = self.create_lifecycle_publisher(
                Int16MultiArray, self.HALL_TOPIC, 10)
        self._failsafe_pub = self.create_lifecycle_publisher(
            Bool, '~/failsafe',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._commanded_pub = self.create_lifecycle_publisher(JointState, '/joint_commanded', 20)
        self._owners_pub    = self.create_lifecycle_publisher(String, '~/joint_owners', 1)

        node_name = self.get_name()
        body_names = [n for n, _, _ in self.BODY_JOINTS]
        self.get_logger().info(
            f'{node_name} configured | port={self._serial_port} | '
            f'body={len(self.BODY_JOINTS)}, face={len(self.FACE_JOINTS)} joints')
        return TransitionCallbackReturn.SUCCESS

    def _lifecycle_pubs(self):
        return [p for p in (self._ultrasonic_pub, self._pir_pub, self._hall_pub,
                            self._failsafe_pub, self._commanded_pub, self._owners_pub)
                if p is not None]

    def on_activate(self, state):
        for p in self._lifecycle_pubs():
            p.on_activate(state)

        # Check that the port exists on the filesystem
        if not os.path.exists(self._serial_port):
            self.get_logger().error(
                f'Serial port not found: {self._serial_port} → FAILURE (retry)')
            for p in self._lifecycle_pubs():
                p.on_deactivate(state)
            return TransitionCallbackReturn.FAILURE

        self._connect_serial()
        if self._ser is None:
            for p in self._lifecycle_pubs():
                p.on_deactivate(state)
            return TransitionCallbackReturn.FAILURE

        self._tx_timer = self.create_timer(0.02, self._send_servos)
        self._owners_timer = self.create_timer(1.0, self._publish_owners)

        self._running   = True
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

        self.get_logger().info(
            f'{self.get_name()} active | port={self._serial_port}')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        # Send servos to rest position
        self._send_rest_positions()

        # Stop the RX thread
        self._running = False
        if self._rx_thread and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=2.0)
        self._rx_thread = None

        # Stop the TX timer
        if self._tx_timer:
            self.destroy_timer(self._tx_timer)
            self._tx_timer = None
        if self._owners_timer:
            self.destroy_timer(self._owners_timer)
            self._owners_timer = None

        # Close serial
        self._close_serial()

        for p in self._lifecycle_pubs():
            p.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self._close_serial()
        # Destroy what on_configure created — a re-configure must not duplicate them
        for s in self._subs:
            self.destroy_subscription(s)
        self._subs = []
        for p in self._lifecycle_pubs():
            self.destroy_lifecycle_publisher(p)
        self._ultrasonic_pub = self._pir_pub = self._hall_pub = None
        self._failsafe_pub = self._commanded_pub = self._owners_pub = None
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

    def _connect_serial(self, log_errors: bool = True) -> None:
        try:
            ser = serial.Serial(self._serial_port, self._baudrate, timeout=0.1)
            time.sleep(2.0)           # wait for Arduino bootloader
            # No reset_input_buffer — FrameParser discards bootloader garbage via SOF check
            self._parser = FrameParser()
            with self._lock:
                # A fresh Arduino boots awake with table speeds — resend both
                self._sleep_dirty  = True
                self._speeds_dirty = True
            self._ser = ser
            self.get_logger().info(f'Serial connected: {self._serial_port}')
        except (serial.SerialException, OSError) as e:
            if log_errors:
                self.get_logger().error(f'Serial open failed: {e}')
            self._ser = None

    def _link_lost(self, reason: str) -> None:
        """Serial error: close the handle; the RX thread reconnects."""
        if self._ser is None:
            return
        self.get_logger().error(
            f'Serial link lost ({reason}) — reconnecting in the background')
        self._close_serial()

    def _reconnect_step(self, attempt: int) -> int:
        """One reconnect attempt from the RX thread; returns the next attempt number."""
        delay = min(_RECONNECT_MIN_SEC * (2 ** attempt), _RECONNECT_MAX_SEC)
        deadline = time.monotonic() + delay
        while self._running and time.monotonic() < deadline:
            time.sleep(0.1)
        if not self._running:
            return attempt
        if not os.path.exists(self._serial_port):
            if attempt == 0:
                self.get_logger().warn(f'Serial port {self._serial_port} is gone — waiting')
            return attempt + 1
        self._connect_serial(log_errors=(attempt == 0))
        if not self._running:          # deactivated during the 2 s bootloader wait
            self._close_serial()
            return attempt
        if self._ser is not None:
            self.get_logger().info(f'Serial link restored after {attempt + 1} attempt(s)')
            return 0
        return attempt + 1

    # -----------------------------------------------------------------------
    # RX loop (sensor data from Arduino)
    # -----------------------------------------------------------------------

    def _rx_loop(self) -> None:
        attempt = 0
        try:
            while self._running:
                ser = self._ser
                if ser is None or not ser.is_open:
                    attempt = self._reconnect_step(attempt)
                    continue
                try:
                    chunk = ser.read(64)
                except (serial.SerialException, OSError, TypeError) as e:
                    # TypeError: pyserial on a handle closed by another thread
                    self._link_lost(f'read: {e}')
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

        elif cmd == CMD_HALL and self._hall_pub and len(data) >= 10:
            msg = Int16MultiArray()
            msg.data = [(data[i] << 8) | data[i + 1] for i in range(0, 10, 2)]
            self._hall_pub.publish(msg)

        elif cmd == CMD_STATUS and len(data) >= 1:
            active = bool(data[0])
            if active:
                self.get_logger().error(
                    'Arduino entered host-loss FAILSAFE (no servo frames for 1.5 s) '
                    '— returning to rest')
            else:
                self.get_logger().warn('Arduino left failsafe — servo frames resumed')
            if self._failsafe_pub:
                self._failsafe_pub.publish(Bool(data=active))

    # -----------------------------------------------------------------------
    # Command callbacks
    # -----------------------------------------------------------------------

    def _apply_joint(self, name: str, pos_rad: float, vel: float) -> None:
        """Applies a position to the body or face map. Called under self._lock."""
        # NaN/Inf would raise in round() and kill the executor → drop the joint
        if not math.isfinite(pos_rad):
            return
        if not math.isfinite(vel):
            vel = 0.0
        if name in self._body_map:
            i = self._body_map[name]
            _, center, _ = self.BODY_JOINTS[i]
            self._body_degs[i] = clamp_deg(rad_to_deg(pos_rad, center))
            step = (deg_per_sec_to_step(math.degrees(abs(vel))) if vel != 0.0
                    else self._default_body_steps[i])
            if step and step != self._body_speeds[i]:
                self._body_speeds[i] = step
                self._speeds_dirty = True
        elif name in self._face_map:
            i = self._face_map[name]
            _, center, _ = self.FACE_JOINTS[i]
            self._face_degs[i] = clamp_deg(rad_to_deg(pos_rad, center))
            step = (deg_per_sec_to_step(math.degrees(abs(vel))) if vel != 0.0
                    else self._default_face_steps[i])
            if step and step != self._face_speeds[i]:
                self._face_speeds[i] = step
                self._speeds_dirty = True

    def _sleep_cb(self, msg: Bool) -> None:
        with self._lock:
            if msg.data != self._sleeping:
                self._sleeping    = msg.data
                self._sleep_dirty = True

    def _arb_cmd_cb(self, msg: JointCommand) -> None:
        self._handle_cmd(msg.source or 'unknown', msg.priority, msg.lease_sec,
                         msg.release, msg.cmd)

    def _legacy_cmd_cb(self, msg: JointState) -> None:
        self._handle_cmd('legacy', JointCommand.PRIORITY_LEGACY, 0.0, False, msg)

    def _owns(self, name: str) -> bool:
        """This board drives the joint, or its mirrored eye."""
        return (name in self._body_map or name in self._face_map
                or self.EYE_SYNC.get(name, '') in self._body_map
                or self.EYE_SYNC.get(name, '') in self._face_map)

    def _handle_cmd(self, source: str, priority: int, lease_sec: float,
                    release: bool, js: JointState) -> None:
        accepted = JointState()
        with self._lock:
            vels = js.velocity
            for idx, (name, pos_rad) in enumerate(zip(js.name, js.position)):
                if not self._owns(name):
                    continue
                if release:
                    self._arbiter.release(name, source)
                    continue
                ok, owner = self._arbiter.claim(name, source, priority, lease_sec)
                if not ok:
                    self.get_logger().debug(
                        f'{name}: {source}(p{priority}) rejected — owned by '
                        f'{owner.source}(p{owner.priority})')
                    continue
                vel = vels[idx] if idx < len(vels) else 0.0
                self._apply_joint(name, pos_rad, vel)
                # Mirror the paired eye (eye_lr_L <-> eye_lr_R, eye_ud_L <-> eye_ud_R)
                mirror = self.EYE_SYNC.get(name)
                if mirror:
                    self._apply_joint(mirror, pos_rad, vel)
                if math.isfinite(pos_rad):
                    accepted.name.append(name)
                    accepted.position.append(pos_rad)
        if accepted.name and self._commanded_pub is not None:
            accepted.header.stamp = self.get_clock().now().to_msg()
            self._commanded_pub.publish(accepted)

    def _publish_owners(self) -> None:
        with self._lock:
            owners = self._arbiter.owners()
            rejected = self._arbiter.rejected
        self._owners_pub.publish(String(data=json.dumps(
            {'owners': owners, 'rejected_total': rejected})))

    # -----------------------------------------------------------------------
    # TX: send servo packet at 50 Hz
    # -----------------------------------------------------------------------

    def _send_servos(self) -> None:
        ser = self._ser
        if ser is None or not ser.is_open:
            return

        now = time.monotonic()
        with self._lock:
            if now - self._last_state_resend >= _STATE_RESEND_SEC:
                # Periodic resync (see module docstring) — idempotent in firmware
                self._last_state_resend = now
                self._sleep_dirty  = True
                self._speeds_dirty = True
            body = list(self._body_degs)
            face = list(self._face_degs)
            dirty = self._speeds_dirty
            if dirty:
                body_spd = list(self._body_speeds)
                face_spd = list(self._face_speeds)
                self._speeds_dirty = False
            sleep_dirty = self._sleep_dirty
            sleeping    = self._sleeping
            self._sleep_dirty = False

        try:
            # Send sleep state first (on change / reconnect / periodic resync)
            if sleep_dirty:
                ser.write(build_sleep(sleeping))
            # Send speed table (on change / reconnect / periodic resync)
            if dirty:
                spd_frame = build_set_speeds(body_spd + face_spd)
                ser.write(spd_frame)
            # Send servo positions
            frame = build_set_servos(body + face)
            ser.write(frame)
            ser.flush()
        except (serial.SerialException, OSError, TypeError) as e:
            self._link_lost(f'write: {e}')

    # destroy_node replaced by on_shutdown / on_deactivate (lifecycle)
