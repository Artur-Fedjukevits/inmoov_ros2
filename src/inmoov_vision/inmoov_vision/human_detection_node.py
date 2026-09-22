#!/usr/bin/env python3
"""
human_detection_node.py
=======================
Human presence: YOLO (object type) + ultrasonic (precise distance).

YOLO confirms that the object is a person.
Ultrasonic validates the distance (more accurate than stereo depth).
Final distance = ultrasonic if available, otherwise z from YOLO.

Topics:
  /objects/detections        (String JSON)  <- oak_node
  /ultrasonic_left_distance  (Int16, cm)    <- arduino_left_node
  /ultrasonic_right_distance (Int16, cm)    <- arduino_right_node
  /human_detected            (Bool)         -> identity_manager / BT
  /human_angle_deg           (Float32)      -> BT (SoundScanBehaviour — aims
                              the head at the nearest person BEFORE
                              face_detection/head_tracker have had a chance to
                              see them. atan2(x_mm, z_mm) from the OAK-D,
                              + = person is to the right (DepthAI convention:
                              X is positive to the right of the camera) — NOT
                              validated physically, like the other sign
                              conventions in the project; check by hand on
                              first use)

Parameters:
  max_distance_m      — max range (default 4.0 m)
  min_confidence      — min YOLO confidence (default 0.45)
  lost_timeout_sec    — time without a detection before publishing False (default 4.0 s)
  publish_rate_hz     — publish rate (default 5.0 Hz)
  use_ultrasonic      — use ultrasonic for distance (default True)
  ultrasonic_stale_s  — ultrasonic staleness limit (default 1.0 s)

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import json
import math
import threading
import time

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, Int16, String


class HumanDetectionNode(LifecycleNode):
    def __init__(self):
        super().__init__('human_detection_node')
        self._lock         = threading.Lock()
        self._pub          = None
        self._timer        = None
        self._last_seen    = 0.0
        self._yolo_z_m     = 0.0
        self._angle_deg    = 0.0
        self._angle_pub    = None
        self._was_detected = False
        self._sleeping     = False
        self._us_left_m    = self._us_right_m  = 0.0
        self._us_left_ts   = self._us_right_ts = 0.0

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores re-declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('max_distance_m',     4.0)
        self._dp('min_confidence',     0.45)
        self._dp('lost_timeout_sec',   4.0)
        self._dp('publish_rate_hz',    5.0)
        self._dp('use_ultrasonic',     True)
        self._dp('ultrasonic_stale_s', 1.0)

        self._max_z_mm     = self.get_parameter('max_distance_m').value * 1000.0
        self._min_conf     = self.get_parameter('min_confidence').value
        self._lost_timeout = self.get_parameter('lost_timeout_sec').value
        self._use_us       = self.get_parameter('use_ultrasonic').value
        self._us_stale     = self.get_parameter('ultrasonic_stale_s').value
        self._rate_hz      = self.get_parameter('publish_rate_hz').value

        _latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Bool,   '/robot_sleep',       self._robot_sleep_cb, _latched)
        self.create_subscription(String, '/objects/detections', self._detections_cb, 10)
        if self._use_us:
            self.create_subscription(Int16, '/ultrasonic_left_distance',  self._us_left_cb,  10)
            self.create_subscription(Int16, '/ultrasonic_right_distance', self._us_right_cb, 10)

        self._pub       = self.create_lifecycle_publisher(Bool, '/human_detected', 10)
        self._angle_pub = self.create_lifecycle_publisher(Float32, '/human_angle_deg', 10)
        self.get_logger().info(
            f'HumanDetection configured. '
            f'Range: {self._max_z_mm/1000:.1f} m, confidence: {self._min_conf}')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._pub.on_activate(state)
        self._angle_pub.on_activate(state)
        self._timer = self.create_timer(1.0 / self._rate_hz, self._publish_state)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        if self._timer:
            self.destroy_timer(self._timer)
            self._timer = None
        self._pub.on_deactivate(state)
        self._angle_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _robot_sleep_cb(self, msg: Bool):
        with self._lock:
            was_sleeping = self._sleeping
            self._sleeping = msg.data
            if msg.data and not was_sleeping:
                # Reset state so that we start from a clean slate on wake-up
                self._last_seen    = 0.0
                self._yolo_z_m     = 0.0
                self._us_left_ts   = 0.0
                self._us_right_ts  = 0.0
        if msg.data and not was_sleeping:
            # Publish False once when entering sleep
            out = Bool()
            out.data = False
            self._pub.publish(out)
            self._was_detected = False
            self.get_logger().info('Sleep mode: human_detection paused')
        elif not msg.data and was_sleeping:
            self.get_logger().info('Wake-up: human_detection resumed')

    def _us_left_cb(self, msg: Int16):
        if msg.data > 0:
            with self._lock:
                if self._sleeping:
                    return
                self._us_left_m  = msg.data / 100.0  # cm -> m
                self._us_left_ts = time.time()

    def _us_right_cb(self, msg: Int16):
        if msg.data > 0:
            with self._lock:
                if self._sleeping:
                    return
                self._us_right_m  = msg.data / 100.0
                self._us_right_ts = time.time()

    def _detections_cb(self, msg: String):
        with self._lock:
            if self._sleeping:
                return
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        persons = [
            o for o in data.get('objects', [])
            if o.get('label') == 'person'
            and o.get('confidence', 0.0) >= self._min_conf
            and 0 < o.get('z_mm', 0.0) <= self._max_z_mm
        ]

        if persons:
            closest = min(persons, key=lambda o: o['z_mm'])
            with self._lock:
                self._last_seen = time.time()
                self._yolo_z_m  = closest['z_mm'] / 1000.0
                x_mm = closest.get('x_mm', 0.0)
                z_mm = closest.get('z_mm', 0.0)
                if z_mm > 0:
                    self._angle_deg = math.degrees(math.atan2(x_mm, z_mm))

    # ── Publishing ────────────────────────────────────────────────────────────

    def _best_distance_m(self) -> float:
        """Best distance estimate: ultrasonic if fresh, otherwise YOLO."""
        if not self._use_us:
            return self._yolo_z_m

        now = time.time()
        candidates = []
        if now - self._us_left_ts  < self._us_stale and self._us_left_m  > 0:
            candidates.append(self._us_left_m)
        if now - self._us_right_ts < self._us_stale and self._us_right_m > 0:
            candidates.append(self._us_right_m)

        if candidates:
            return min(candidates)   # the nearer of the two sensors
        return self._yolo_z_m

    def _publish_state(self):
        with self._lock:
            if self._sleeping:
                return
            detected  = (time.time() - self._last_seen) < self._lost_timeout
            dist_m    = self._best_distance_m()
            angle_deg = self._angle_deg

        msg = Bool()
        msg.data = detected
        self._pub.publish(msg)

        if detected:
            angle_msg = Float32()
            angle_msg.data = float(angle_deg)
            self._angle_pub.publish(angle_msg)

        if detected != self._was_detected:
            if detected:
                src = 'US' if self._use_us else 'YOLO'
                self.get_logger().info(
                    f'Person detected (body) at {dist_m:.1f} m [{src}]')
            else:
                self.get_logger().info('Person not detected (body detection)')
            self._was_detected = detected



def main():
    rclpy.init()
    node = HumanDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
