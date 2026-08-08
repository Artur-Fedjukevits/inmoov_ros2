#!/usr/bin/env python3
"""
human_detection_node.py
=======================
Присутствие человека: YOLO (тип объекта) + ультразвук (точная дистанция).

YOLO подтверждает что объект — человек.
Ультразвук валидирует расстояние (точнее стерео-глубины).
Итоговая дистанция = ultrasonic если доступен, иначе z из YOLO.

Топики:
  /objects/detections        (String JSON)  ← oak_node
  /ultrasonic_left_distance  (Int16, см)    ← arduino_left_node
  /ultrasonic_right_distance (Int16, см)    ← arduino_right_node
  /human_detected            (Bool)         → identity_manager / BT

Параметры:
  max_distance_m      — макс. дальность (default 4.0м)
  min_confidence      — мин. уверенность YOLO (default 0.45)
  lost_timeout_sec    — время без детекции до False (default 4.0с)
  publish_rate_hz     — частота публикации (default 5.0 Гц)
  use_ultrasonic      — использовать ультразвук для дистанции (default True)
  ultrasonic_stale_s  — устаревание ультразвука (default 1.0с)
"""

import json
import threading
import time

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, Int16, String


class HumanDetectionNode(LifecycleNode):
    def __init__(self):
        super().__init__('human_detection_node')
        self._lock         = threading.Lock()
        self._pub          = None
        self._timer        = None
        self._last_seen    = 0.0
        self._yolo_z_m     = 0.0
        self._was_detected = False
        self._sleeping     = False
        self._us_left_m    = self._us_right_m  = 0.0
        self._us_left_ts   = self._us_right_ts = 0.0

    def _dp(self, name, default=None):
        """Безопасный declare_parameter: игнорирует повторное объявление при re-configure."""
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

        self._pub = self.create_lifecycle_publisher(Bool, '/human_detected', 10)
        self.get_logger().info(
            f'HumanDetection configured. '
            f'Дальность: {self._max_z_mm/1000:.1f}м, confidence: {self._min_conf}')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._pub.on_activate(state)
        self._timer = self.create_timer(1.0 / self._rate_hz, self._publish_state)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        if self._timer:
            self.destroy_timer(self._timer)
            self._timer = None
        self._pub.on_deactivate(state)
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
                # Сбрасываем состояние чтобы при пробуждении начать с чистого листа
                self._last_seen    = 0.0
                self._yolo_z_m     = 0.0
                self._us_left_ts   = 0.0
                self._us_right_ts  = 0.0
        if msg.data and not was_sleeping:
            # Одна публикация False при входе в сон
            out = Bool()
            out.data = False
            self._pub.publish(out)
            self._was_detected = False
            self.get_logger().info('Спящий режим: human_detection приостановлен')
        elif not msg.data and was_sleeping:
            self.get_logger().info('Пробуждение: human_detection возобновлён')

    def _us_left_cb(self, msg: Int16):
        if msg.data > 0:
            with self._lock:
                if self._sleeping:
                    return
                self._us_left_m  = msg.data / 100.0  # см → м
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

    # ── Публикация ────────────────────────────────────────────────────────────

    def _best_distance_m(self) -> float:
        """Лучшая оценка дистанции: ультразвук если свежий, иначе YOLO."""
        if not self._use_us:
            return self._yolo_z_m

        now = time.time()
        candidates = []
        if now - self._us_left_ts  < self._us_stale and self._us_left_m  > 0:
            candidates.append(self._us_left_m)
        if now - self._us_right_ts < self._us_stale and self._us_right_m > 0:
            candidates.append(self._us_right_m)

        if candidates:
            return min(candidates)   # ближайшее из двух датчиков
        return self._yolo_z_m

    def _publish_state(self):
        with self._lock:
            if self._sleeping:
                return
            detected = (time.time() - self._last_seen) < self._lost_timeout
            dist_m   = self._best_distance_m()

        msg = Bool()
        msg.data = detected
        self._pub.publish(msg)

        if detected != self._was_detected:
            if detected:
                src = 'US' if self._use_us else 'YOLO'
                self.get_logger().info(
                    f'Человек обнаружен (body) на расстоянии {dist_m:.1f}м [{src}]')
            else:
                self.get_logger().info('Человек не обнаружен (body detection)')
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
