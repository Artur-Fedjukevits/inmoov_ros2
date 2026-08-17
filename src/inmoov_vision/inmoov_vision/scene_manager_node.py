#!/usr/bin/env python3
"""
scene_manager_node.py — агрегатор сцены: объекты + люди вокруг робота.

Слушает сырые детекции OAK-D (oak_node) и превращает поток кадров в
устойчивую сводку: сколько людей в кадре, какие предметы рядом, на каком
расстоянии и с какой стороны. Сводка передаётся дальше в inmoov_cognition
(llm_node — system prompt, behavior_manager_node — Blackboard).

Топики:
  /objects/detections  (String JSON)  ← oak_node
  /robot_sleep          (Bool, latched) ← identity_manager_node
  /scene/objects        (String JSON)  → llm_node, behavior_manager_node

Параметры:
  min_confidence        — мин. уверенность детекции (default 0.5)
  max_distance_m        — макс. дальность учёта объекта (default 4.0м)
  object_ttl_sec        — сколько метка остаётся в сводке без подтверждений (default 8.0с;
                            YOLO иногда пропускает кадр даже для неподвижного объекта —
                            короткий ttl даёт заметное мигание в сводке)
  smoothing_window_sec  — окно сглаживания счёта по кадрам (default 1.0с)
  publish_rate_hz        — частота публикации сводки (default 1.0 Гц)
  top_k_objects          — сколько меток включать в сводку (default 8)
  location_name          — статичное имя текущего места робота (default '');
                            читается заново на каждой публикации, можно
                            обновить на лету: ros2 param set
                            /scene_manager_node location_name "кухня"
"""

import json
import threading
import time
from collections import deque

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, String


class SceneManagerNode(LifecycleNode):
    def __init__(self):
        super().__init__('scene_manager_node')
        self._lock     = threading.Lock()
        self._pub      = None
        self._timer    = None
        self._sleeping = False
        # label -> {'last_seen': ts, 'window': deque[(ts, count)],
        #           'distance_m': float, 'direction': str}
        self._labels: dict[str, dict] = {}

    def _dp(self, name, default=None):
        """Безопасный declare_parameter: игнорирует повторное объявление при re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('min_confidence',       0.5)
        self._dp('max_distance_m',       4.0)
        self._dp('object_ttl_sec',       8.0)
        self._dp('smoothing_window_sec', 1.0)
        self._dp('publish_rate_hz',      1.0)
        self._dp('top_k_objects',        8)
        self._dp('location_name',        '')

        self._min_conf   = self.get_parameter('min_confidence').value
        self._max_z_mm   = self.get_parameter('max_distance_m').value * 1000.0
        self._ttl_sec     = self.get_parameter('object_ttl_sec').value
        self._window_sec  = self.get_parameter('smoothing_window_sec').value
        self._rate_hz      = self.get_parameter('publish_rate_hz').value
        self._top_k         = self.get_parameter('top_k_objects').value

        _latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Bool,   '/robot_sleep',       self._robot_sleep_cb, _latched)
        self.create_subscription(String, '/objects/detections', self._detections_cb, 10)

        self._pub = self.create_lifecycle_publisher(String, '/scene/objects', 10)
        self.get_logger().info(
            f'SceneManager configured. conf>={self._min_conf}, '
            f'dist<={self._max_z_mm/1000:.1f}м, ttl={self._ttl_sec:.1f}с')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._pub.on_activate(state)
        self._timer = self.create_timer(1.0 / self._rate_hz, self._publish_scene)
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
                self._labels.clear()
        if msg.data and not was_sleeping:
            self.get_logger().info('Спящий режим: scene_manager приостановлен')
        elif not msg.data and was_sleeping:
            self.get_logger().info('Пробуждение: scene_manager возобновлён')

    def _detections_cb(self, msg: String):
        with self._lock:
            if self._sleeping:
                return
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        objects = [
            o for o in data.get('objects', [])
            if o.get('confidence', 0.0) >= self._min_conf
            and 0 < o.get('z_mm', 0.0) <= self._max_z_mm
        ]

        # Считаем сколько экземпляров каждой метки в ЭТОМ кадре, плюс
        # ближайший экземпляр каждой метки (для distance/direction).
        frame_counts: dict[str, int] = {}
        nearest_per_label: dict[str, dict] = {}
        for o in objects:
            label = o.get('label', '?')
            frame_counts[label] = frame_counts.get(label, 0) + 1
            if label not in nearest_per_label or o['z_mm'] < nearest_per_label[label]['z_mm']:
                nearest_per_label[label] = o

        now = time.time()
        with self._lock:
            for label, count in frame_counts.items():
                entry = self._labels.setdefault(label, {
                    'last_seen': 0.0, 'window': deque(), 'last_count': 0,
                    'distance_m': 0.0, 'direction': 'center',
                })
                entry['window'].append((now, count))
                # чистим окно сглаживания сразу, чтобы не расти бесконечно
                while entry['window'] and now - entry['window'][0][0] > self._window_sec:
                    entry['window'].popleft()
                entry['last_seen']  = now
                entry['last_count'] = count
                nearest = nearest_per_label[label]
                entry['distance_m'] = round(nearest['z_mm'] / 1000.0, 2)
                bbox = nearest.get('bbox', [0.0, 0.0, 1.0, 1.0])
                cx = (bbox[0] + bbox[2]) / 2.0
                if cx < 0.35:
                    entry['direction'] = 'left'
                elif cx > 0.65:
                    entry['direction'] = 'right'
                else:
                    entry['direction'] = 'center'

    # ── Публикация ────────────────────────────────────────────────────────────

    def _publish_scene(self):
        with self._lock:
            if self._sleeping:
                return
            now = time.time()
            # Выбрасываем метки, не подтверждавшиеся дольше object_ttl_sec
            stale = [l for l, e in self._labels.items() if now - e['last_seen'] > self._ttl_sec]
            for l in stale:
                del self._labels[l]

            objects = []
            for label, entry in self._labels.items():
                # Максимум по свежим кадрам гасит дрожание счёта, пока детекции
                # идут потоком. Если свежих кадров в окне нет (детектор молчал
                # дольше smoothing_window_sec, но метка ещё не устарела по ttl) —
                # держим последний известный счёт вместо обнуления.
                recent = [c for t, c in entry['window'] if now - t <= self._window_sec]
                count = max(recent) if recent else entry['last_count']
                if count <= 0:
                    continue
                objects.append({
                    'label':       label,
                    'count':       count,
                    'distance_m':  entry['distance_m'],
                    'direction':   entry['direction'],
                })

        objects.sort(key=lambda o: o['distance_m'])
        objects = objects[:self._top_k]
        person_count = next((o['count'] for o in objects if o['label'] == 'person'), 0)

        location = self.get_parameter('location_name').value

        payload = {
            'location':     location,
            'person_count': person_count,
            'objects':      objects,
            'updated_at':   now,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = SceneManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
