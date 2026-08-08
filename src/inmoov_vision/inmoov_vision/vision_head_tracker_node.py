#!/usr/bin/env python3
"""
vision_head_tracker_node.py
===========================
Управляет взглядом робота. Поддержка двух камер с автоматическим fallback.

Логика ведущего/ведомого:
  Ведущий — левый глаз (когда доступен):
    управляет головой (rothead + neck) и публикует eye_lr_L / eye_ud_L.
    EYE_SYNC в arduino_comm_node зеркалирует правый глаз автоматически.

  Fallback — правый глаз (если левый недоступен > STALE_SEC):
    берёт управление головой и публикует eye_lr_R / eye_ud_R.
    EYE_SYNC зеркалирует левый глаз.

  Публиковать оба набора одновременно нельзя — EYE_SYNC создаёт гонку
  (последний joint в пакете перезаписывает зеркало предыдущего).

Публикует:
  /joint_command  (JointState) — rothead, neck
  /face_command   (JointState) — eye_lr_L + eye_ud_L  | eye_lr_R + eye_ud_R

Подписки:
  /face/tracks/left    (String JSON)
  /face/tracks/right   (String JSON)
  /head_tracker/enable (Bool latched)
"""

import json
import math
import threading
import time

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Bool

_STALE_SEC    = 2.0  # bbox устаревает (с) — камера не давала лица
_FALLBACK_SEC = 5.0  # правая камера управляет головой только если левая отсутствует 5+с


def _bbox_area(bbox) -> float:
    if len(bbox) < 4:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _deg_to_rad(deg: float, center: float = 90.0) -> float:
    return (deg - center) * math.pi / 180.0


class VisionHeadTrackerNode(LifecycleNode):
    def __init__(self):
        super().__init__('vision_head_tracker_node')
        self._head_pub        = None
        self._face_pub        = None
        self._timer           = None
        self._lock            = threading.Lock()
        self._enabled         = False
        self._rothead_range   = (30.0, 130.0)
        self._neck_range      = (1.0,  100.0)
        self._eye_lr_range    = (80.0, 100.0)
        self._eye_ud_range    = (80.0, 110.0)
        self._left_bbox       = self._right_bbox = None
        self._last_left_t     = self._last_right_t = 0.0
        # Отдельный таймер: когда левая КАМЕРА последний раз прислала любое сообщение
        # (даже без лица). Используется для определения "камера жива", а не "лицо есть".
        self._last_left_msg_t: float = 0.0
        self._active_side     = 'left'
        self._head_stale      = 0
        self._at_rest         = False
        self._target_person_id = self._target_track_id = None

    def _dp(self, name, default=None):
        """Безопасный declare_parameter: игнорирует повторное объявление при re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('image_width',        640)
        self._dp('image_height',       480)
        self._dp('fov_h_deg',          60.0)
        self._dp('fov_v_deg',          45.0)
        self._dp('gain_head',           0.5)
        self._dp('gain_eye',            0.2)
        self._dp('dead_zone_px',        20)
        self._dp('head_dead_zone_px',   80)
        self._dp('eye_limit_deg',       8.0)
        self._dp('return_timeout_sec', 12.0)
        self._dp('track_hz',           15.0)
        self._dp('min_det_score',       0.50)
        self._dp('head_pan_dir',    1)
        self._dp('head_tilt_dir',  -1)
        self._dp('eye_pan_dir',    -1)
        self._dp('eye_tilt_dir',   -1)
        self._dp('max_step_deg',    2.0)
        self._dp('rest_rothead',   90.0)
        self._dp('rest_neck',      40.0)
        self._dp('rest_eye_lr',    90.0)
        self._dp('rest_eye_ud',   100.0)
        self._dp('bbox_ema_alpha',  0.4)
        self._dp('max_stale_ticks', 5)

        self._img_w          = self.get_parameter('image_width').value
        self._img_h          = self.get_parameter('image_height').value
        self._gain_head      = self.get_parameter('gain_head').value
        self._gain_eye       = self.get_parameter('gain_eye').value
        self._dead_zone      = self.get_parameter('dead_zone_px').value
        self._head_dead_zone = self.get_parameter('head_dead_zone_px').value
        self._ret_tmo        = self.get_parameter('return_timeout_sec').value
        self._hz             = self.get_parameter('track_hz').value
        self._min_det_score  = self.get_parameter('min_det_score').value
        self._head_pan_dir   = float(self.get_parameter('head_pan_dir').value)
        self._head_tilt_dir  = float(self.get_parameter('head_tilt_dir').value)
        self._eye_pan_dir    = float(self.get_parameter('eye_pan_dir').value)
        self._eye_tilt_dir   = float(self.get_parameter('eye_tilt_dir').value)
        self._max_step       = self.get_parameter('max_step_deg').value
        self._ema_alpha      = self.get_parameter('bbox_ema_alpha').value
        self._max_stale      = self.get_parameter('max_stale_ticks').value
        self._rothead        = self.get_parameter('rest_rothead').value
        self._neck           = self.get_parameter('rest_neck').value
        self._eye_lr         = self.get_parameter('rest_eye_lr').value
        self._eye_ud         = self.get_parameter('rest_eye_ud').value

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(String, '/face/tracks/left',    self._left_tracks_cb,  10)
        self.create_subscription(String, '/face/tracks/right',   self._right_tracks_cb, 10)
        self.create_subscription(Bool,   '/head_tracker/enable', self._enable_cb, latched_qos)
        self.create_subscription(String, '/face/identity',       self._identity_cb,     10)
        self.create_subscription(String, '/social_context',      self._social_context_cb, 10)
        self._head_pub = self.create_lifecycle_publisher(JointState, '/joint_command', 10)
        self._face_pub = self.create_lifecycle_publisher(JointState, '/face_command',  10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._head_pub.on_activate(state)
        self._face_pub.on_activate(state)
        self._timer = self.create_timer(1.0 / self._hz, self._tick)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        if self._timer:
            self.destroy_timer(self._timer)
            self._timer = None
        self._head_pub.on_deactivate(state)
        self._face_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

    # ── Enable ────────────────────────────────────────────────────────────

    def _enable_cb(self, msg: Bool):
        with self._lock:
            self._enabled = msg.data
            if msg.data:
                # Сбрасываем устаревшие timestamp чтобы не логировать "нет трека 348с"
                # сразу после включения (трекер был выключен, timestamp остались старые)
                now = time.time()
                self._last_left_t  = now
                self._last_right_t = now
                self._at_rest      = False
        if not msg.data:
            self._return_to_rest()
            self.get_logger().info('HeadTracker: выключен → покой')
        else:
            self.get_logger().info('HeadTracker: включён')

    # ── Целевой собеседник ────────────────────────────────────────────────

    def _social_context_cb(self, msg: String):
        """Получаем кто сейчас собеседник из social_context."""
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        state     = data.get('state', 'idle')
        person_id = data.get('person_id')
        with self._lock:
            if state == 'interacting' and person_id:
                if person_id != self._target_person_id:
                    self._target_person_id = person_id
                    self._target_track_id  = None  # ждём подтверждения от identity
            else:
                self._target_person_id = None
                self._target_track_id  = None

    def _identity_cb(self, msg: String):
        """Обновляем track_id целевого собеседника по результатам распознавания."""
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        if not data.get('locked') or not data.get('is_known'):
            return
        with self._lock:
            if data.get('person_id') == self._target_person_id:
                self._target_track_id = data.get('track_id')

    # ── Track callbacks ───────────────────────────────────────────────────

    def _left_tracks_cb(self, msg: String):
        if not self._enabled:
            return
        # Обновляем "камера жива" при ЛЮБОМ сообщении — даже если лица нет.
        # Это предотвращает ложный fallback на правую камеру когда левая активна
        # но временно не видит лицо (плохой угол, моргание).
        with self._lock:
            self._last_left_msg_t = time.time()
        bbox = self._extract_primary(msg)
        if bbox is None:
            return
        with self._lock:
            self._last_left_t = time.time()
            self._left_bbox   = self._ema(self._left_bbox, bbox)
            self._head_stale  = 0   # новая детекция → голова может двигаться
            self._at_rest     = False

    def _right_tracks_cb(self, msg: String):
        if not self._enabled:
            return
        bbox = self._extract_primary(msg)
        if bbox is None:
            return
        with self._lock:
            now = time.time()
            self._last_right_t = now
            self._right_bbox   = self._ema(self._right_bbox, bbox)
            # Если правый — ведущий (левый недоступен), сбрасываем stale
            if (now - self._last_left_t) >= _STALE_SEC:
                self._head_stale = 0
            self._at_rest = False

    def _extract_primary(self, msg: String) -> list | None:
        try:
            tracks = json.loads(msg.data).get('tracks', [])
        except Exception:
            return None
        tracks = [t for t in tracks if t.get('det_score', 1.0) >= self._min_det_score]
        if not tracks:
            return None

        with self._lock:
            target_id = self._target_track_id

        if target_id is not None:
            # В диалоге — следим только за track_id собеседника
            target = next((t for t in tracks if t.get('track_id') == target_id), None)
            if target is not None:
                return target.get('bbox')
            # Лицо собеседника не в кадре — не двигаемся к чужому
            return None

        # Не в диалоге — следуем за самым большим лицом
        return max(tracks, key=lambda t: _bbox_area(t.get('bbox', [0, 0, 0, 0]))).get('bbox')

    def _ema(self, current: list | None, new: list) -> list:
        if current is None:
            return list(new)
        a = self._ema_alpha
        return [a * n + (1.0 - a) * c for n, c in zip(new, current)]

    # ── Основной тик ─────────────────────────────────────────────────────

    def _tick(self):
        with self._lock:
            if not self._enabled:
                return
            now         = time.time()
            left_fresh   = (now - self._last_left_t)  < _STALE_SEC
            right_fresh  = (now - self._last_right_t) < _STALE_SEC
            # "Левая отсутствует" = не присылала НИКАКИХ сообщений (даже без лица)
            # дольше FALLBACK_SEC. Если левая активна но не видит лицо — это НЕ отсутствие,
            # это просто плохой угол. Используем _last_left_msg_t вместо _last_left_t.
            left_msg_ref = self._last_left_msg_t if self._last_left_msg_t > 0.0 \
                           else self._last_left_t
            left_absent  = (now - left_msg_ref) >= _FALLBACK_SEC
            left_bbox    = self._left_bbox  if left_fresh  else None
            right_bbox   = self._right_bbox if (right_fresh and left_absent) else None
            head_stale  = self._head_stale
            last_any    = max(self._last_left_t, self._last_right_t)

        if left_bbox is None and right_bbox is None:
            elapsed = time.time() - last_any
            if elapsed > self._ret_tmo:
                if not self._at_rest:
                    self.get_logger().info(
                        f'HeadTracker: нет трека {elapsed:.0f}с → покой '
                        f'(rothead={self._rothead:.1f}°, neck={self._neck:.1f}°)')
                    self._return_to_rest()   # сбрасывает углы, обнуляет bboxes, публикует
                else:
                    # Уже в покое — просто держим позицию (без лога и без сброса состояния)
                    self._publish_head()
                    self._publish_eyes('left')
            return

        # Ведущий — левый если доступен
        if left_bbox is not None:
            lead_bbox  = left_bbox
            new_side   = 'left'
        else:
            lead_bbox  = right_bbox
            new_side   = 'right'

        # Смена стороны — лог
        if new_side != self._active_side:
            self.get_logger().info(
                f'HeadTracker: {self._active_side} → {new_side} (EYE_SYNC активен) '
                f'rothead={self._rothead:.1f}° neck={self._neck:.1f}°')
            self._active_side = new_side

        norm_x, norm_y = self._bbox_to_norm(lead_bbox)

        # ── Голова (с защитой от осцилляции) ─────────────────────────────
        if head_stale < self._max_stale:
            self._step_head(norm_x, norm_y)
            with self._lock:
                self._head_stale += 1

        # ── Глаза: один набор joint names, EYE_SYNC зеркалирует второй ──
        self._set_eye(norm_x, norm_y)

        self._publish_head()
        self._publish_eyes(self._active_side)

    # ── P-регулятор ───────────────────────────────────────────────────────

    def _bbox_to_norm(self, bbox: list) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        norm_x = ((x1 + x2) / 2.0 - self._img_w / 2.0) / (self._img_w / 2.0)
        norm_y = ((y1 + y2) / 2.0 - self._img_h / 2.0) / (self._img_h / 2.0)
        return norm_x, norm_y

    def _step_head(self, norm_x: float, norm_y: float):
        head_dz_x = self._head_dead_zone / (self._img_w / 2.0)
        head_dz_y = self._head_dead_zone / (self._img_h / 2.0)
        step_h, step_v = 0.0, 0.0
        if abs(norm_x) >= head_dz_x:
            step_h = _clamp(
                self._head_pan_dir * norm_x * self._max_step * self._gain_head,
                -self._max_step, self._max_step)
        if abs(norm_y) >= head_dz_y:
            step_v = _clamp(
                self._head_tilt_dir * norm_y * self._max_step * self._gain_head,
                -self._max_step, self._max_step)
        self._rothead = _clamp(self._rothead + step_h, *self._rothead_range)
        self._neck    = _clamp(self._neck    + step_v, *self._neck_range)

    def _set_eye(self, norm_x: float, norm_y: float):
        eye_dz_x = self._dead_zone / (self._img_w / 2.0)
        eye_dz_y = self._dead_zone / (self._img_h / 2.0)
        if abs(norm_x) < eye_dz_x and abs(norm_y) < eye_dz_y:
            return
        half_h = (self._eye_lr_range[1] - self._eye_lr_range[0]) / 2.0
        half_v = (self._eye_ud_range[1] - self._eye_ud_range[0]) / 2.0
        ctr_lr = (self._eye_lr_range[0] + self._eye_lr_range[1]) / 2.0
        ctr_ud = (self._eye_ud_range[0] + self._eye_ud_range[1]) / 2.0
        self._eye_lr = _clamp(
            ctr_lr + self._eye_pan_dir  * norm_x * half_h * self._gain_eye,
            *self._eye_lr_range)
        self._eye_ud = _clamp(
            ctr_ud + self._eye_tilt_dir * norm_y * half_v * self._gain_eye,
            *self._eye_ud_range)

    # ── Покой ─────────────────────────────────────────────────────────────

    def _return_to_rest(self):
        self._rothead = self.get_parameter('rest_rothead').value
        self._neck    = self.get_parameter('rest_neck').value
        self._eye_lr  = self.get_parameter('rest_eye_lr').value
        self._eye_ud  = self.get_parameter('rest_eye_ud').value
        with self._lock:
            self._left_bbox        = None
            self._right_bbox       = None
            self._head_stale       = 0
            self._target_track_id  = None
            self._target_person_id = None
            self._at_rest          = True
        self._publish_head()
        # В покое публикуем левый — EYE_SYNC зеркалирует правый
        self._publish_eyes('left')

    # ── Публикаторы ───────────────────────────────────────────────────────

    def _publish_head(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = ['rothead', 'neck']
        msg.position = [
            _deg_to_rad(self._rothead, center=90.0),
            _deg_to_rad(self._neck,    center=90.0),
        ]
        self._head_pub.publish(msg)

    def _publish_eyes(self, side: str):
        """Публикует только один глаз — EYE_SYNC в arduino_comm_node зеркалирует второй."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        if side == 'left':
            msg.name = ['eye_lr_L', 'eye_ud_L']
        else:
            msg.name = ['eye_lr_R', 'eye_ud_R']
        msg.position = [
            _deg_to_rad(self._eye_lr, center=90.0),
            _deg_to_rad(self._eye_ud, center=90.0),
        ]
        self._face_pub.publish(msg)




def main():
    rclpy.init()
    node = VisionHeadTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
