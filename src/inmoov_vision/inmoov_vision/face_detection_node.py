#!/usr/bin/env python3
"""
face_detection_node.py
======================
Запускает insightface на кадрах с указанного глаза.
Параметр camera_side ('left' | 'right') определяет:
  - какую камеру читать:   /camera/eye_{side}/compressed
  - куда публиковать:      /face/detections/{side}

insightface запускается в фоновом потоке.
Частота анализа регулируется параметром detection_hz (default 1.5 Hz).

Fallback-по-требованию (параметр fallback_for):
  Если задан непустым (топик другого глаза, напр. '/face/detections/left'),
  нода не гоняет insightface постоянно — она слушает этот топик как heartbeat
  primary-камеры и запускает собственную детекцию только когда primary не
  публиковал дольше primary_timeout_sec (default 1.5с). Экономит CPU: правый
  глаз (fallback) не молотит insightface параллельно с левым (primary) 24/7,
  а включается только когда левый реально пропал (камера отвалилась/процесс
  упал) — то есть настоящий fallback, а не постоянное дублирование.

Подписки:
  /camera/eye_{camera_side}/compressed  (sensor_msgs/CompressedImage)

Публикует:
  /face/detections/{camera_side}  (std_msgs/String — JSON)

Формат JSON:
  {
    "stamp": 1234567890.0,
    "faces": [
      {
        "bbox":      [x1, y1, x2, y2],
        "det_score": 0.98,
        "embedding": [0.01, -0.03, ...],   // 512-d normed (buffalo_l only)
        "kps":       [[x,y], ...]          // 5 точек
      }
    ]
  }
"""

import json
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from insightface.app import FaceAnalysis


class FaceDetectionNode(LifecycleNode):
    def __init__(self):
        super().__init__('face_detection_node')
        self._pub          = None
        self._timers       = []
        self._frame_lock   = threading.Lock()
        self._latest_frame = None
        self._latest_stamp = 0.0
        self._busy         = False
        self._enabled      = False
        self._last_frame_t = self._last_detect_t = self._no_face_since = 0.0
        self._fallback_for       = ''
        self._primary_timeout    = 1.5
        self._last_primary_t     = 0.0
        self._fallback_active    = False
        # Диагностика "подвисшей" камеры (см. _trigger_detection) — кадр
        # приходит по расписанию, но его stamp не меняется.
        self._last_processed_stamp   = -1.0
        self._stale_frame_streak     = 0
        self._STALE_FRAME_WARN_STREAK = 3

    def _dp(self, name, default=None):
        """Безопасный declare_parameter: игнорирует повторное объявление при re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('camera_side',  'left')
        self._dp('detection_hz', 1.5)
        self._dp('det_size',     640)
        self._dp('det_thresh',   0.4)
        self._dp('model_name',   'buffalo_l')
        self._dp('intra_op_threads', 2)
        self._dp('inter_op_threads', 1)
        self._dp('fallback_for',        '')
        self._dp('primary_timeout_sec', 1.5)

        self._side   = self.get_parameter('camera_side').value
        self._det_hz = self.get_parameter('detection_hz').value
        det_size     = self.get_parameter('det_size').value
        det_thresh   = self.get_parameter('det_thresh').value
        model_name   = self.get_parameter('model_name').value
        intra_op     = self.get_parameter('intra_op_threads').value
        inter_op     = self.get_parameter('inter_op_threads').value
        self._fallback_for    = self.get_parameter('fallback_for').value
        self._primary_timeout = self.get_parameter('primary_timeout_sec').value

        # По умолчанию ORT разворачивает пул потоков под все логические ядра
        # для КАЖДОЙ из ~5 моделей buffalo_l (intra_op_num_threads=0 = auto).
        # При двух процессах (left+right) на 16 логических ядрах это даёт
        # конкуренцию за потоки и ~350-370% CPU на процесс. Ограничиваем пул
        # явно — insightface.FaceAnalysis(**kwargs) прокидывает sess_options
        # до onnxruntime.InferenceSession без изменений.
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = intra_op
        sess_options.inter_op_num_threads = inter_op

        self.get_logger().info(f'Загрузка insightface ({model_name}, side={self._side})...')
        self._app = FaceAnalysis(
            name=model_name, providers=['CPUExecutionProvider'], sess_options=sess_options)
        self._app.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=det_thresh)
        self.get_logger().info('insightface загружен')

        from std_msgs.msg import Bool as _Bool
        cam_topic = f'/camera/eye_{self._side}/compressed'
        det_topic = f'/face/detections/{self._side}'
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(CompressedImage, cam_topic, self._frame_callback, 5)
        self.create_subscription(_Bool, '/face_detection/enable', self._enable_cb, latched_qos)
        if self._fallback_for:
            self.create_subscription(
                String, self._fallback_for, self._primary_heartbeat_cb, 5)
        self._pub = self.create_lifecycle_publisher(String, det_topic, 10)
        fb_note = f', fallback_for={self._fallback_for} (timeout={self._primary_timeout}с)' \
            if self._fallback_for else ''
        self.get_logger().info(
            f'FaceDetection configured (side={self._side}, hz={self._det_hz}, '
            f'det_size={det_size}{fb_note})')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._pub.on_activate(state)
        # Грейс-период: считаем primary живым с момента активации, а не с
        # эпохи (иначе _last_primary_t=0.0 → time.time()-0 огромно >
        # primary_timeout → ложный "primary молчит" на самом первом тике,
        # ещё до того как primary вообще успел прислать первый heartbeat).
        self._last_primary_t = time.time()
        self._timers.append(self.create_timer(1.0 / self._det_hz, self._trigger_detection))
        self._timers.append(self.create_timer(5.0, self._watchdog))
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        for t in self._timers:
            self.destroy_timer(t)
        self._timers.clear()
        self._pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

    def _frame_callback(self, msg: CompressedImage):
        """Сохраняем последний кадр без обработки — только буфер."""
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_stamp = stamp
        self._last_frame_t = time.time()

    def _enable_cb(self, msg) -> None:
        self._enabled = msg.data
        self._no_face_since = 0.0
        if msg.data:
            # Грейс-период на КАЖДОЕ включение (не только on_activate — сам
            # lifecycle activate происходит раз при старте стека, а enable
            # дальше дёргается toggle'ом через PIRScan/пробуждение). Без
            # сброса здесь _last_primary_t остаётся протухшим с прошлого
            # раза → primary_alive сразу ложно False на первом тике.
            self._last_primary_t = time.time()
            self._fallback_active = False
            self._last_processed_stamp = -1.0
            self._stale_frame_streak   = 0
        role = f', fallback-резерв за {self._fallback_for}' if self._fallback_for else ', primary'
        self.get_logger().info(
            f'FaceDetection: {"включена" if msg.data else "выключена"}{role}')

    def _primary_heartbeat_cb(self, msg) -> None:
        """Приход ЛЮБОГО сообщения от primary-камеры = она жива, не важно нашла ли лицо."""
        self._last_primary_t = time.time()

    def _trigger_detection(self):
        if not self._enabled:
            return
        if self._busy:
            return
        if self._fallback_for:
            primary_alive = (time.time() - self._last_primary_t) < self._primary_timeout
            if primary_alive:
                if self._fallback_active:
                    self.get_logger().info(
                        f'Primary ({self._fallback_for}) восстановилась — fallback ({self._side}) в резерв')
                self._fallback_active = False
                return
            if not self._fallback_active:
                self._fallback_active = True
                self.get_logger().warn(
                    f'Primary ({self._fallback_for}) молчит >{self._primary_timeout}с — '
                    f'активирую fallback-детекцию на {self._side}')
        with self._frame_lock:
            if self._latest_frame is None:
                return
            frame = self._latest_frame.copy()
            stamp = self._latest_stamp

        # Живой баг 2026-08-31/09-01: подозрение, что USB-камера в глазу может
        # физически "подвиснуть" (сенсор перестал обновляться), но кадры с
        # последним удачным снимком продолжают приходить по /camera/eye_*
        # — _frame_callback добросовестно обновляет _last_frame_t на КАЖДОЕ
        # сообщение, поэтому обычный "камера молчит" watchdog это не ловит.
        # Раньше это приводило к тому, что head_tracker получал "свежие" (по
        # времени сообщения) детекции с НЕИЗМЕНЫМ bbox/offset — голову
        # уводило в сторону без остановки, потому что реальной новой
        # картинки не было. Сравниваем stamp КАДРА (не сообщения) с прошлым
        # запуском детекции — если он не продвинулся, это тот же кадр.
        if stamp == self._last_processed_stamp:
            self._stale_frame_streak += 1
            if self._stale_frame_streak == self._STALE_FRAME_WARN_STREAK:
                self.get_logger().warn(
                    f'FaceDetection ({self._side}): кадр НЕ обновляется '
                    f'{self._stale_frame_streak} детекций подряд (stamp='
                    f'{stamp:.3f}) — камера могла "подвиснуть" (сообщения '
                    f'приходят, но картинка та же)')
        else:
            self._stale_frame_streak = 0
        self._last_processed_stamp = stamp

        self._last_detect_t = time.time()
        self._busy = True
        threading.Thread(
            target=self._detect, args=(frame, stamp), daemon=True).start()

    def _detect(self, frame: np.ndarray, stamp: float):
        try:
            faces = self._app.get(frame)
            result = {
                'stamp': stamp,
                'faces': [],
            }
            for face in faces:
                entry = {
                    'bbox':      [int(v) for v in face.bbox.tolist()],
                    'det_score': float(face.det_score),
                    'kps':       [[int(p[0]), int(p[1])] for p in face.kps.tolist()]
                    if face.kps is not None else [],
                }
                if face.normed_embedding is not None:
                    entry['embedding'] = face.normed_embedding.tolist()
                result['faces'].append(entry)

            if result['faces']:
                self._no_face_since = 0.0
                self.get_logger().debug(
                    f'Обнаружено лиц: {len(result["faces"])}')
            else:
                if self._no_face_since == 0.0:
                    self._no_face_since = time.time()

            msg = String()
            msg.data = json.dumps(result)
            self._pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f'Ошибка детекции: {e}')
        finally:
            self._busy = False


    def _watchdog(self):
        if not self._enabled:
            return
        now = time.time()
        # Кадры не приходят?
        if self._last_frame_t > 0.0 and (now - self._last_frame_t) > 3.0:
            self.get_logger().warn(
                f'Камера молчит {now - self._last_frame_t:.1f}с — нет кадров с /camera/eye_{self._side}/compressed')
        elif self._last_frame_t == 0.0:
            self.get_logger().warn('Кадры с камеры ещё не получены')
        # Детекция не запускается? (fallback-нода в резерве не детектит — это норма, не варним)
        in_standby = bool(self._fallback_for) and not self._fallback_active
        if not in_standby and self._last_detect_t > 0.0 and (now - self._last_detect_t) > 3.0:
            self.get_logger().warn(
                f'Детекция не запускается {now - self._last_detect_t:.1f}с (busy={self._busy})')
        # Лицо давно не найдено
        if self._no_face_since > 0.0 and (now - self._no_face_since) > 10.0:
            self.get_logger().warn(
                f'Лицо не обнаружено уже {now - self._no_face_since:.0f}с')
            self._no_face_since = now  # сбрасываем чтобы не спамить каждые 5с




def main():
    rclpy.init()
    node = FaceDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
