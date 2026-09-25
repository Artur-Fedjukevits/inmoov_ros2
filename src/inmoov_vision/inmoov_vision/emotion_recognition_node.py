#!/usr/bin/env python3
"""
emotion_recognition_node.py
===========================
Detects facial emotion using hsemotion-onnx (EfficientNet-B0 / AffectNet).

Redundancy: works with the left eye by default. If the left eye is
unavailable (no tracks for > STALE_SEC), automatically switches to the
right eye. Switches back as soon as the left eye sends tracks again.

Subscriptions:
  /camera/eye_left/compressed   (sensor_msgs/CompressedImage)
  /camera/eye_right/compressed  (sensor_msgs/CompressedImage)
  /face/tracks/left             (String JSON)
  /face/tracks/right            (String JSON)

Publishes:
  /face/emotion  (String JSON)
  {
    "track_id": 3,
    "emotion":  "happy",
    "confidence": 0.89,
    "source": "left",
    "all": {"angry":0.01, "happy":0.89, ...}
  }

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import collections
import json
import threading
import time

import cv2
import numpy as np

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String

_STALE_SEC = 2.0  # unavailability threshold (s)
# Frames are buffered (still JPEG-compressed) so that the face is cropped from
# the SAME frame the tracks bbox was computed on (tracks carry that frame's
# 'stamp'), not from the newest frame. ~1.3 s at 15 fps covers detection lag.
_FRAME_BUF_LEN   = 20
_STAMP_MATCH_SEC = 0.02   # a frame within this of the tracks stamp is "the" frame

_LABEL_MAP = {
    'Anger':    'angry',
    'Contempt': 'contempt',
    'Disgust':  'disgust',
    'Fear':     'fear',
    'Happiness':'happy',
    'Neutral':  'neutral',
    'Sadness':  'sad',
    'Surprise': 'surprise',
}


# Camera frames: newest only, no retransmits (face_capture publishes RELIABLE —
# a BEST_EFFORT subscriber is compatible with it)
_CAMERA_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


class EmotionRecognitionNode(LifecycleNode):
    def __init__(self):
        super().__init__('emotion_recognition_node')
        self._pub          = None
        self._timer        = None
        self._frame_lock   = threading.Lock()
        self._tracks_lock  = threading.Lock()
        # deque of (stamp, jpeg bytes), newest last
        self._left_frames  = collections.deque(maxlen=_FRAME_BUF_LEN)
        self._right_frames = collections.deque(maxlen=_FRAME_BUF_LEN)
        self._left_tracks  = self._right_tracks = []
        self._left_stamp   = self._right_stamp  = 0.0
        self._last_left_t  = self._last_right_t = 0.0
        self._sleeping     = False
        self._busy         = False
        # Lifecycle ACTIVE flag + generation: frames are ignored while INACTIVE, and a
        # worker started before deactivate→activate must not publish its stale result.
        self._lc_active    = False
        self._generation   = 0
        self._active_side  = 'left'

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores repeated declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('analysis_hz',   2.0)
        self._dp('min_face_size', 48)

        self._analysis_hz = self.get_parameter('analysis_hz').value
        self._min_face    = self.get_parameter('min_face_size').value

        self.get_logger().info('Loading hsemotion-onnx emotion model...')
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        self._recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf')
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self._recognizer.predict_emotions(dummy, logits=False)
        self.get_logger().info('Emotion model ready (onnxruntime, AVX512)')

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(CompressedImage, '/camera/eye_left/compressed',  self._left_frame_cb, _CAMERA_QOS)
        self.create_subscription(CompressedImage, '/camera/eye_right/compressed', self._right_frame_cb, _CAMERA_QOS)
        self.create_subscription(String, '/face/tracks/left',  self._left_tracks_cb,  10)
        self.create_subscription(String, '/face/tracks/right', self._right_tracks_cb, 10)
        self.create_subscription(Bool, '/robot_sleep', self._sleep_cb, latched_qos)
        self._pub = self.create_lifecycle_publisher(String, '/face/emotion', 10)
        self.get_logger().info(f'EmotionRecognition configured (hz={self._analysis_hz})')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._lc_active = True
        self._pub.on_activate(state)
        self._timer = self.create_timer(1.0 / self._analysis_hz, self._trigger)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self._lc_active = False
        self._generation += 1
        with self._frame_lock:
            self._left_frames.clear()
            self._right_frames.clear()
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

    def _sleep_cb(self, msg: Bool):
        self._sleeping = msg.data

    # ── Frame callbacks ───────────────────────────────────────────────────

    def _left_frame_cb(self, msg: CompressedImage):
        if self._sleeping or not self._lc_active:
            return
        with self._frame_lock:
            self._left_frames.append((self._msg_stamp(msg), bytes(msg.data)))

    def _right_frame_cb(self, msg: CompressedImage):
        if self._sleeping or not self._lc_active:
            return
        with self._frame_lock:
            self._right_frames.append((self._msg_stamp(msg), bytes(msg.data)))

    @staticmethod
    def _msg_stamp(msg: CompressedImage) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    @staticmethod
    def _decode(data: bytes) -> np.ndarray | None:
        try:
            return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None

    @staticmethod
    def _frame_for_stamp(frames, stamp: float) -> bytes | None:
        """The buffered JPEG whose stamp matches the tracks stamp, or None."""
        best = min(frames, key=lambda f: abs(f[0] - stamp), default=None)
        if best is None or abs(best[0] - stamp) > _STAMP_MATCH_SEC:
            return None
        return best[1]

    # ── Track callbacks ───────────────────────────────────────────────────

    def _left_tracks_cb(self, msg: String):
        if self._sleeping:
            return
        try:
            data = json.loads(msg.data)
            with self._tracks_lock:
                self._left_tracks  = data.get('tracks', [])
                self._left_stamp   = float(data.get('stamp', 0.0))
                self._last_left_t  = time.time()
        except Exception:
            pass

    def _right_tracks_cb(self, msg: String):
        if self._sleeping:
            return
        try:
            data = json.loads(msg.data)
            with self._tracks_lock:
                self._right_tracks  = data.get('tracks', [])
                self._right_stamp   = float(data.get('stamp', 0.0))
                self._last_right_t  = time.time()
        except Exception:
            pass

    # ── Analysis trigger ──────────────────────────────────────────────────

    def _trigger(self):
        if self._sleeping or self._busy:
            return

        now = time.time()
        with self._tracks_lock:
            left_fresh  = (now - self._last_left_t)  < _STALE_SEC
            right_fresh = (now - self._last_right_t) < _STALE_SEC
            left_tracks  = list(self._left_tracks)
            right_tracks = list(self._right_tracks)
            left_stamp   = self._left_stamp
            right_stamp  = self._right_stamp

        # Choose the active source: left has priority.
        # Fall back to the right only if the left camera is dead (not sending
        # messages at all), not simply when there are no faces in the frame.
        if left_fresh:
            if not left_tracks:
                return  # left is alive but no faces — don't switch to the right
            tracks, side, stamp, frames = left_tracks, 'left', left_stamp, self._left_frames
        elif right_fresh and right_tracks:
            tracks, side, stamp, frames = right_tracks, 'right', right_stamp, self._right_frames
        else:
            return

        with self._frame_lock:
            jpeg = self._frame_for_stamp(frames, stamp)
        if jpeg is None:
            return  # the bbox's frame is not buffered (yet / any more) — skip this tick
        frame = self._decode(jpeg)
        if frame is None:
            return

        if side != self._active_side:
            self.get_logger().info(
                f'EmotionRecognition: switching {self._active_side} → {side}')
            self._active_side = side

        self._busy = True
        threading.Thread(
            target=self._analyze, args=(frame, tracks, side, self._generation),
            daemon=True).start()

    # ── Emotion analysis ──────────────────────────────────────────────────

    def _analyze(self, frame: np.ndarray, tracks: list, source: str, generation: int):
        try:
            h, w = frame.shape[:2]
            for track in tracks:
                bbox = track.get('bbox', [])
                if len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if (x2 - x1) < self._min_face or (y2 - y1) < self._min_face:
                    continue

                crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                try:
                    dominant_raw, scores = self._recognizer.predict_emotions(
                        crop, logits=False)
                    labels    = self._recognizer.idx_to_class
                    dominant  = _LABEL_MAP.get(dominant_raw, dominant_raw.lower())
                    idx       = next(i for i, v in labels.items() if v == dominant_raw)
                    confidence = float(scores[idx])

                    all_scores = {
                        _LABEL_MAP.get(labels[i], labels[i].lower()): round(float(scores[i]), 3)
                        for i in labels
                    }

                    if generation != self._generation:
                        return   # deactivated while inferring
                    msg = String()
                    msg.data = json.dumps({
                        'track_id':   track['track_id'],
                        'emotion':    dominant,
                        'confidence': round(confidence, 3),
                        'source':     source,
                        'all':        all_scores,
                    })
                    self._pub.publish(msg)

                except Exception as e:
                    self.get_logger().debug(f'Emotion skip: {e}')
        finally:
            self._busy = False




def main():
    rclpy.init()
    node = EmotionRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
