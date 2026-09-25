#!/usr/bin/env python3
"""
face_detection_node.py
======================
Runs insightface on frames from the given eye.
The camera_side parameter ('left' | 'right') determines:
  - which camera to read from:   /camera/eye_{side}/compressed
  - where to publish to:         /face/detections/{side}

insightface runs in a background thread.
Analysis rate is controlled by the detection_hz parameter (default 1.5 Hz).

On-demand fallback (fallback_for parameter):
  If set to a non-empty value (the other eye's topic, e.g.
  '/face/detections/left'), the node does not run insightface continuously —
  it listens to that topic as the primary camera's heartbeat and only starts
  its own detection once the primary hasn't published for longer than
  primary_timeout_sec (default 1.5s). Saves CPU: the right eye (fallback)
  doesn't grind insightface in parallel with the left (primary) 24/7 — it
  only kicks in once the left has actually disappeared (camera disconnected /
  process crashed) — i.e. a real fallback, not permanent duplication.

Subscriptions:
  /camera/eye_{camera_side}/compressed  (sensor_msgs/CompressedImage)

Publishes:
  /face/detections/{camera_side}  (std_msgs/String — JSON)

JSON format:
  {
    "stamp": 1234567890.0,
    "faces": [
      {
        "bbox":      [x1, y1, x2, y2],
        "det_score": 0.98,
        "embedding": [0.01, -0.03, ...],   // 512-d normed (buffalo_l only)
        "kps":       [[x,y], ...]          // 5 points
      }
    ]
  }

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
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

# Camera frames: newest only, no retransmits (face_capture publishes RELIABLE —
# a BEST_EFFORT subscriber is compatible with it)
_CAMERA_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


class FaceDetectionNode(LifecycleNode):
    def __init__(self):
        super().__init__('face_detection_node')
        self._pub          = None
        self._timers       = []
        self._frame_lock   = threading.Lock()
        self._latest_frame = None
        self._latest_stamp = 0.0
        self._busy         = False
        # Lifecycle ACTIVE flag + generation: frames are ignored while INACTIVE, and a
        # worker started before deactivate→activate must not publish its stale result.
        self._lc_active    = False
        self._generation   = 0
        self._enabled      = False
        self._last_frame_t = self._last_detect_t = self._no_face_since = 0.0
        self._fallback_for       = ''
        self._primary_timeout    = 1.5
        self._last_primary_t     = 0.0
        self._fallback_active    = False
        # Diagnostics for a "stuck" camera (see _trigger_detection) — a frame
        # arrives on schedule, but its stamp doesn't change.
        self._last_processed_stamp   = -1.0
        self._stale_frame_streak     = 0
        self._STALE_FRAME_WARN_STREAK = 3

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores repeated declaration on re-configure."""
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

        # By default ORT spins up a thread pool across all logical cores for
        # EACH of the ~5 buffalo_l models (intra_op_num_threads=0 = auto).
        # With two processes (left+right) on 16 logical cores this causes
        # thread contention and ~350-370% CPU per process. We limit the pool
        # explicitly — insightface.FaceAnalysis(**kwargs) forwards
        # sess_options to onnxruntime.InferenceSession unchanged.
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = intra_op
        sess_options.inter_op_num_threads = inter_op

        self.get_logger().info(f'Loading insightface ({model_name}, side={self._side})...')
        self._app = FaceAnalysis(
            name=model_name, providers=['CPUExecutionProvider'], sess_options=sess_options)
        self._app.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=det_thresh)
        self.get_logger().info('insightface loaded')

        from std_msgs.msg import Bool as _Bool
        cam_topic = f'/camera/eye_{self._side}/compressed'
        det_topic = f'/face/detections/{self._side}'
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        # BEST_EFFORT/depth 1: only the newest frame matters, no retransmits of stale ones
        self.create_subscription(CompressedImage, cam_topic, self._frame_callback, _CAMERA_QOS)
        self.create_subscription(_Bool, '/face_detection/enable', self._enable_cb, latched_qos)
        if self._fallback_for:
            self.create_subscription(
                String, self._fallback_for, self._primary_heartbeat_cb, 5)
        self._pub = self.create_lifecycle_publisher(String, det_topic, 10)
        fb_note = f', fallback_for={self._fallback_for} (timeout={self._primary_timeout}s)' \
            if self._fallback_for else ''
        self.get_logger().info(
            f'FaceDetection configured (side={self._side}, hz={self._det_hz}, '
            f'det_size={det_size}{fb_note})')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._lc_active = True
        self._pub.on_activate(state)
        # Grace period: consider primary alive from the moment of activation,
        # not from the epoch (otherwise _last_primary_t=0.0 → time.time()-0
        # is huge > primary_timeout → a false "primary silent" on the very
        # first tick, before primary has even had a chance to send its first
        # heartbeat).
        self._last_primary_t = time.time()
        self._timers.append(self.create_timer(1.0 / self._det_hz, self._trigger_detection))
        self._timers.append(self.create_timer(5.0, self._watchdog))
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self._lc_active = False
        self._generation += 1
        for t in self._timers:
            self.destroy_timer(t)
        self._timers.clear()
        with self._frame_lock:
            self._latest_frame = None   # don't detect on a pre-sleep frame after WAKE
        self._pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

    def _frame_callback(self, msg: CompressedImage):
        """Store the latest JPEG without processing it — decoded only when a detection
        actually runs (det_hz, and only while enabled), not for every camera frame."""
        if not self._lc_active:
            return
        frame = msg.data   # JPEG bytes; decoded in _detect
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_stamp = stamp
        self._last_frame_t = time.time()

    def _enable_cb(self, msg) -> None:
        self._enabled = msg.data
        self._no_face_since = 0.0
        if msg.data:
            # Grace period on EVERY enable (not just on_activate — the lifecycle
            # activate itself happens once when the stack starts, whereas enable
            # is toggled further on by PIRScan/wakeup). Without resetting here
            # _last_primary_t stays stale from last time → primary_alive is
            # immediately falsely False on the first tick.
            self._last_primary_t = time.time()
            self._fallback_active = False
            self._last_processed_stamp = -1.0
            self._stale_frame_streak   = 0
        role = f', fallback-reserve for {self._fallback_for}' if self._fallback_for else ', primary'
        self.get_logger().info(
            f'FaceDetection: {"enabled" if msg.data else "disabled"}{role}')

    def _primary_heartbeat_cb(self, msg) -> None:
        """Receiving ANY message from the primary camera = it's alive, whether or not it found a face."""
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
                        f'Primary ({self._fallback_for}) recovered — fallback ({self._side}) back to standby')
                self._fallback_active = False
                return
            if not self._fallback_active:
                self._fallback_active = True
                self.get_logger().warn(
                    f'Primary ({self._fallback_for}) silent >{self._primary_timeout}s — '
                    f'activating fallback detection on {self._side}')
        with self._frame_lock:
            if self._latest_frame is None:
                return
            frame = self._latest_frame   # immutable JPEG bytes — no copy needed
            stamp = self._latest_stamp

        # Live bug 2026-08-31/09-01: suspicion that the USB camera in the eye
        # can physically "freeze" (the sensor stops updating), but frames with
        # the last successful image keep arriving on /camera/eye_* —
        # _frame_callback dutifully updates _last_frame_t on EVERY message, so
        # the usual "camera is silent" watchdog doesn't catch this. Previously
        # this caused head_tracker to receive "fresh" (by message time)
        # detections with an UNCHANGED bbox/offset — the head would drift to
        # one side without stopping, because there was no actually new image.
        # We compare the FRAME's stamp (not the message's) with the previous
        # detection run — if it hasn't advanced, it's the same frame.
        if stamp == self._last_processed_stamp:
            self._stale_frame_streak += 1
            if self._stale_frame_streak == self._STALE_FRAME_WARN_STREAK:
                self.get_logger().warn(
                    f'FaceDetection ({self._side}): frame NOT updating for '
                    f'{self._stale_frame_streak} detections in a row (stamp='
                    f'{stamp:.3f}) — camera may have "frozen" (messages '
                    f'arriving, but the picture is the same)')
        else:
            self._stale_frame_streak = 0
        self._last_processed_stamp = stamp

        self._last_detect_t = time.time()
        self._busy = True
        threading.Thread(
            target=self._detect, args=(frame, stamp, self._generation), daemon=True).start()

    def _detect(self, jpeg, stamp: float, generation: int):
        try:
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return
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
                    f'Faces detected: {len(result["faces"])}')
            else:
                if self._no_face_since == 0.0:
                    self._no_face_since = time.time()

            if generation != self._generation:
                return   # deactivated (and maybe re-activated) while inferring
            msg = String()
            msg.data = json.dumps(result)
            self._pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f'Detection error: {e}')
        finally:
            self._busy = False


    def _watchdog(self):
        if not self._enabled:
            return
        now = time.time()
        # No frames arriving?
        if self._last_frame_t > 0.0 and (now - self._last_frame_t) > 3.0:
            self.get_logger().warn(
                f'Camera silent for {now - self._last_frame_t:.1f}s — no frames from /camera/eye_{self._side}/compressed')
        elif self._last_frame_t == 0.0:
            self.get_logger().warn('No frames received from the camera yet')
        # Detection not running? (a standby fallback node not detecting is normal — no warning)
        in_standby = bool(self._fallback_for) and not self._fallback_active
        if not in_standby and self._last_detect_t > 0.0 and (now - self._last_detect_t) > 3.0:
            self.get_logger().warn(
                f'Detection not running for {now - self._last_detect_t:.1f}s (busy={self._busy})')
        # Face not found for a long time
        if self._no_face_since > 0.0 and (now - self._no_face_since) > 10.0:
            self.get_logger().warn(
                f'No face detected for {now - self._no_face_since:.0f}s')
            self._no_face_since = now  # reset to avoid spamming every 5s




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
