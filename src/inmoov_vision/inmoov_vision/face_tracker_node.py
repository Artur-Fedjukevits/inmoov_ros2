#!/usr/bin/env python3
"""
face_tracker_node.py
====================
IOU tracker: assigns persistent track_ids to faces across detection frames.
The camera_side parameter ('left' | 'right') determines:
  - where to read from: /face/detections/{side}
  - where to publish:   /face/tracks/{side}

Format of /face/tracks/{side}:
  {
    "stamp": 1234567890.0,
    "tracks": [
      {
        "track_id":  3,
        "bbox":      [x1, y1, x2, y2],
        "det_score": 0.97,
        "embedding": [...],
        "kps":       [...],
        "age":       5
      }
    ]
  }

Lip activity (who is speaking — step 1, log/topic only, no decisions yet):
  between the 1.5 Hz face detections the node runs InsightFace 2d106det on
  each active track's face crop of /camera/eye_{side}/compressed at lip_hz,
  keeps a mouth-openness series per track, and for every phrase on
  /voice/segment (voice_detector: time window + loudness envelope) publishes
  per-track mouth excursion vs rest + verdict on /face/mouth_activity/{side}
  (see lip_activity.analyze_segment for the fields).

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import collections
import json
import os
import threading

import cv2
import numpy as np
import onnxruntime as ort

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Bool

from inmoov_vision.lip_activity import analyze_segment, landmarks_bbox, mouth_openness

# Camera frames: newest only (same as face_detection_node)
_CAMERA_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
# Wait this long after a phrase ends before analyzing it: the lip worker may still
# be processing the last frames, and a little rest after the speech is wanted (the
# phrase already ends with silence_duration_sec = 1.0 s of it). Kept short — llm_node
# waits for this verdict before accepting a gaze-only address (since 2026-10-04).
_SEGMENT_SETTLE_SEC = 0.5


def _smooth_bbox(prev: list | None, new: list,
                 center_alpha: float = 0.4, size_alpha: float = 0.1) -> list:
    """EMA of the lip crop. Feeding each frame's landmark box straight back as the
    next crop was a jitter loop: on a ~85 px face a 3-8% crop shift alone moves
    the openness by 0.01-0.07 — as much as speech itself (live log 2026-10-01:
    act≈base≈0.06 for a person talking to the robot)."""
    if prev is None:
        return new
    pcx, pcy = (prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2
    ncx, ncy = (new[0] + new[2]) / 2, (new[1] + new[3]) / 2
    cx = pcx + center_alpha * (ncx - pcx)
    cy = pcy + center_alpha * (ncy - pcy)
    w = (prev[2] - prev[0]) + size_alpha * ((new[2] - new[0]) - (prev[2] - prev[0]))
    h = (prev[3] - prev[1]) + size_alpha * ((new[3] - new[1]) - (prev[3] - prev[1]))
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _iou(a: list, b: list) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class FaceTrackerNode(LifecycleNode):
    def __init__(self):
        super().__init__('face_tracker_node')
        self._pub     = None
        self._tracks  = {}
        self._next_id = 1
        # Lip activity — see the module docstring
        self._lock          = threading.Lock()   # _tracks/_mouth/_lip_bbox vs the lip worker
        self._lip_model     = None
        self._mouth_pub     = None
        self._mouth: dict[int, collections.deque] = {}   # tid → (stamp, openness)
        self._lip_bbox: dict[int, list] = {}             # tid → crop for the next frame
        self._lip_frame     = None                       # (jpeg, stamp)
        self._lip_last_stamp = 0.0
        self._lip_busy      = False
        self._lip_timer     = None
        self._vision_on     = False
        self._lc_active     = False

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores re-declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('camera_side',     'left')
        self._dp('iou_threshold',   0.35)
        self._dp('max_lost_frames', 30)
        self._dp('max_tracks',      8)
        self._dp('lip_enabled',     True)
        self._dp('lip_hz',          15.0)
        self._dp('lip_buffer_sec',  20.0)
        self._dp('lip_model',       '~/.insightface/models/buffalo_l/2d106det.onnx')

        side             = self.get_parameter('camera_side').value
        self._iou_thr    = self.get_parameter('iou_threshold').value
        self._max_lost   = self.get_parameter('max_lost_frames').value
        self._max_tracks = self.get_parameter('max_tracks').value
        self._tracks     = {}
        self._next_id    = 1
        self._side       = side
        self._lip_hz     = float(self.get_parameter('lip_hz').value)
        self._lip_buf_sec = float(self.get_parameter('lip_buffer_sec').value)
        self._lip_model  = None
        if self.get_parameter('lip_enabled').value:
            self._lip_model = self._load_lip_model(
                os.path.expanduser(self.get_parameter('lip_model').value))

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(String, f'/face/detections/{side}', self._callback, 10)
        self.create_subscription(Bool, '/face_detection/enable', self._enable_cb, latched_qos)
        self._pub = self.create_lifecycle_publisher(String, f'/face/tracks/{side}', 10)
        if self._lip_model is not None:
            self.create_subscription(CompressedImage, f'/camera/eye_{side}/compressed',
                                     self._frame_cb, _CAMERA_QOS)
            self.create_subscription(String, '/voice/segment', self._segment_cb, 10)
            self._mouth_pub = self.create_lifecycle_publisher(
                String, f'/face/mouth_activity/{side}', 10)
        self.get_logger().info(
            f'FaceTracker ready (side={side}, lips={"on" if self._lip_model else "off"})')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._pub.on_activate(state)
        self._lc_active = True
        if self._lip_model is not None:
            self._mouth_pub.on_activate(state)
            self._lip_timer = self.create_timer(1.0 / self._lip_hz, self._lip_tick)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self._lc_active = False
        if self._lip_timer is not None:
            self.destroy_timer(self._lip_timer)
            self._lip_timer = None
        if self._mouth_pub is not None:
            self._mouth_pub.on_deactivate(state)
        self._pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

    def _enable_cb(self, msg: Bool):
        self._vision_on = msg.data
        if not msg.data:
            # When vision is disabled, drop all tracks — start clean on the next enable
            if self._tracks:
                self.get_logger().info(
                    f'Vision disabled — dropping {len(self._tracks)} tracks')
            with self._lock:
                self._tracks  = {}
                self._next_id = 1
                self._mouth.clear()
                self._lip_bbox.clear()

    def _callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'JSON error: {e}')
            return
        with self._lock:
            msg_out = self._update_tracks(data)
        self._pub.publish(msg_out)

    def _update_tracks(self, data: dict) -> String:
        detections = data.get('faces', [])
        stamp      = data.get('stamp', 0.0)

        # ── Step 1: match existing tracks to detections by IOU (tracks × detections) ──
        track_ids  = list(self._tracks.keys())
        matched_t  = set()
        matched_d  = set()

        for ti, tid in enumerate(track_ids):
            best_iou, best_di = 0.0, -1
            for di, det in enumerate(detections):
                if di in matched_d:
                    continue
                score = _iou(self._tracks[tid]['bbox'], det['bbox'])
                if score > best_iou:
                    best_iou, best_di = score, di

            if best_iou >= self._iou_thr and best_di >= 0:
                det = detections[best_di]
                self._tracks[tid].update({
                    'bbox':      det['bbox'],
                    'det_score': det.get('det_score', 1.0),
                    'kps':       det.get('kps', []),
                    'lost':      0,
                    'age':       self._tracks[tid]['age'] + 1,
                })
                if 'embedding' in det:
                    self._tracks[tid]['embedding'] = det['embedding']
                # The lip crop follows the face between detections; drop it if it
                # drifted away from this (authoritative) detection
                lip_bb = self._lip_bbox.get(tid)
                if lip_bb is not None and _iou(lip_bb, det['bbox']) < 0.3:
                    self._lip_bbox.pop(tid, None)
                matched_t.add(tid)
                matched_d.add(best_di)

        # ── Step 2: unmatched detections → new tracks ─────────────────────
        for di, det in enumerate(detections):
            if di in matched_d:
                continue
            if len(self._tracks) >= self._max_tracks:
                break
            self._tracks[self._next_id] = {
                'bbox':      det['bbox'],
                'det_score': det.get('det_score', 1.0),
                'embedding': det.get('embedding', []),
                'kps':       det.get('kps', []),
                'age':       1,
                'lost':      0,
            }
            self._next_id += 1

        # ── Step 3: unmatched tracks → marked as lost ─────────────────────
        to_delete = []
        for tid in track_ids:
            if tid not in matched_t:
                self._tracks[tid]['lost'] += 1
                if self._tracks[tid]['lost'] > self._max_lost:
                    to_delete.append(tid)
        for tid in to_delete:
            del self._tracks[tid]
            self._mouth.pop(tid, None)
            self._lip_bbox.pop(tid, None)

        # ── Publish only active tracks (lost==0) ──────────────────────────
        # Lost tracks are kept internally for IOU re-matching but are NOT
        # published — otherwise head_tracker would follow a stale bbox.
        tracks_out = []
        for tid, t in self._tracks.items():
            if t['lost'] > 0:
                continue
            tracks_out.append({
                'track_id':  tid,
                'bbox':      t['bbox'],
                'det_score': t['det_score'],
                'embedding': t.get('embedding', []),
                'kps':       t.get('kps', []),
                'age':       t['age'],
            })

        msg_out = String()
        msg_out.data = json.dumps({'stamp': stamp, 'tracks': tracks_out})
        return msg_out

    # ── Lip activity ───────────────────────────────────────────────────────

    def _load_lip_model(self, path: str):
        if not os.path.exists(path):
            self.get_logger().warn(f'Lip model not found ({path}) — lip activity off')
            return None
        from insightface.model_zoo import get_model
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1   # ~2 ms per face crop on one core
        so.inter_op_num_threads = 1
        model = get_model(path, providers=['CPUExecutionProvider'], sess_options=so)
        model.prepare(ctx_id=0)
        return model

    def _frame_cb(self, msg: CompressedImage):
        # face_capture republishes the OTHER eye's frame when this camera fails —
        # its geometry doesn't match our tracks' bboxes
        if not self._lc_active or msg.header.frame_id not in ('', f'eye_{self._side}'):
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._lip_frame = (bytes(msg.data), stamp)

    def _lip_tick(self):
        frame = self._lip_frame
        if (self._lip_busy or not self._vision_on or frame is None
                or frame[1] == self._lip_last_stamp):
            return
        with self._lock:
            crops = {tid: (self._lip_bbox.get(tid) or t['bbox'])
                     for tid, t in self._tracks.items() if t['lost'] == 0}
        if not crops:
            return
        self._lip_last_stamp = frame[1]
        self._lip_busy = True
        threading.Thread(target=self._lip_work, args=(frame, crops), daemon=True).start()

    def _lip_work(self, frame, crops: dict):
        from insightface.app.common import Face
        try:
            jpeg, stamp = frame
            img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return
            results = {}
            for tid, bbox in crops.items():
                lm = self._lip_model.get(img, Face(bbox=np.array(bbox, dtype=np.float32)))
                new_bb = landmarks_bbox(lm)
                # Sanity: the landmark face must roughly match the crop (else the
                # crop slid off the face — fall back to the detection bbox)
                w0 = bbox[2] - bbox[0]
                w1 = new_bb[2] - new_bb[0]
                cx0, cx1 = (bbox[0] + bbox[2]) / 2, (new_bb[0] + new_bb[2]) / 2
                if not (0.5 < w1 / max(w0, 1.0) < 2.0 and abs(cx1 - cx0) < 0.5 * w0):
                    results[tid] = (None, None)
                    continue
                results[tid] = (mouth_openness(lm), new_bb)
            with self._lock:
                for tid, (openness, new_bb) in results.items():
                    if tid not in self._tracks:
                        continue   # the track died while we were working
                    if new_bb is None:
                        self._lip_bbox.pop(tid, None)
                        continue
                    self._lip_bbox[tid] = _smooth_bbox(self._lip_bbox.get(tid), new_bb)
                    if openness is not None:
                        buf = self._mouth.setdefault(tid, collections.deque())
                        buf.append((stamp, openness))
                        while buf and buf[0][0] < stamp - self._lip_buf_sec:
                            buf.popleft()
        except Exception as e:
            self.get_logger().error(f'Lip worker error: {e}')
        finally:
            self._lip_busy = False

    def _segment_cb(self, msg: String):
        try:
            seg = json.loads(msg.data)
        except Exception:
            return
        timer = threading.Timer(_SEGMENT_SETTLE_SEC, self._analyze_segment, args=(seg,))
        timer.daemon = True
        timer.start()

    def _analyze_segment(self, seg: dict):
        if not self._lc_active:
            return
        with self._lock:
            series = {tid: list(buf) for tid, buf in self._mouth.items()}
        t0, t1 = seg['t_start'], seg['t_end']
        tracks = []
        for tid, samples in series.items():
            res = analyze_segment(samples, t0, t1, seg.get('envelope', []),
                                  seg.get('env_hz', 20), seg.get('vad'))
            if res is not None:
                tracks.append({'track_id': tid, **res})
        self._mouth_pub.publish(String(data=json.dumps({
            'segment_id': seg.get('id'), 't_start': t0, 't_end': t1,
            'sv_rejected': bool(seg.get('sv_rejected')),
            'side': self._side, 'tracks': tracks})))
        if not tracks:
            self.get_logger().info(
                f'Lips seg#{seg.get("id")} ({t1 - t0:.1f}s): no face in view during the phrase')
            return
        parts = ' | '.join(
            f'track {t["track_id"]}: {t["verdict"].upper()} excess={t["excess"]:.3f} '
            f'(p90={t["p90"]:.3f} rest={t["rest"]}) open={t["open_frac"]} '
            f'sync={t["sync"]} speech={t["speech_sec"]}s cov={t["coverage"]} n={t["n"]}'
            for t in sorted(tracks, key=lambda t: -t['excess']))
        self.get_logger().info(f'Lips seg#{seg.get("id")} ({t1 - t0:.1f}s{", SV-rejected voice" if seg.get("sv_rejected") else ""}): {parts}')


def main():
    rclpy.init()
    node = FaceTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
