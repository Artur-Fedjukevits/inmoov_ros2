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

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import json
import time

import numpy as np

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String, Bool


def _iou(a: list, b: list) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
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

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores re-declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('camera_side',     'left')
        self._dp('iou_threshold',   0.35)
        self._dp('max_lost_frames', 30)
        self._dp('max_tracks',      8)

        side             = self.get_parameter('camera_side').value
        self._iou_thr    = self.get_parameter('iou_threshold').value
        self._max_lost   = self.get_parameter('max_lost_frames').value
        self._max_tracks = self.get_parameter('max_tracks').value
        self._tracks     = {}
        self._next_id    = 1

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(String, f'/face/detections/{side}', self._callback, 10)
        self.create_subscription(Bool, '/face_detection/enable', self._enable_cb, latched_qos)
        self._pub = self.create_lifecycle_publisher(String, f'/face/tracks/{side}', 10)
        self.get_logger().info(f'FaceTracker ready (side={side})')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._pub.on_activate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self._pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

    def _enable_cb(self, msg: Bool):
        if not msg.data:
            # When vision is disabled, drop all tracks — start clean on the next enable
            if self._tracks:
                self.get_logger().info(
                    f'Vision disabled — dropping {len(self._tracks)} tracks')
            self._tracks  = {}
            self._next_id = 1

    def _callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'JSON error: {e}')
            return

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
        self._pub.publish(msg_out)



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
