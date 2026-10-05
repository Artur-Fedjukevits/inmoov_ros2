#!/usr/bin/env python3
"""
face_gallery_node.py
====================
Automatically saves face crops from both eyes into a photo gallery.

Directory structure:
  {gallery_dir}/
    persons/
      1_Artur/        ← photos from interaction sessions
        left_20260430_143022_a1b2.jpg
        right_20260430_143022_a1b2.jpg
    _pending/          ← photos collected BEFORE introduction completes
        left_20260430_144500_c3d4.jpg

Capture logic:
  INTRODUCING  → shoot every enroll_interval_sec seconds into _pending/
  INTERACTING  → shoot every interact_interval_sec seconds into persons/{id}_{name}/

  After enrollment (INTRODUCING → INTERACTING + person_id appeared):
    _pending/ → persons/{id}_{name}/ + gallery_add for each photo

Photo limit (max_photos_per_person=30):
  If the directory already has 30 photos:
    - only add 1 left + 1 right per session (JPEG quality=100)
    - delete the 2 oldest (oldest left + oldest right) from disk and DB

Embedding recompute:
  At the end of a session (interacting→idle, or going to sleep) —
  gallery_rebuild_embedding for the specific person_id: averages all gallery
  embeddings → updates persons.embedding

Quality filtering:
  - det_score >= min_det_score (0.80)
  - bbox width >= min_face_px (60 pixels)
  - cosine similarity to the last N photos < max_diversity_sim (0.90) → don't duplicate

Subscriptions:
  /camera/eye_left/compressed   — for cropping with the left bbox
  /camera/eye_right/compressed  — for cropping with the right bbox
  /face/tracks/left             — bbox + det_score + embedding (main track)
  /face/tracks/right            — bbox + det_score of the right eye
  /social_context                — person_id + state (INTERACTING/INTRODUCING)

Service:
  /memory/query  — gallery_add, gallery_remove, gallery_rebuild_embedding

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import json
import os
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String
from inmoov_msgs.srv import MemoryQuery


# Camera frames: newest only, no retransmits (face_capture publishes RELIABLE —
# a BEST_EFFORT subscriber is compatible with it)
_CAMERA_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


class FaceGalleryNode(LifecycleNode):
    def __init__(self):
        super().__init__('face_gallery_node')
        self._timer                = None
        self._lock                 = threading.Lock()
        self._sleeping             = False
        self._lc_active            = False   # lifecycle ACTIVE: frames ignored otherwise
        self._last_left            = self._last_right = None
        self._left_bbox            = self._right_bbox = None
        self._left_det_score       = self._right_det_score = 0.0
        self._left_embedding       = self._right_embedding = None
        self._state                = 'idle'
        self._person_id            = None
        self._person_name          = ''
        self._prev_state           = 'idle'
        self._last_capture         = 0.0
        self._pending_photos       = []
        self._recent_embeddings    = []
        self._session_person_id    = None
        self._session_left_saved   = self._session_right_saved = False
        self._last_interacting_pid = None
        self._target_track_id      = None

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores repeated declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('gallery_dir',            os.path.expanduser('~/inmoov_faces'))
        self._dp('enroll_interval_sec',    1.0)
        self._dp('interact_interval_sec',  15.0)
        self._dp('min_det_score',          0.80)
        self._dp('min_face_px',            60)
        self._dp('max_diversity_sim',      0.90)
        self._dp('max_pending_photos',     40)
        self._dp('max_photos_per_person',  30)

        self._gallery_dir   = Path(self.get_parameter('gallery_dir').value)
        self._enroll_iv     = self.get_parameter('enroll_interval_sec').value
        self._interact_iv   = self.get_parameter('interact_interval_sec').value
        self._min_det_score = self.get_parameter('min_det_score').value
        self._min_face_px   = self.get_parameter('min_face_px').value
        self._max_div_sim   = self.get_parameter('max_diversity_sim').value
        self._max_pending   = self.get_parameter('max_pending_photos').value
        self._max_photos    = self.get_parameter('max_photos_per_person').value

        (self._gallery_dir / 'persons').mkdir(parents=True, exist_ok=True)
        (self._gallery_dir / '_pending').mkdir(parents=True, exist_ok=True)

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(CompressedImage, '/camera/eye_left/compressed',  self._left_cb, _CAMERA_QOS)
        self.create_subscription(CompressedImage, '/camera/eye_right/compressed', self._right_cb, _CAMERA_QOS)
        self.create_subscription(String, '/face/tracks/left',  self._left_tracks_cb,  10)
        self.create_subscription(String, '/face/tracks/right', self._right_tracks_cb, 10)
        self.create_subscription(String, '/social_context',    self._social_cb,        10)
        self.create_subscription(String, '/face/identity',     self._identity_cb,      10)
        self.create_subscription(Bool, '/robot_sleep', self._sleep_cb, latched_qos)
        self._mem = self.create_client(MemoryQuery, '/memory/query')
        self.get_logger().info(
            f'FaceGallery configured | dir={self._gallery_dir} | max_photos={self._max_photos}')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._lc_active = True
        self._timer = self.create_timer(0.5, self._capture_tick)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self._lc_active = False
        if self._timer:
            self.destroy_timer(self._timer)
            self._timer = None
        # Pre-sleep frames/bboxes must not be saved as a "fresh" photo after WAKE
        with self._lock:
            self._last_left = self._last_right = None
            self._left_bbox = self._right_bbox = None
            self._left_embedding = self._right_embedding = None
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

    def _sleep_cb(self, msg: Bool):
        was_sleeping = self._sleeping
        self._sleeping = msg.data
        if msg.data and not was_sleeping:
            with self._lock:
                pid = self._last_interacting_pid
            if pid is not None:
                self.get_logger().info(
                    f'Sleep → recomputing embedding for person_id={pid}')
                threading.Thread(
                    target=self._rebuild_embedding, args=(pid,), daemon=True).start()

    # ── Callbacks ─────────────────────────────────────────────────────────

    # Frames are kept as JPEG bytes and decoded only when a photo is actually
    # saved (_do_capture), not for every camera frame.
    def _left_cb(self, msg: CompressedImage):
        if self._sleeping or not self._lc_active:
            return
        with self._lock:
            self._last_left = msg.data

    def _right_cb(self, msg: CompressedImage):
        if self._sleeping or not self._lc_active:
            return
        with self._lock:
            self._last_right = msg.data

    @staticmethod
    def _decode(jpeg) -> np.ndarray | None:
        if jpeg is None:
            return None
        try:
            return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _identity_cb(self, msg: String):
        """Update the interlocutor's track_id from recognition results."""
        if self._sleeping:
            return
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        if not data.get('locked') or not data.get('is_known'):
            return
        with self._lock:
            if data.get('person_id') == self._person_id:
                self._target_track_id = data.get('track_id')

    def _left_tracks_cb(self, msg: String):
        if self._sleeping:
            return
        try:
            tracks = json.loads(msg.data).get('tracks', [])
        except Exception:
            return
        with self._lock:
            if not tracks:
                self._left_bbox = None
                self._left_det_score = 0.0
                self._left_embedding = None
                return
            target_id = self._target_track_id
            if target_id is not None:
                # In a dialogue — only take the interlocutor's track
                best = next((t for t in tracks if t.get('track_id') == target_id), None)
                if best is None:
                    return  # interlocutor not visible — don't touch the bbox
            else:
                best = max(tracks, key=lambda t: (
                    (t['bbox'][2] - t['bbox'][0]) * (t['bbox'][3] - t['bbox'][1])))
            self._left_bbox      = best['bbox']
            self._left_det_score = best.get('det_score', 0.0)
            self._left_embedding = best.get('embedding')

    def _right_tracks_cb(self, msg: String):
        if self._sleeping:
            return
        try:
            tracks = json.loads(msg.data).get('tracks', [])
        except Exception:
            return
        with self._lock:
            if not tracks:
                self._right_bbox      = None
                self._right_det_score = 0.0
                self._right_embedding = None
                return
            target_id = self._target_track_id
            if target_id is not None:
                best = next((t for t in tracks if t.get('track_id') == target_id), None)
                if best is None:
                    return
            else:
                best = max(tracks, key=lambda t: (
                    (t['bbox'][2] - t['bbox'][0]) * (t['bbox'][3] - t['bbox'][1])))
            self._right_bbox      = best['bbox']
            self._right_det_score = best.get('det_score', 0.0)
            self._right_embedding = best.get('embedding')

    def _social_cb(self, msg: String):
        if self._sleeping:
            return
        try:
            ctx = json.loads(msg.data)
        except Exception:
            return

        new_state   = ctx.get('state', 'idle')
        person_id   = ctx.get('person_id')
        person_name = ctx.get('name', '')

        with self._lock:
            prev_state        = self._prev_state
            self._prev_state  = new_state
            self._state       = new_state
            self._person_id   = person_id
            self._person_name = person_name

        # Transition INTRODUCING → INTERACTING — move pending photos
        if (prev_state == 'introducing'
                and new_state == 'interacting'
                and person_id is not None):
            threading.Thread(
                target=self._flush_pending,
                args=(person_id, person_name),
                daemon=True,
            ).start()

        # Introduction started — clear the buffer
        if new_state == 'introducing' and prev_state != 'introducing':
            with self._lock:
                self._recent_embeddings = []
                self._pending_photos    = []
                self._last_capture      = 0.0
            self._clear_pending_dir()
            self.get_logger().info('Gallery: introduction started — pending cleared')

        # Interaction started — update session flags
        if new_state == 'interacting' and person_id is not None:
            with self._lock:
                new_session = (
                    person_id != self._session_person_id
                    or prev_state != 'interacting'
                )
                if new_session:
                    self._session_person_id   = person_id
                    self._session_left_saved  = False
                    self._session_right_saved = False
                    self._target_track_id     = None  # wait for confirmation from identity
                self._last_interacting_pid = person_id
        elif new_state != 'interacting':
            with self._lock:
                self._target_track_id = None

        # End of session: was INTERACTING, became non-INTERACTING
        if prev_state == 'interacting' and new_state != 'interacting':
            with self._lock:
                pid = self._last_interacting_pid
            if pid is not None:
                self.get_logger().info(
                    f'Session ended → recomputing embedding for person_id={pid}')
                threading.Thread(
                    target=self._rebuild_embedding, args=(pid,), daemon=True).start()

    # ── Capture ticker ────────────────────────────────────────────────────

    def _capture_tick(self):
        if self._sleeping:
            return
        with self._lock:
            state       = self._state
            person_id   = self._person_id
            person_name = self._person_name
            last_cap    = self._last_capture
            pending_cnt = len(self._pending_photos)

        now = time.time()

        if state == 'introducing':
            if pending_cnt >= self._max_pending:
                return
            if (now - last_cap) < self._enroll_iv:
                return
            self._do_capture(person_id=None, person_name=None, source='enroll')

        elif state == 'interacting' and person_id is not None:
            if (now - last_cap) < self._interact_iv:
                return
            self._do_capture(person_id=person_id, person_name=person_name, source='interact')

    # ── Frame capture ─────────────────────────────────────────────────────

    def _do_capture(self, person_id: int | None, person_name: str | None, source: str):
        with self._lock:
            frame_left  = self._last_left
            frame_right = self._last_right
            left_bbox   = self._left_bbox
            left_score  = self._left_det_score
            left_emb    = self._left_embedding
            right_bbox  = self._right_bbox
            right_score = self._right_det_score
            right_emb   = self._right_embedding

        if left_bbox is None or left_emb is None:
            return
        if left_score < self._min_det_score:
            return
        x1, y1, x2, y2 = [int(v) for v in left_bbox]
        if (x2 - x1) < self._min_face_px:
            return

        emb_arr = np.array(left_emb, dtype=np.float32)
        norm = np.linalg.norm(emb_arr)
        if norm > 0:
            emb_arr /= norm
        if self._is_duplicate(emb_arr):
            return

        frame_left  = self._decode(frame_left)
        frame_right = self._decode(frame_right) if right_bbox is not None else None

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        uid   = uuid.uuid4().hex[:6]

        # ── Limit check (only for a known person) ──────────────────────────
        at_limit = False
        left_photos: list[Path]  = []
        right_photos: list[Path] = []

        if person_id is not None:
            person_dir   = self._get_person_dir(person_id, person_name)
            left_photos  = sorted(person_dir.glob('left_*.jpg'))
            right_photos = sorted(person_dir.glob('right_*.jpg'))
            total        = len(left_photos) + len(right_photos)
            at_limit     = total >= self._max_photos

        if at_limit:
            with self._lock:
                left_done  = self._session_left_saved
                right_done = self._session_right_saved
            if left_done and right_done:
                return  # already added the session photos for this person
            quality = 100  # maximum quality during rotation
        else:
            left_done = right_done = False
            quality   = 90

        saved: list[tuple[str, list]] = []

        # Left eye
        if frame_left is not None and (not at_limit or not left_done):
            path = self._save_crop(
                frame_left, left_bbox, padding=0.20,
                dest=self._dest_path(person_id, person_name, f'left_{stamp}_{uid}.jpg'),
                quality=quality)
            if path:
                saved.append((path, left_emb))
                if at_limit:
                    if left_photos:
                        self._rotate_oldest(left_photos[0])
                    with self._lock:
                        self._session_left_saved = True

        # Right eye — uses right_emb (buffalo_l on the right camera),
        # falls back to left_emb only if the right detector didn't return an embedding
        if (frame_right is not None
                and right_bbox is not None
                and right_score >= self._min_det_score
                and (not at_limit or not right_done)):
            path = self._save_crop(
                frame_right, right_bbox, padding=0.20,
                dest=self._dest_path(person_id, person_name, f'right_{stamp}_{uid}.jpg'),
                quality=quality)
            if path:
                saved.append((path, right_emb if right_emb is not None else left_emb))
                if at_limit:
                    if right_photos:
                        self._rotate_oldest(right_photos[0])
                    with self._lock:
                        self._session_right_saved = True

        if not saved:
            return

        with self._lock:
            self._last_capture = time.time()
            self._recent_embeddings.append(emb_arr)
            if len(self._recent_embeddings) > 20:
                self._recent_embeddings.pop(0)
            if person_id is None:
                self._pending_photos.extend(saved)

        if person_id is not None:
            for path, emb in saved:
                threading.Thread(
                    target=self._gallery_add,
                    args=(person_id, path, emb, left_score, source),
                    daemon=True,
                ).start()

        self.get_logger().debug(
            f'gallery capture: {len(saved)} photo(s) | source={source} | '
            f'det={left_score:.2f} | at_limit={at_limit}')

    def _rotate_oldest(self, oldest_path: Path):
        """Deletes the oldest photo from disk and from the gallery DB."""
        path_str = str(oldest_path)
        try:
            oldest_path.unlink(missing_ok=True)
            self.get_logger().debug(f'gallery rotate: deleted {oldest_path.name}')
        except Exception as e:
            self.get_logger().warn(f'rotate_oldest: error deleting {oldest_path}: {e}')
            return
        threading.Thread(
            target=self._gallery_remove, args=(path_str,), daemon=True).start()

    def _is_duplicate(self, emb: np.ndarray) -> bool:
        recent = self._recent_embeddings
        if not recent:
            return False
        mat  = np.stack(recent)
        sims = mat @ emb
        return bool(sims.max() > self._max_div_sim)

    def _get_person_dir(self, person_id: int, person_name: str | None) -> Path:
        safe_name  = (person_name or 'unknown').replace(' ', '_')
        person_dir = self._gallery_dir / 'persons' / f'{person_id}_{safe_name}'
        person_dir.mkdir(parents=True, exist_ok=True)
        return person_dir

    def _dest_path(self, person_id: int | None, person_name: str | None, filename: str) -> str:
        if person_id is None:
            return str(self._gallery_dir / '_pending' / filename)
        return str(self._get_person_dir(person_id, person_name) / filename)

    def _save_crop(self, frame: np.ndarray, bbox: list, padding: float,
                   dest: str, quality: int = 90) -> str | None:
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            pad_x = int((x2 - x1) * padding)
            pad_y = int((y2 - y1) * padding)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            cv2.imwrite(dest, crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return dest
        except Exception as e:
            self.get_logger().warn(f'Error saving crop: {e}')
            return None

    # ── Move pending → person dir after enrollment ───────────────────────

    def _flush_pending(self, person_id: int, person_name: str):
        with self._lock:
            pending = list(self._pending_photos)
            self._pending_photos = []

        if not pending:
            return

        person_dir   = self._get_person_dir(person_id, person_name)
        left_photos  = sorted(person_dir.glob('left_*.jpg'))
        right_photos = sorted(person_dir.glob('right_*.jpg'))
        total        = len(left_photos) + len(right_photos)
        available    = max(0, self._max_photos - total)
        if len(pending) > available:
            pending = pending[:available]

        moved = 0
        for src_path, embedding in pending:
            try:
                fname    = Path(src_path).name
                dst_path = str(person_dir / fname)
                shutil.move(src_path, dst_path)
                self._gallery_add(person_id, dst_path, embedding, 1.0, 'enroll')
                moved += 1
            except Exception as e:
                self.get_logger().warn(f'Error moving {src_path}: {e}')

        self.get_logger().info(
            f'flush_pending: {moved} photo(s) → persons/{person_id}_{person_name}')

        with self._lock:
            self._recent_embeddings = []

    def _clear_pending_dir(self):
        pending_dir = self._gallery_dir / '_pending'
        try:
            for f in pending_dir.glob('*.jpg'):
                f.unlink(missing_ok=True)
        except Exception:
            pass

    # ── /memory/query calls ─────────────────────────────────────────────

    def _call_memory(self, payload: dict, timeout: float = 5.0) -> dict | None:
        if not self._mem.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f'/memory/query unavailable (op={payload.get("op")})')
            return None
        req = MemoryQuery.Request()
        req.request_json = json.dumps(payload)
        future = self._mem.call_async(req)
        done   = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout=timeout):
            self.get_logger().warn(f'Timeout on /memory/query (op={payload.get("op")})')
            return None
        try:
            return json.loads(future.result().response_json)
        except Exception as e:
            self.get_logger().warn(f'Error parsing /memory/query response: {e}')
            return None

    def _gallery_add(self, person_id: int, photo_path: str,
                     embedding: list, quality: float, source: str):
        result = self._call_memory({
            'op':         'gallery_add',
            'person_id':  person_id,
            'photo_path': photo_path,
            'embedding':  embedding,
            'quality':    quality,
            'source':     source,
        })
        if result and not result.get('added'):
            self.get_logger().debug(
                f'gallery_add: skipped ({result.get("reason", "unknown")}) {photo_path}')

    def _gallery_remove(self, photo_path: str):
        result = self._call_memory({'op': 'gallery_remove', 'photo_path': photo_path})
        if result and result.get('removed'):
            self.get_logger().debug(f'gallery_remove: removed from DB {Path(photo_path).name}')

    def _rebuild_embedding(self, person_id: int):
        result = self._call_memory(
            {'op': 'gallery_rebuild_embedding', 'person_id': person_id},
            timeout=10.0)
        if result and result.get('rebuilt'):
            self.get_logger().info(
                f'Embedding recomputed: person_id={person_id} | '
                f'{result.get("gallery_count", 0)} gallery photo(s)')


def main():
    rclpy.init()
    node = FaceGalleryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
