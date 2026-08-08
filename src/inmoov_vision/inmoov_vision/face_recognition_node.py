#!/usr/bin/env python3
"""
face_recognition_node.py
========================
Распознаёт личность по трекам лиц с обоих глаз.

Режимы работы:
  left-primary  — левый глаз активен: его track_id канонический,
                  правый обогащает embed_buf совпадающего левого трека.
  right-only    — левый недоступен > STALE_SEC: правый становится primary,
                  публикует со своими track_id.

Переключение сопровождается сбросом кеша (чистый старт).

Подписки:
  /face/tracks/left   (String JSON)
  /face/tracks/right  (String JSON)

Публикует:
  /face/identity  (String JSON)
"""

import json
import threading
import time

import numpy as np

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, String
from inmoov_msgs.srv import MemoryQuery

_STALE_SEC = 2.0  # порог недоступности камеры (с)


def _bbox_area(bbox: list) -> float:
    if len(bbox) < 4:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _cosine_sim(a: list, b: list) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va / na, vb / nb))


class FaceRecognitionNode(LifecycleNode):
    def __init__(self):
        super().__init__('face_recognition_node')
        self._pub          = None
        self._lock         = threading.Lock()
        self._sleeping     = False
        self._cache        = {}
        self._active_side  = 'none'
        self._last_left_t  = self._last_right_t = 0.0
        self._left_tracks  = []

    def _dp(self, name, default=None):
        """Безопасный declare_parameter: игнорирует повторное объявление при re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('recognition_cooldown_sec', 2.0)
        self._dp('lock_after_known',         True)
        self._dp('lock_after_attempts',      3)
        self._dp('embed_buf_size',           5)
        self._dp('min_det_score',            0.65)
        self._dp('cross_eye_sim_thresh',     0.45)

        self._cooldown       = self.get_parameter('recognition_cooldown_sec').value
        self._lock_known     = self.get_parameter('lock_after_known').value
        self._lock_attempts  = self.get_parameter('lock_after_attempts').value
        self._embed_buf_size = self.get_parameter('embed_buf_size').value
        self._min_det_score  = self.get_parameter('min_det_score').value
        self._cross_sim_thr  = self.get_parameter('cross_eye_sim_thresh').value

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(String, '/face/tracks/left',  self._left_cb,  10)
        self.create_subscription(String, '/face/tracks/right', self._right_cb, 10)
        self.create_subscription(Bool, '/robot_sleep', self._sleep_cb, latched_qos)
        self._mem_client = self.create_client(MemoryQuery, '/memory/query')
        self._pub = self.create_lifecycle_publisher(String, '/face/identity', 10)
        self.get_logger().info(
            f'FaceRecognition configured (embed_buf={self._embed_buf_size})')
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

    def _sleep_cb(self, msg: Bool):
        self._sleeping = msg.data

    # ── Left callback (primary) ───────────────────────────────────────────

    def _left_cb(self, msg: String):
        if self._sleeping:
            return
        try:
            tracks = json.loads(msg.data).get('tracks', [])
        except Exception as e:
            self.get_logger().error(f'JSON ошибка left: {e}')
            return

        now = time.time()
        with self._lock:
            self._last_left_t = now
            self._left_tracks = tracks

            if self._active_side != 'left':
                self.get_logger().info('FaceRecognition: активен левый глаз')
                self._cache       = {}
                self._active_side = 'left'

            # Удаляем кеш исчезнувших левых треков
            current_ids = {t['track_id'] for t in tracks}
            for tid in list(self._cache):
                if tid not in current_ids:
                    del self._cache[tid]

        self._process_tracks(tracks, now)

    # ── Right callback (fallback / enrichment) ────────────────────────────

    def _right_cb(self, msg: String):
        if self._sleeping:
            return
        try:
            tracks = json.loads(msg.data).get('tracks', [])
        except Exception as e:
            self.get_logger().error(f'JSON ошибка right: {e}')
            return

        now = time.time()
        with self._lock:
            self._last_right_t  = now
            left_available = (now - self._last_left_t) < _STALE_SEC

        if left_available:
            self._enrich_from_right(tracks)
            return

        # Fallback: правый глаз становится primary
        with self._lock:
            if self._active_side != 'right':
                self.get_logger().warn(
                    'FaceRecognition: левый глаз недоступен → fallback на правый')
                self._cache       = {}
                self._active_side = 'right'
                self._left_tracks = []

            current_ids = {t['track_id'] for t in tracks}
            for tid in list(self._cache):
                if tid not in current_ids:
                    del self._cache[tid]

        self._process_tracks(tracks, now)

    # ── Обогащение кеша левых треков embedding'ами правого глаза ─────────

    def _enrich_from_right(self, right_tracks: list):
        if not right_tracks:
            return
        with self._lock:
            left_tracks = list(self._left_tracks)
        if not left_tracks:
            return

        r_primary = max(right_tracks, key=lambda t: _bbox_area(t.get('bbox', [])))
        l_primary = max(left_tracks,  key=lambda t: _bbox_area(t.get('bbox', [])))

        r_emb = r_primary.get('embedding', [])
        l_emb = l_primary.get('embedding', [])
        if not r_emb or not l_emb:
            return

        if _cosine_sim(r_emb, l_emb) < self._cross_sim_thr:
            return  # разные люди или низкое качество — не мёрджим

        tid = l_primary['track_id']
        with self._lock:
            cached = self._cache.get(tid)
            if cached is None or cached.get('locked'):
                return
            buf = cached['embed_buf']
            buf.append(r_emb)
            if len(buf) > self._embed_buf_size * 2:
                buf.pop(0)

    # ── Унифицированная обработка треков ─────────────────────────────────

    def _process_tracks(self, tracks: list, now: float):
        for track in tracks:
            tid       = track['track_id']
            embedding = track.get('embedding', [])
            det_score = track.get('det_score', 1.0)

            if not embedding:
                continue

            with self._lock:
                cached = self._cache.setdefault(tid, {'embed_buf': []})

                if cached.get('locked'):
                    self._publish(tid, cached)
                    continue

                if det_score < self._min_det_score:
                    continue

                buf = cached['embed_buf']
                buf.append(embedding)
                if len(buf) > self._embed_buf_size * 2:
                    buf.pop(0)

                if len(buf) < self._embed_buf_size:
                    continue

                if (now - cached.get('last_called', 0)) < self._cooldown:
                    if cached.get('is_known') is not None:
                        self._publish(tid, cached)
                    continue

                avg_embedding = self._average_embeddings(buf)
                cached['last_called'] = now

            threading.Thread(
                target=self._recognize,
                args=(tid, avg_embedding),
                daemon=True,
            ).start()

    # ── Утилиты ───────────────────────────────────────────────────────────

    @staticmethod
    def _average_embeddings(embeddings: list) -> list:
        mat  = np.array(embeddings, dtype=np.float32)
        avg  = mat.mean(axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg /= norm
        return avg.tolist()

    # ── Запрос к БД ───────────────────────────────────────────────────────

    def _recognize(self, track_id: int, embedding: list):
        if not self._mem_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('/memory/query недоступен')
            return

        req = MemoryQuery.Request()
        req.request_json = json.dumps({
            'op':        'lookup_person',
            'embedding': embedding,
        })

        future = self._mem_client.call_async(req)
        done_event = threading.Event()
        future.add_done_callback(lambda _: done_event.set())
        if not done_event.wait(timeout=5.0):
            self.get_logger().warn('Таймаут /memory/query')
            return

        try:
            resp   = future.result()
            result = json.loads(resp.response_json)
        except Exception as e:
            self.get_logger().error(f'Ошибка сервиса: {e}')
            return

        is_known = result.get('person_id') is not None
        identity = {
            'person_id':           result.get('person_id'),
            'name':                result.get('name') or result.get('best_candidate_name'),
            'similarity':          result.get('similarity', 0.0),
            'is_known':            is_known,
            'confidence':          result.get('confidence', 'unknown'),
            'best_candidate_id':   result.get('best_candidate_id'),
            'best_candidate_name': result.get('best_candidate_name'),
        }

        with self._lock:
            entry = self._cache.setdefault(track_id, {'embed_buf': []})
            entry.update(identity)
            entry['attempts'] = entry.get('attempts', 0) + 1

            if is_known and self._lock_known:
                entry['locked'] = True
                self.get_logger().info(
                    f'Трек {track_id} → {identity["name"]} '
                    f'(sim={identity["similarity"]:.3f}, side={self._active_side}) — LOCKED')
            elif entry['attempts'] >= self._lock_attempts:
                entry['locked'] = True
                self.get_logger().info(
                    f'Трек {track_id} → неизвестный после {entry["attempts"]} попыток — LOCKED')

            identity['locked'] = entry.get('locked', False)

        self._publish(track_id, identity)

    # ── Публикация ────────────────────────────────────────────────────────

    def _publish(self, track_id: int, identity: dict):
        msg = String()
        msg.data = json.dumps({
            'track_id':            track_id,
            'person_id':           identity.get('person_id'),
            'name':                identity.get('name'),
            'similarity':          identity.get('similarity', 0.0),
            'is_known':            identity.get('is_known', False),
            'confidence':          identity.get('confidence', 'unknown'),
            'best_candidate_id':   identity.get('best_candidate_id'),
            'best_candidate_name': identity.get('best_candidate_name'),
            'locked':              identity.get('locked', False),
        })
        self._pub.publish(msg)




def main():
    rclpy.init()
    node = FaceRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
