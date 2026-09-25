#!/usr/bin/env python3
"""
memory_node.py — Memory Manager Node (v2)
==========================================
Single ROS2 node that manages all of the robot's memory:

  SOCIAL MEMORY (SQLite, ~/inmoov_memory.db):
    persons, person_gallery, person_notes, robot_knowledge
    — face/voice recognition, notes about people

  WORKING MEMORY (RAM, WorkingMemory):
    time, location, mode, people nearby — always in the LLM context

  EPISODIC MEMORY (SQLite, ~/inmoov_episodic.db):
    short-term memories, dialogues — last 5 included in prompting

  SEMANTIC MEMORY (SQLite + ChromaDB, ~/inmoov_semantic.db):
    long-term facts, preferences — only via tool call

ROS2 interfaces:
  /memory/query   (srv MemoryQuery) — all operations (ops below)
  /memory/context (pub String, 30s) — working memory + episodes for the LLM prompt
  /social_context (sub String)      — working memory updates from identity_manager
  /conversation_end (sub String)    — after a dialogue: summary + fact extraction
  /robot_sleep    (sub Bool, latched)

/memory/query operations:
  Social memory (original):
    lookup_person, save_person, update_embedding, get_context,
    update_seen, set_note, get_knowledge, set_knowledge,
    gallery_add, gallery_remove, gallery_rebuild_embedding,
    gallery_list, merge_persons, reload_gallery,
    get_voice_embedding, save_voice_embedding, update_voice_embedding, lookup_by_voice

  New (3-layer memory):
    search_semantic        — semantic search in long-term memory
    save_semantic_fact     — save a fact to long-term memory
    search_episodes        — search episodic memory
    get_recent_episodes    — last N episodes (for an LLM tool)
    save_episode           — save an episode manually
    get_working_memory     — current working memory state
    update_working_memory  — update working memory
    get_memory_context     — assemble text context for the LLM

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import glob
import json
import logging
import os
import shutil
import sqlite3
import threading
import time
from datetime import datetime

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String, Bool
from inmoov_msgs.srv import MemoryQuery

from inmoov_memory.memory_manager import MemoryManager, strip_code_fence
from inmoov_memory.reminder_db import ReminderDB
from inmoov_memory.sqlite_util import connect, session

logger = logging.getLogger(__name__)

# Background work (episode summary, reminder extraction, Chroma reconcile) gets
# this long to finish before the DBs are closed on cleanup/shutdown.
_BG_JOIN_TIMEOUT_SEC = 20.0


def as_embedding(raw, dim: int) -> np.ndarray:
    """Validates a JSON embedding (length, finite, non-zero) and L2-normalizes it.

    Raises ValueError — the /memory/query dispatcher turns it into an error reply,
    so a malformed vector never reaches the DB or the caches.
    """
    emb = np.asarray(raw, dtype=np.float32)
    if emb.ndim != 1 or emb.shape[0] != dim:
        raise ValueError(f'embedding must have {dim} values, got shape {emb.shape}')
    if not np.all(np.isfinite(emb)):
        raise ValueError('embedding contains NaN/Inf')
    norm = float(np.linalg.norm(emb))
    if norm < 1e-8:
        raise ValueError('embedding has zero norm')
    return emb / norm


class MemoryNode(LifecycleNode):
    def __init__(self):
        super().__init__('memory_node')
        # Instance variables — populated in on_configure
        self._db              = None
        self._lock            = threading.Lock()
        self._gallery_cache: dict[int, np.ndarray] = {}
        self._voice_cache: dict[int, np.ndarray]   = {}
        self._voice_emb_dim   = 192   # ECAPA-TDNN
        self._face_emb_dim    = 512   # InsightFace buffalo_l
        self._voice_gallery_cache: dict[int, list] = {}
        self._SV_GALLERY_MAX  = 10
        self._SV_REFRESH_DAYS = 7
        self._reminder_db     = None
        self._mm              = None
        self._ctx_pub         = None
        self._tg_push_pub     = None
        self._timers          = []
        self.sim_threshold       = 0.55
        self.uncertain_threshold = 0.40
        self._tg_reminder_person_id = 5
        self._reminder_default_time = '07:00'
        self._sleeping        = False   # /robot_sleep latched state
        # Telegram reminders sent but not yet ACKed: push id → monotonic send time
        self._tg_inflight: dict[str, float] = {}
        self._bg_threads: list[threading.Thread] = []
        self._bg_lock         = threading.Lock()

    # ── Lifecycle callbacks ────────────────────────────────────────────────

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores repeated declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        # ── Parameters ─────────────────────────────────────────────────────
        self._dp('db_path',                    os.path.expanduser('~/inmoov_memory.db'))
        self._dp('episodic_db_path',           os.path.expanduser('~/inmoov_episodic.db'))
        self._dp('semantic_db_path',           os.path.expanduser('~/inmoov_semantic.db'))
        self._dp('chroma_path',                os.path.expanduser('~/inmoov_chroma'))
        self._dp('reminder_db_path',           os.path.expanduser('~/inmoov_reminders.db'))
        self._dp('gallery_dir',                os.path.expanduser('~/inmoov_faces'))  # face_gallery_node's gallery_dir
        self._dp('llm_url',   'http://192.168.10.118:18020/v1/chat/completions')
        self._dp('llm_model', 'qwen3.8-27b')
        self._dp('bearer_token', '')
        self._dp('similarity_threshold',       0.55)
        self._dp('uncertain_threshold',        0.40)
        self._dp('context_publish_rate',       30.0)
        self._dp('telegram_reminder_person_id', 5)
        self._dp('telegram_reminder_check_sec', 60.0)
        self._dp('reminder_default_time',      '07:00')
        self._dp('episodic_retention_days',    7)
        self._dp('episodic_cleanup_max_importance', 0.5)
        self._dp('episodic_cleanup_interval_sec',   21600.0)

        db_path      = self.get_parameter('db_path').value
        episodic_db  = self.get_parameter('episodic_db_path').value
        self._episodic_db_path = episodic_db
        self._gallery_dir      = self.get_parameter('gallery_dir').value
        semantic_db  = self.get_parameter('semantic_db_path').value
        chroma_path  = self.get_parameter('chroma_path').value
        reminder_db  = self.get_parameter('reminder_db_path').value
        llm_url      = self.get_parameter('llm_url').value
        llm_model    = self.get_parameter('llm_model').value
        bearer_token = self.get_parameter('bearer_token').value
        self.sim_threshold          = self.get_parameter('similarity_threshold').value
        self.uncertain_threshold    = self.get_parameter('uncertain_threshold').value
        self._tg_reminder_person_id = self.get_parameter('telegram_reminder_person_id').value
        self._reminder_default_time = self.get_parameter('reminder_default_time').value

        # ── Social memory (SQLite) ────────────────────────────────────────
        self._db = connect(db_path, check_same_thread=False)
        self._init_social_db()
        self._load_gallery_cache()
        self._load_voice_cache()
        self._load_voice_gallery_cache()

        # ── Reminders ──────────────────────────────────────────────────────
        self._reminder_db = ReminderDB(reminder_db)
        self.get_logger().info(f'ReminderDB: {reminder_db}')

        # ── 3-layer memory ─────────────────────────────────────────────────
        self._mm = MemoryManager(
            db_path=episodic_db,
            semantic_db_path=semantic_db,
            chroma_path=chroma_path,
            llm_url=llm_url,
            llm_model=llm_model,
            bearer_token=bearer_token,
        )
        self.get_logger().info(
            f'MemoryManager: episodic={episodic_db} semantic={semantic_db}')
        self._spawn_bg(self._reconcile_chroma)

        # ── Service — created in configure, available right after activation ─
        self.create_service(MemoryQuery, '/memory/query', self._handle)

        # ── Subscriptions ──────────────────────────────────────────────────
        self.create_subscription(String, '/social_context',   self._social_context_cb, 10)
        self.create_subscription(String, '/conversation_end', self._conversation_end_cb, 10)
        _latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Bool, '/robot_sleep', self._robot_sleep_cb, _latched)
        self.create_subscription(String, '/telegram_push_ack', self._tg_push_ack_cb, 10)

        # ── Lifecycle publishers (silent until on_activate is called) ──────
        self._ctx_pub     = self.create_lifecycle_publisher(String, '/memory/context', 10)
        self._tg_push_pub = self.create_lifecycle_publisher(String, '/telegram_push', 10)

        self.get_logger().info(
            f'MemoryNode configured. DB: {db_path} | threshold: {self.sim_threshold}')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        # MUST activate every lifecycle publisher, passing state
        self._ctx_pub.on_activate(state)
        self._tg_push_pub.on_activate(state)

        ctx_rate     = self.get_parameter('context_publish_rate').value
        tg_check_sec = self.get_parameter('telegram_reminder_check_sec').value
        cleanup_sec  = self.get_parameter('episodic_cleanup_interval_sec').value

        self._timers.append(self.create_timer(ctx_rate, self._publish_memory_context))
        if self._tg_reminder_person_id > 0:
            self._timers.append(
                self.create_timer(tg_check_sec, self._send_due_reminders_to_telegram))
        if cleanup_sec > 0:
            self._timers.append(self.create_timer(cleanup_sec, self._cleanup_episodic))
            self._cleanup_episodic()

        # Publish context immediately the first time
        self._publish_memory_context()

        self.get_logger().info(
            f'MemoryNode active. Context every {ctx_rate}s | '
            f'telegram reminders person_id={self._tg_reminder_person_id} '
            f'every {tg_check_sec:.0f}s')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        for t in self._timers:
            self.destroy_timer(t)
        self._timers.clear()
        self._ctx_pub.on_deactivate(state)
        self._tg_push_pub.on_deactivate(state)
        self.get_logger().info('MemoryNode deactivated')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self._join_bg()
        self._close_db()
        self._gallery_cache.clear()
        self._voice_cache.clear()
        self._voice_gallery_cache.clear()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self._join_bg()
        self._close_db()
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        self._close_db()
        return TransitionCallbackReturn.SUCCESS

    # ── Background threads ─────────────────────────────────────────────

    def _spawn_bg(self, target, *args):
        """Starts a tracked daemon thread; _join_bg() waits for it before the DBs close."""
        t = threading.Thread(target=target, args=args, daemon=True)
        with self._bg_lock:
            self._bg_threads = [x for x in self._bg_threads if x.is_alive()]
            self._bg_threads.append(t)
        t.start()

    def _join_bg(self, timeout: float = _BG_JOIN_TIMEOUT_SEC):
        with self._bg_lock:
            threads = [t for t in self._bg_threads if t.is_alive()]
        if threads:
            self.get_logger().info(
                f'Waiting for {len(threads)} background memory task(s) (≤{timeout:.0f}s)...')
        deadline = time.monotonic() + timeout
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))
        left = sum(t.is_alive() for t in threads)
        if left:
            self.get_logger().warn(f'{left} background memory task(s) still running — abandoned')

    def _reconcile_chroma(self):
        try:
            res = self._mm.semantic.reconcile_chroma()
            if res['upserted'] or res['deleted']:
                self.get_logger().info(
                    f'Chroma reconcile: re-indexed {res["upserted"]}, '
                    f'removed {res["deleted"]} stale')
        except Exception as e:
            self.get_logger().warn(f'Chroma reconcile failed: {e}')

    def _close_db(self):
        try:
            if self._db:
                with self._lock:
                    self._db.close()
                self._db = None
        except Exception:
            pass
        try:
            if self._reminder_db:
                self._reminder_db.close()
                self._reminder_db = None
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════════════
    # Social DB initialization (formerly _init_db)
    # ════════════════════════════════════════════════════════════════════

    def _init_social_db(self):
        with self._lock:
            self._db.executescript('''
                CREATE TABLE IF NOT EXISTS persons (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    name            TEXT    NOT NULL,
                    embedding       BLOB    NOT NULL,
                    first_seen      TEXT    NOT NULL,
                    last_seen       TEXT    NOT NULL,
                    meet_count      INTEGER DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS person_gallery (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id   INTEGER NOT NULL,
                    photo_path  TEXT    NOT NULL,
                    embedding   BLOB    NOT NULL,
                    quality     REAL    DEFAULT 1.0,
                    source      TEXT    DEFAULT 'auto',
                    created_at  TEXT    NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES persons(id)
                );
                CREATE TABLE IF NOT EXISTS person_notes (
                    person_id   INTEGER NOT NULL,
                    key         TEXT    NOT NULL,
                    value       TEXT    NOT NULL,
                    PRIMARY KEY (person_id, key),
                    FOREIGN KEY (person_id) REFERENCES persons(id)
                );
                CREATE TABLE IF NOT EXISTS robot_knowledge (
                    key     TEXT PRIMARY KEY,
                    value   TEXT NOT NULL,
                    updated TEXT NOT NULL
                );
            ''')
            self._db.commit()
            try:
                self._db.execute('ALTER TABLE persons ADD COLUMN voice_embedding BLOB')
                self._db.commit()
                self.get_logger().info('DB migration: added column voice_embedding')
            except sqlite3.OperationalError:
                pass
            try:
                self._db.execute('ALTER TABLE persons ADD COLUMN telegram_id INTEGER DEFAULT NULL')
                self._db.commit()
                self.get_logger().info('DB migration: added column telegram_id')
            except sqlite3.OperationalError:
                pass
            # Voice gallery (up to 10 embeddings per person)
            self._db.execute('''
                CREATE TABLE IF NOT EXISTS voice_gallery (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id   INTEGER NOT NULL,
                    embedding   BLOB    NOT NULL,
                    recorded_at REAL    NOT NULL DEFAULT 0,
                    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
                )
            ''')
            self._db.commit()
            # Migration: move the single voice_embedding into voice_gallery (only if the gallery is empty)
            rows = self._db.execute(
                'SELECT id, voice_embedding FROM persons WHERE voice_embedding IS NOT NULL'
            ).fetchall()
            for pid, blob in rows:
                if blob and not self._db.execute(
                        'SELECT 1 FROM voice_gallery WHERE person_id=?', (pid,)).fetchone():
                    self._db.execute(
                        'INSERT INTO voice_gallery (person_id, embedding, recorded_at) VALUES (?,?,0)',
                        (pid, blob))
            self._db.commit()

    # ════════════════════════════════════════════════════════════════════
    # Gallery and voice caches
    # ════════════════════════════════════════════════════════════════════

    def _load_gallery_cache(self):
        with self._lock:
            rows = self._db.execute(
                'SELECT person_id, embedding FROM person_gallery'
            ).fetchall()
        by_person: dict[int, list] = {}
        for pid, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32).copy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            by_person.setdefault(pid, []).append(emb)
        self._gallery_cache = {
            pid: np.stack(embs) for pid, embs in by_person.items()
        }
        total = sum(len(v) for v in self._gallery_cache.values())
        self.get_logger().info(
            f'Gallery: {len(self._gallery_cache)} people, {total} photos')

    def _load_voice_cache(self):
        try:
            with self._lock:
                rows = self._db.execute(
                    'SELECT id, voice_embedding FROM persons WHERE voice_embedding IS NOT NULL'
                ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        self._voice_cache = {}
        skipped = 0
        for pid, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32).copy()
            if emb.shape[0] != self._voice_emb_dim:
                skipped += 1
                continue
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            self._voice_cache[pid] = emb
        self.get_logger().info(
            f'Voice embeddings: {len(self._voice_cache)} people'
            + (f' ({skipped} skipped)' if skipped else ''))

    def _load_voice_gallery_cache(self):
        try:
            with self._lock:
                rows = self._db.execute(
                    'SELECT person_id, embedding, recorded_at FROM voice_gallery ORDER BY recorded_at ASC'
                ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        self._voice_gallery_cache = {}
        for pid, blob, ts in rows:
            emb = np.frombuffer(blob, dtype=np.float32).copy()
            if emb.shape[0] != self._voice_emb_dim:
                continue
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            self._voice_gallery_cache.setdefault(pid, []).append(
                {'emb': emb, 'recorded_at': float(ts)}
            )
        total = sum(len(v) for v in self._voice_gallery_cache.values())
        self.get_logger().info(
            f'Voice gallery: {len(self._voice_gallery_cache)} people, {total} entries')

    # ════════════════════════════════════════════════════════════════════
    # ROS2 callbacks — new
    # ════════════════════════════════════════════════════════════════════

    def _social_context_cb(self, msg: String):
        """Updates working memory from identity_manager's social_context."""
        try:
            ctx = json.loads(msg.data)
            present = ctx.get('person_present', False)
            name    = ctx.get('name', '')

            # List of people nearby
            people = [name] if present and name else []
            self._mm.working.update_environment(people=people)

            # Who the robot is looking at
            self._mm.working.update_robot_state(
                facing=name if present and name else None,
            )

            # Operating mode
            state_raw = ctx.get('state', '')
            mode_map = {
                'IDLE':         'idle',
                'INTERACTING':  'conversation',
                'INTRODUCING':  'conversation',
            }
            mode = mode_map.get(str(state_raw), 'idle')
            if self._sleeping:
                mode = 'sleep'   # social_context keeps coming while asleep — don't overwrite
            self._mm.working.update_robot_state(mode=mode)

        except Exception as e:
            self.get_logger().warn(f'social_context_cb: {e}')

    def _conversation_end_cb(self, msg: String):
        """Handles end of dialogue: summary + fact extraction in the background."""
        try:
            data         = json.loads(msg.data)
            transcript   = data.get('transcript', '')
            participants = data.get('participants', [])
            if not transcript.strip():
                return
            self._spawn_bg(self._run_after_conversation, transcript, participants)
        except Exception as e:
            self.get_logger().warn(f'conversation_end_cb: {e}')

    def _run_after_conversation(self, transcript: str, participants: list):
        try:
            self._mm.after_conversation(transcript, participants=participants)
            self.get_logger().info(
                f'Episode saved: {len(transcript)} chars, '
                f'participants: {participants}')
        except Exception as e:
            self.get_logger().error(f'after_conversation error: {e}')
        finally:
            self._publish_memory_context()
        # Auto-extract reminders from the dialogue (same background thread)
        self._extract_reminders(transcript, participants)

    # Keywords whose presence triggers an LLM call to extract reminders
    _REMINDER_KEYWORDS = [
        'рождени', 'выступлени', 'концерт', 'экзамен', 'спектакл',
        'годовщин', 'свидани', 'соревновани', 'дедлайн', 'визит',
        'операци', 'приём врач', 'праздни', 'юбилей',
        'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
        'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря',
    ]

    def _extract_reminders(self, transcript: str, participants: list):
        """Auto-extracts reminders from the dialogue via the LLM.
        Only runs if the transcript contains keywords associated with dates."""
        if not participants:
            return
        person_name = participants[0]
        transcript_lower = transcript.lower()
        if not any(kw in transcript_lower for kw in self._REMINDER_KEYWORDS):
            return

        with self._lock:
            if self._db is None:   # DB closed while we were queued
                return
            row = self._db.execute(
                'SELECT id FROM persons WHERE name=? LIMIT 1', (person_name,)
            ).fetchone()
        if not row:
            return
        person_id = row[0]

        today = datetime.now().date().isoformat()
        system = (
            'Ты — система извлечения напоминаний. '
            'Отвечай ТОЛЬКО валидным JSON-массивом, без пояснений и markdown.'
        )
        prompt = (
            f'Проанализируй диалог и найди события с конкретными будущими датами, '
            f'о которых нужно напомнить заранее.\n'
            f'Сегодня: {today}\n'
            f'Участник: {person_name}\n\n'
            f'Диалог:\n{transcript}\n\n'
            f'Верни ТОЛЬКО JSON-массив (пустой [] если событий нет):\n'
            f'[{{"event_date":"YYYY-MM-DD","remind_date":"YYYY-MM-DD",'
            f'"message":"Привет, {person_name}! Ты помнишь, что [описание события]?"}}]\n'
            f'Правила:\n'
            f'- Только конкретные БУДУЩИЕ даты (позже {today})\n'
            f'- remind_date = за 1 день до события (или в день события если дата одна)\n'
            f'- Только важные события: дни рождения, концерты, выступления, '
            f'экзамены, встречи, операции, юбилеи. Если нет важных событий, верни пустой массив\n'
            f'- Отвечай ТОЛЬКО JSON-массивом, никаких пояснений'
        )
        try:
            raw = self._mm._call_llm(prompt, system=system)
            raw = strip_code_fence(raw)
            items = json.loads(raw)
            count = 0
            for item in items:
                if not isinstance(item, dict):
                    continue
                remind_date = (item.get('remind_date') or '').strip() or None
                message     = (item.get('message') or '').strip()
                if not message:
                    continue
                if remind_date and remind_date <= today:
                    continue  # past date — skip
                rid = self._reminder_db.add_reminder(
                    person_id=person_id,
                    person_name=person_name,
                    message=message,
                    trigger_date=remind_date,
                    source='auto',
                )
                count += 1
                self.get_logger().info(
                    f'Auto-reminder: id={rid} date={remind_date or "next_meeting"} '
                    f'"{message[:60]}"')
            if count:
                self.get_logger().info(
                    f'Auto-extracted {count} reminders for {person_name}')
        except Exception as e:
            self.get_logger().warn(f'_extract_reminders: {e}')

    def _robot_sleep_cb(self, msg: Bool):
        self._sleeping = bool(msg.data)
        mode = 'sleep' if msg.data else 'idle'
        self._mm.working.update_robot_state(mode=mode)
        if msg.data:
            self._mm.working.update_environment(people=[])
            self._mm.working.update_robot_state(facing=None)

    def _cleanup_episodic(self):
        """Timer: purges old low-importance episodes (the episodic sliding window)."""
        days    = self.get_parameter('episodic_retention_days').value
        max_imp = self.get_parameter('episodic_cleanup_max_importance').value
        try:
            deleted = self._mm.episodic.cleanup(older_than_days=days, max_importance=max_imp)
            if deleted:
                self.get_logger().info(
                    f'Episodic cleanup: deleted {deleted} episodes '
                    f'older than {days}d with importance <= {max_imp}')
        except Exception as e:
            self.get_logger().warn(f'_cleanup_episodic: {e}')

    # ── /memory/context publisher ──────────────────────────────────────

    def _send_due_reminders_to_telegram(self):
        """Timer: checks overdue reminders (delivered=0) and sends them to Telegram.

        Marked delivered=1 only when telegram_bridge ACKs the push
        (_tg_push_ack_cb). While an ACK is pending the reminder isn't re-sent;
        a failed or unanswered push (bridge down, network) is retried on a later
        tick instead of being lost.
        """
        import datetime as _dt
        import html as _html
        now_dt   = _dt.datetime.now()
        today    = now_dt.date().isoformat()
        now_time = now_dt.strftime('%H:%M')
        try:
            reminders = self._reminder_db.get_due(
                self._tg_reminder_person_id, today,
                now_time=now_time,
                default_time=self._reminder_default_time,
            )
        except Exception as e:
            self.get_logger().warn(f'TG reminder timer: get_due error: {e}')
            return

        now = time.monotonic()
        for r in reminders:
            push_id = f'reminder:{r["id"]}'
            sent_at = self._tg_inflight.get(push_id)
            if sent_at is not None and now - sent_at < self._TG_ACK_TIMEOUT_SEC:
                continue   # waiting for the bridge's ACK
            text = f'⏰ <b>Напоминание:</b> {_html.escape(r["message"])}'
            try:
                msg = String()
                msg.data = json.dumps(
                    {'text': text, 'parse_mode': 'HTML', 'id': push_id}, ensure_ascii=False)
                self._tg_push_pub.publish(msg)
                self._tg_inflight[push_id] = now
                self.get_logger().info(
                    f'TG reminder: id={r["id"]} sent'
                    + (' (retry — no ACK)' if sent_at is not None else '')
                    + ', waiting for ACK')
            except Exception as e:
                self.get_logger().warn(f'TG reminder: send error id={r["id"]}: {e}')

    _TG_ACK_TIMEOUT_SEC = 300.0   # no ACK this long → send again

    def _tg_push_ack_cb(self, msg: String):
        """telegram_bridge confirmed (or refused) a push we sent with an id."""
        try:
            ack = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        push_id = str(ack.get('id', ''))
        if not push_id.startswith('reminder:') or push_id not in self._tg_inflight:
            return
        self._tg_inflight.pop(push_id, None)
        rid = int(push_id.split(':', 1)[1])
        if ack.get('ok'):
            try:
                self._reminder_db.mark_delivered(rid)
                self.get_logger().info(f'TG reminder: id={rid} delivered ✓')
            except Exception as e:
                self.get_logger().warn(f'TG reminder: mark_delivered id={rid}: {e}')
        else:
            self.get_logger().warn(
                f'TG reminder: id={rid} not delivered ({ack.get("error", "?")}) — will retry')

    def _publish_memory_context(self):
        """Publishes working memory + recent episodes to /memory/context."""
        try:
            working_text = self._mm.working.to_text()
            recent_text  = self._mm.episodic.get_recent_text(limit=5)

            parts = ['== Текущий момент ==', working_text]
            if recent_text:
                parts += ['', '== Последние события ==', recent_text]

            msg = String()
            msg.data = '\n'.join(parts)
            self._ctx_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'publish_memory_context: {e}')

    # ════════════════════════════════════════════════════════════════════
    # /memory/query dispatcher
    # ════════════════════════════════════════════════════════════════════

    def _handle(self, request, response):
        try:
            req = json.loads(request.request_json)
            op  = req.get('op', '')
            ops = {
                # ── Social memory (original) ──────────────────────────
                'lookup_person':              self._lookup_person,
                'save_person':                self._save_person,
                'update_embedding':           self._update_embedding,
                'get_context':                self._get_context,
                'update_seen':                self._update_seen,
                'set_note':                   self._set_note,
                'get_knowledge':              self._get_knowledge,
                'set_knowledge':              self._set_knowledge,
                'gallery_add':                self._gallery_add,
                'gallery_remove':             self._gallery_remove,
                'gallery_rebuild_embedding':  self._gallery_rebuild_embedding,
                'gallery_list':               self._gallery_list,
                'merge_persons':              self._merge_persons,
                'lookup_by_name':             self._lookup_by_name,
                'verify_person_claim':        self._verify_person_claim,
                'reload_gallery':             self._reload_gallery,
                'get_voice_embedding':        self._get_voice_embedding,
                'save_voice_embedding':       self._save_voice_embedding,
                'update_voice_embedding':     self._update_voice_embedding,
                'lookup_by_voice':            self._lookup_by_voice,
                'get_voice_gallery':          self._get_voice_gallery,
                'add_voice_to_gallery':       self._add_voice_to_gallery,
                # ── 3-layer memory (new) ────────────────────────────────
                'search_semantic':            self._op_search_semantic,
                'save_semantic_fact':         self._op_save_semantic_fact,
                'search_episodes':            self._op_search_episodes,
                'get_recent_episodes':        self._op_get_recent_episodes,
                'save_episode':               self._op_save_episode,
                'get_working_memory':         self._op_get_working_memory,
                'update_working_memory':      self._op_update_working_memory,
                'get_memory_context':         self._op_get_memory_context,
                # ── Reminders ────────────────────────────────────────────
                'add_reminder':              self._op_add_reminder,
                'get_due_reminders':         self._op_get_due_reminders,
                'mark_reminder_delivered':   self._op_mark_reminder_delivered,
                'delete_reminder':           self._op_delete_reminder,
                'confirm_reminders':         self._op_confirm_reminders,
                'list_reminders':            self._op_list_reminders,
            }
            fn = ops.get(op)
            if fn is None:
                result = {'error': f'Unknown op: {op}'}
            else:
                result = fn(req)
        except Exception as e:
            self.get_logger().error(f'Memory error [{op}]: {e}')
            result = {'error': str(e)}

        response.response_json = json.dumps(result, ensure_ascii=False)
        response.success       = 'error' not in result
        return response

    # ════════════════════════════════════════════════════════════════════
    # New operations — 3-layer memory
    # ════════════════════════════════════════════════════════════════════

    def _op_search_semantic(self, req: dict) -> dict:
        query    = req.get('query', '')
        category = req.get('category') or None
        limit    = int(req.get('limit', 5))
        results  = self._mm.semantic.search(query, category=category, limit=limit)
        return {'facts': results}

    def _op_save_semantic_fact(self, req: dict) -> dict:
        fact_id = self._mm.semantic.save_fact(
            subject    = req.get('subject', ''),
            predicate  = req.get('predicate', ''),
            value      = req.get('value', ''),
            category   = req.get('category', 'preference'),
            confidence = float(req.get('confidence', 1.0)),
            source     = req.get('source', 'conversation'),
        )
        return {'fact_id': fact_id, 'saved': True}

    def _op_search_episodes(self, req: dict) -> dict:
        results = self._mm.episodic.search(
            keyword = req.get('keyword', ''),
            date    = req.get('date'),
            limit   = int(req.get('limit', 10)),
        )
        return {'episodes': results}

    def _op_get_recent_episodes(self, req: dict) -> dict:
        results = self._mm.episodic.get_recent(
            limit = int(req.get('limit', 5)),
            date  = req.get('date'),
        )
        return {'episodes': results}

    def _op_save_episode(self, req: dict) -> dict:
        ep_id = self._mm.episodic.save(
            summary      = req.get('summary', ''),
            raw_text     = req.get('raw_text', ''),
            participants = req.get('participants', []),
            location     = req.get('location', self._mm.working.data['location']['room']),
            importance   = float(req.get('importance', 0.3)),
            ep_type      = req.get('type', 'conversation'),
            emotion_tag  = req.get('emotion_tag', 'neutral'),
        )
        return {'episode_id': ep_id, 'saved': True}

    def _op_get_working_memory(self, req: dict) -> dict:
        self._mm.working.refresh_time()
        return {'working_memory': self._mm.working.data}

    def _op_update_working_memory(self, req: dict) -> dict:
        if 'location' in req:
            loc = req['location']
            self._mm.working.update_location(
                room        = loc.get('room', ''),
                landmark    = loc.get('landmark', ''),
                coordinates = loc.get('coordinates'),
            )
        if 'robot_state' in req:
            s = req['robot_state']
            self._mm.working.update_robot_state(
                mode         = s.get('mode'),
                battery      = s.get('battery'),
                current_task = s.get('current_task'),
                facing       = s.get('facing'),
            )
        if 'environment' in req:
            e = req['environment']
            self._mm.working.update_environment(
                people   = e.get('people'),
                noise    = e.get('noise'),
                lighting = e.get('lighting'),
            )
        return {'updated': True}

    def _op_get_memory_context(self, req: dict) -> dict:
        """Returns the text context to insert into the LLM system prompt."""
        working_text = self._mm.working.to_text()
        recent_text  = self._mm.episodic.get_recent_text(limit=int(req.get('limit', 5)))
        parts = ['== Текущий момент ==', working_text]
        if recent_text:
            parts += ['', '== Последние события ==', recent_text]
        return {'context': '\n'.join(parts)}

    # ════════════════════════════════════════════════════════════════════
    # Reminders
    # ════════════════════════════════════════════════════════════════════

    def _op_add_reminder(self, req: dict) -> dict:
        person_id    = req.get('person_id')
        person_name  = req.get('person_name', '')
        message      = req.get('message', '')
        trigger_date = req.get('trigger_date') or None
        trigger_time = req.get('trigger_time') or None
        source       = req.get('source', 'manual')
        if person_id is None or not message:
            return {'error': 'person_id и message обязательны'}
        rid = self._reminder_db.add_reminder(
            person_id=int(person_id),
            person_name=person_name,
            message=message,
            trigger_date=trigger_date,
            trigger_time=trigger_time,
            source=source,
        )
        self.get_logger().info(
            f'Reminder created: id={rid} person={person_name} '
            f'date={trigger_date or "next_meeting"} time={trigger_time or "default"} source={source}')
        return {'reminder_id': rid, 'saved': True}

    def _op_get_due_reminders(self, req: dict) -> dict:
        person_id    = req.get('person_id')
        today        = req.get('today', datetime.now().date().isoformat())
        now_time     = req.get('now_time')
        default_time = req.get('default_time', self._reminder_default_time)
        if person_id is None:
            return {'error': 'person_id обязателен'}
        reminders = self._reminder_db.get_due(
            int(person_id), today,
            now_time=now_time,
            default_time=default_time,
        )
        return {'reminders': reminders}

    def _op_mark_reminder_delivered(self, req: dict) -> dict:
        reminder_id = req.get('reminder_id')
        if reminder_id is None:
            return {'error': 'reminder_id обязателен'}
        self._reminder_db.mark_delivered(int(reminder_id))
        return {'marked': True, 'reminder_id': reminder_id}

    def _op_delete_reminder(self, req: dict) -> dict:
        reminder_id = req.get('reminder_id')
        if reminder_id is None:
            return {'error': 'reminder_id обязателен'}
        self._reminder_db.delete_reminder(int(reminder_id))
        return {'deleted': True, 'reminder_id': reminder_id}

    def _op_confirm_reminders(self, req: dict) -> dict:
        """Deletes all shown (delivered=1) reminders — the user has confirmed them."""
        person_id = req.get('person_id')
        if person_id is None:
            return {'error': 'person_id обязателен'}
        count = self._reminder_db.confirm_all_delivered(int(person_id))
        self.get_logger().info(
            f'Reminders confirmed: person_id={person_id}, deleted={count}')
        return {'confirmed': count}

    def _op_list_reminders(self, req: dict) -> dict:
        person_id = req.get('person_id')
        reminders = self._reminder_db.list_reminders(
            person_id=int(person_id) if person_id is not None else None)
        return {'reminders': reminders}

    # ════════════════════════════════════════════════════════════════════
    # Social memory — original operations (unchanged)
    # ════════════════════════════════════════════════════════════════════

    def _lookup_person(self, req: dict) -> dict:
        query_emb = np.array(req['embedding'], dtype=np.float32)
        query_emb /= np.linalg.norm(query_emb) + 1e-8

        best_pid, best_sim = None, -1.0
        gallery = dict(self._gallery_cache)
        if gallery:
            for pid, matrix in gallery.items():
                sims = matrix @ query_emb
                person_best = float(sims.max())
                if person_best > best_sim:
                    best_sim, best_pid = person_best, pid
        else:
            with self._lock:
                rows = self._db.execute('SELECT id, embedding FROM persons').fetchall()
            for pid, blob in rows:
                emb = np.frombuffer(blob, dtype=np.float32).copy()
                emb /= np.linalg.norm(emb) + 1e-8
                sim = float(np.dot(query_emb, emb))
                if sim > best_sim:
                    best_sim, best_pid = sim, pid

        best_name = None
        if best_pid is not None:
            with self._lock:
                row = self._db.execute(
                    'SELECT name FROM persons WHERE id=?', (best_pid,)).fetchone()
            best_name = row[0] if row else None

        if best_sim >= self.sim_threshold:
            self.get_logger().info(f'Recognized: {best_name} (sim={best_sim:.3f})')
            return {'person_id': best_pid, 'name': best_name,
                    'similarity': best_sim, 'confidence': 'high'}

        if best_sim >= self.uncertain_threshold:
            self.get_logger().info(f'Probably {best_name} (sim={best_sim:.3f}) — uncertain')
            return {'person_id': None, 'similarity': best_sim, 'confidence': 'uncertain',
                    'best_candidate_id': best_pid, 'best_candidate_name': best_name}

        self.get_logger().info(f'Unknown (best_sim={best_sim:.3f})')
        return {'person_id': None, 'similarity': best_sim, 'confidence': 'unknown'}

    def _save_person(self, req: dict) -> dict:
        name = req['name'].strip()
        emb  = as_embedding(req['embedding'], self._face_emb_dim)
        now = datetime.now().isoformat()
        with self._lock:
            cur = self._db.execute(
                'INSERT INTO persons (name, embedding, first_seen, last_seen) VALUES (?, ?, ?, ?)',
                (name, emb.tobytes(), now, now),
            )
            self._db.commit()
            pid = cur.lastrowid
        self.get_logger().info(f'New person: {name} (id={pid})')
        return {'person_id': pid, 'name': name}

    def _update_embedding(self, req: dict) -> dict:
        pid   = req['person_id']
        alpha = float(req.get('alpha', 0.3))
        new_emb = as_embedding(req['embedding'], self._face_emb_dim)
        with self._lock:
            row = self._db.execute(
                'SELECT embedding FROM persons WHERE id=?', (pid,)).fetchone()
            if not row:
                return {'error': f'Person {pid} not found'}
            old_emb = np.frombuffer(row[0], dtype=np.float32).copy()
            old_emb /= np.linalg.norm(old_emb) + 1e-8
            merged = (1 - alpha) * old_emb + alpha * new_emb
            merged /= np.linalg.norm(merged) + 1e-8
            self._db.execute('UPDATE persons SET embedding=? WHERE id=?', (merged.tobytes(), pid))
            self._db.commit()
        return {'updated': True, 'person_id': pid}

    def _get_context(self, req: dict) -> dict:
        pid = req['person_id']
        with self._lock:
            row = self._db.execute(
                'SELECT name, first_seen, last_seen, meet_count FROM persons WHERE id=?', (pid,)
            ).fetchone()
            if not row:
                return {'error': f'Person {pid} not found'}
            notes = dict(self._db.execute(
                'SELECT key, value FROM person_notes WHERE person_id=?', (pid,)
            ).fetchall())
        name, first_seen, last_seen, meet_count = row
        gallery_count = len(self._gallery_cache.get(pid, []))
        return {
            'person_id':     pid,
            'name':          name,
            'first_seen':    first_seen,
            'last_seen':     last_seen,
            'meet_count':    meet_count,
            'gallery_count': gallery_count,
            'notes':         notes,
        }

    def _update_seen(self, req: dict) -> dict:
        pid = req['person_id']
        now = datetime.now().isoformat()
        with self._lock:
            self._db.execute(
                'UPDATE persons SET last_seen=?, meet_count=meet_count+1 WHERE id=?', (now, pid))
            self._db.commit()
        return {'updated': True}

    def _set_note(self, req: dict) -> dict:
        with self._lock:
            self._db.execute(
                'INSERT OR REPLACE INTO person_notes (person_id, key, value) VALUES (?, ?, ?)',
                (req['person_id'], req['key'], req['value']),
            )
            self._db.commit()
        return {'saved': True}

    def _get_knowledge(self, req: dict) -> dict:
        key = req.get('key')
        with self._lock:
            if key:
                row = self._db.execute(
                    'SELECT value FROM robot_knowledge WHERE key=?', (key,)).fetchone()
                return {'value': row[0] if row else None}
            rows = self._db.execute('SELECT key, value FROM robot_knowledge').fetchall()
        return {'knowledge': dict(rows)}

    def _set_knowledge(self, req: dict) -> dict:
        now = datetime.now().isoformat()
        with self._lock:
            self._db.execute(
                'INSERT OR REPLACE INTO robot_knowledge (key, value, updated) VALUES (?, ?, ?)',
                (req['key'], req['value'], now),
            )
            self._db.commit()
        return {'saved': True}

    # ── Gallery ────────────────────────────────────────────────────────

    def _gallery_add(self, req: dict) -> dict:
        pid        = req['person_id']
        photo_path = req['photo_path']
        quality    = float(req.get('quality', 1.0))
        source     = req.get('source', 'auto')
        now        = datetime.now().isoformat()

        emb = as_embedding(req['embedding'], self._face_emb_dim)

        existing = self._gallery_cache.get(pid)
        if existing is not None and len(existing) > 0:
            sims    = existing @ emb
            max_sim = float(sims.max())
            if max_sim > 0.95:
                return {'added': False, 'reason': 'duplicate'}
            if len(existing) >= 3 and max_sim < self.uncertain_threshold:
                self.get_logger().warn(
                    f'gallery_add REJECTED pid={pid}: sim={max_sim:.3f} — different person')
                return {'added': False, 'reason': 'embedding_mismatch'}

        with self._lock:
            # Person may have been merged away while face_gallery was saving the crop
            if not self._db.execute('SELECT 1 FROM persons WHERE id=?', (pid,)).fetchone():
                return {'added': False, 'reason': 'person_not_found'}
            self._db.execute(
                'INSERT INTO person_gallery '
                '(person_id, photo_path, embedding, quality, source, created_at) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (pid, photo_path, emb.tobytes(), quality, source, now),
            )
            self._db.commit()

        if pid in self._gallery_cache:
            self._gallery_cache[pid] = np.vstack([self._gallery_cache[pid], emb[None]])
        else:
            self._gallery_cache[pid] = emb[None].copy()

        count = len(self._gallery_cache[pid])
        self.get_logger().info(
            f'gallery_add: pid={pid} | {photo_path} | q={quality:.2f} | total={count}')
        return {'added': True, 'gallery_count': count}

    def _gallery_remove(self, req: dict) -> dict:
        photo_path = req['photo_path']
        with self._lock:
            row = self._db.execute(
                'SELECT id, person_id, embedding FROM person_gallery WHERE photo_path=?',
                (photo_path,),
            ).fetchone()
            if not row:
                return {'removed': False, 'reason': 'not_found'}
            gid, pid, blob = row
            self._db.execute('DELETE FROM person_gallery WHERE id=?', (gid,))
            self._db.commit()

        if pid in self._gallery_cache:
            emb = np.frombuffer(blob, dtype=np.float32).copy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            cache = self._gallery_cache[pid]
            sims  = cache @ emb
            idx   = int(sims.argmax())
            if sims[idx] > 0.999:
                new_cache = np.delete(cache, idx, axis=0)
                if len(new_cache) > 0:
                    self._gallery_cache[pid] = new_cache
                else:
                    del self._gallery_cache[pid]

        self.get_logger().info(f'gallery_remove: {photo_path} (pid={pid})')
        return {'removed': True, 'person_id': pid}

    def _gallery_rebuild_embedding(self, req: dict) -> dict:
        pid = req['person_id']
        with self._lock:
            rows = self._db.execute(
                'SELECT embedding FROM person_gallery WHERE person_id=?', (pid,)
            ).fetchall()
        if not rows:
            return {'rebuilt': False, 'reason': 'no_gallery_photos'}

        embs = []
        for (blob,) in rows:
            emb = np.frombuffer(blob, dtype=np.float32).copy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            embs.append(emb)

        mean_emb = np.mean(embs, axis=0)
        mean_emb /= np.linalg.norm(mean_emb) + 1e-8

        with self._lock:
            self._db.execute(
                'UPDATE persons SET embedding=? WHERE id=?', (mean_emb.tobytes(), pid))
            self._db.commit()

        self.get_logger().info(
            f'gallery_rebuild_embedding: pid={pid} | {len(embs)} photos averaged')
        return {'rebuilt': True, 'person_id': pid, 'gallery_count': len(embs)}

    def _gallery_list(self, req: dict) -> dict:
        pid = req.get('person_id')
        with self._lock:
            if pid:
                rows = self._db.execute(
                    'SELECT photo_path, quality, source, created_at '
                    'FROM person_gallery WHERE person_id=? ORDER BY created_at', (pid,)
                ).fetchall()
            else:
                rows = self._db.execute(
                    'SELECT person_id, photo_path, quality, source, created_at '
                    'FROM person_gallery ORDER BY person_id, created_at'
                ).fetchall()
        return {'photos': [list(r) for r in rows]}

    def _merge_persons(self, req: dict) -> dict:
        from_id = req['from_id']
        to_id   = req['to_id']
        GALLERY_LIMIT = 30
        VOICE_LIMIT   = self._SV_GALLERY_MAX
        if from_id == to_id:
            return {'error': f'merge_persons: from_id == to_id ({from_id})'}

        with self._lock:
            # Names before deletion — needed for the episodic DB and logs
            from_row = self._db.execute('SELECT name FROM persons WHERE id=?', (from_id,)).fetchone()
            to_row   = self._db.execute('SELECT name FROM persons WHERE id=?', (to_id,)).fetchone()
            if not from_row or not to_row:
                return {'error': f'person not found: from={from_id}, to={to_id}'}
            from_name = from_row[0]
            to_name   = to_row[0]

            # Optional gallery similarity check before merging
            if req.get('check_similarity', False):
                fg = self._gallery_cache.get(from_id)
                tg = self._gallery_cache.get(to_id)
                if fg is not None and tg is not None and len(fg) > 0 and len(tg) > 0:
                    fc = np.mean(fg, axis=0); fc /= np.linalg.norm(fc) + 1e-8
                    tc = np.mean(tg, axis=0); tc /= np.linalg.norm(tc) + 1e-8
                    sim = float(np.dot(fc, tc))
                    if sim < 0.35:
                        return {
                            'merged':     False,
                            'reason':     'similarity_too_low',
                            'similarity': round(sim, 3),
                            'message':    (f'Сходство лиц {from_name}↔{to_name} '
                                          f'слишком низкое ({sim:.2f}). Это точно один человек?'),
                        }

            # ── Photo gallery: keep the best by quality within the limit ──
            all_photos = self._db.execute(
                'SELECT id, person_id, photo_path, quality FROM person_gallery '
                'WHERE person_id IN (?,?) ORDER BY quality DESC',
                (from_id, to_id)
            ).fetchall()
            to_count    = sum(1 for p in all_photos if p[1] == to_id)
            available   = max(0, GALLERY_LIMIT - to_count)
            from_photos = [(p[0], p[2]) for p in all_photos if p[1] == from_id]
            to_transfer = from_photos[:available]
            to_discard  = from_photos[available:]

            # 1. Copy the kept photos out of the source folder (it's deleted in
            #    step 3). A failure before the commit only leaves spare copies.
            persons_dir = os.path.join(self._gallery_dir, 'persons')
            from_dirs = [d for d in glob.glob(os.path.join(persons_dir, f'{from_id}_*'))
                         if os.path.isdir(d)]
            from_dirs_abs = {os.path.abspath(d) for d in from_dirs}
            to_dir = self._person_gallery_dir(to_id, to_name)
            copies = []   # (gallery row id, new path)
            try:
                for gid, path in to_transfer:
                    if os.path.dirname(os.path.abspath(path)) not in from_dirs_abs:
                        continue   # lives outside the doomed folder — path stays valid
                    if not os.path.exists(path):
                        continue
                    dest = self._unique_path(to_dir, os.path.basename(path))
                    shutil.copy2(path, dest)
                    copies.append((gid, dest))
            except OSError as e:
                for _, dest in copies:
                    self._remove_quietly(dest)
                return {'error': f'merge_persons: copying photos failed: {e}'}

            # 2. One transaction for every social-DB change
            try:
                with self._db:
                    if to_transfer:
                        ids = tuple(p[0] for p in to_transfer)
                        ph  = ','.join('?' * len(ids))
                        self._db.execute(
                            f'UPDATE person_gallery SET person_id=? WHERE id IN ({ph})',
                            (to_id, *ids))
                    for gid, dest in copies:
                        self._db.execute(
                            'UPDATE person_gallery SET photo_path=? WHERE id=?', (dest, gid))
                    for gid, _ in to_discard:
                        self._db.execute('DELETE FROM person_gallery WHERE id=?', (gid,))

                    # Voice gallery: transfer up to free slots (newest first)
                    target_vc = self._db.execute(
                        'SELECT COUNT(*) FROM voice_gallery WHERE person_id=?', (to_id,)).fetchone()[0]
                    vslots = max(0, VOICE_LIMIT - target_vc)
                    voice_rows = self._db.execute(
                        'SELECT embedding, recorded_at FROM voice_gallery '
                        'WHERE person_id=? ORDER BY recorded_at DESC',
                        (from_id,)
                    ).fetchall()
                    for emb_blob, ts in voice_rows[:vslots]:
                        self._db.execute(
                            'INSERT INTO voice_gallery (person_id, embedding, recorded_at) VALUES (?,?,?)',
                            (to_id, emb_blob, ts))
                    self._db.execute('DELETE FROM voice_gallery WHERE person_id=?', (from_id,))

                    # Notes, meeting counter, delete the source
                    self._db.execute(
                        'INSERT OR IGNORE INTO person_notes (person_id, key, value) '
                        'SELECT ?, key, value FROM person_notes WHERE person_id=?', (to_id, from_id))
                    self._db.execute('DELETE FROM person_notes WHERE person_id=?', (from_id,))
                    self._db.execute(
                        'UPDATE persons SET meet_count=meet_count+'
                        '(SELECT COALESCE(meet_count,0) FROM persons WHERE id=?) WHERE id=?',
                        (from_id, to_id))
                    self._db.execute('DELETE FROM persons WHERE id=?', (from_id,))
            except sqlite3.Error as e:
                for _, dest in copies:
                    self._remove_quietly(dest)
                return {'error': f'merge_persons: DB update failed, nothing merged: {e}'}

            # Caches straight from the committed DB (never out of sync with it)
            self._gallery_cache.pop(from_id, None)
            self._voice_cache.pop(from_id, None)
            self._voice_gallery_cache.pop(from_id, None)
            self._reload_person_caches(to_id)

        # 3. Committed — now the source files can go
        for _, path in to_discard:
            self._remove_quietly(path)
        for d in from_dirs:
            shutil.rmtree(d, ignore_errors=True)

        # ── Episodic memory: rename the participant ──────────────────────
        try:
            with session(self._episodic_db_path) as ep_conn:
                for ep_id, parts_json in ep_conn.execute(
                        'SELECT id, participants FROM episodes').fetchall():
                    try:
                        parts = json.loads(parts_json) if parts_json else []
                    except Exception:
                        continue
                    if from_name in parts:
                        new_parts = [to_name if p == from_name else p for p in parts]
                        ep_conn.execute('UPDATE episodes SET participants=? WHERE id=?',
                                        (json.dumps(new_parts, ensure_ascii=False), ep_id))
        except Exception as e:
            self.get_logger().warn(f'merge_persons: episodic update failed: {e}')

        self.get_logger().info(f'merge_persons: {from_name}({from_id}) → {to_name}({to_id})')
        return {'merged': True, 'from_id': from_id, 'to_id': to_id,
                'from_name': from_name, 'to_name': to_name}

    def _person_gallery_dir(self, pid: int, name: str) -> str:
        """Person's photo folder — same layout as face_gallery_node (persons/{id}_{name})."""
        existing = [d for d in glob.glob(os.path.join(self._gallery_dir, 'persons', f'{pid}_*'))
                    if os.path.isdir(d)]
        if existing:
            return existing[0]
        d = os.path.join(self._gallery_dir, 'persons', f'{pid}_{(name or "unknown").replace(" ", "_")}')
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def _unique_path(directory: str, filename: str) -> str:
        stem, ext = os.path.splitext(filename)
        dest, n = os.path.join(directory, filename), 1
        while os.path.exists(dest):
            dest = os.path.join(directory, f'{stem}_m{n}{ext}')
            n += 1
        return dest

    @staticmethod
    def _remove_quietly(path: str):
        try:
            os.remove(path)
        except OSError:
            pass

    def _reload_person_caches(self, pid: int):
        """Rebuilds one person's face/voice caches from the DB. Called under self._lock."""
        rows = self._db.execute(
            'SELECT embedding FROM person_gallery WHERE person_id=?', (pid,)).fetchall()
        embs = []
        for (blob,) in rows:
            emb = np.frombuffer(blob, dtype=np.float32).copy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                embs.append(emb / norm)
        if embs:
            self._gallery_cache[pid] = np.stack(embs)
        else:
            self._gallery_cache.pop(pid, None)

        rows = self._db.execute(
            'SELECT embedding, recorded_at FROM voice_gallery WHERE person_id=? '
            'ORDER BY recorded_at ASC', (pid,)).fetchall()
        entries = []
        for blob, ts in rows:
            emb = np.frombuffer(blob, dtype=np.float32).copy()
            norm = np.linalg.norm(emb)
            if emb.shape[0] == self._voice_emb_dim and norm > 0:
                entries.append({'emb': emb / norm, 'recorded_at': float(ts)})
        if entries:
            self._voice_gallery_cache[pid] = entries
        else:
            self._voice_gallery_cache.pop(pid, None)

    def _lookup_by_name(self, req: dict) -> dict:
        """Looks up a person by exact name (case-insensitive)."""
        name = req.get('name', '').strip()
        with self._lock:
            row = self._db.execute(
                'SELECT id, name FROM persons WHERE LOWER(name)=LOWER(?)', (name,)
            ).fetchone()
        if row:
            return {'person_id': row[0], 'name': row[1]}
        return {'person_id': None}

    def _verify_person_claim(self, req: dict) -> dict:
        """Checks how well the current embeddings match the claimed person_id.

        Returns face_sim and voice_sim (0.0 if no data is available).
        """
        person_id  = req['person_id']
        face_emb   = req.get('face_embedding')
        voice_emb  = req.get('voice_embedding')

        face_sim  = 0.0
        voice_sim = 0.0
        has_voice_gallery = False

        if face_emb:
            q = np.array(face_emb, dtype=np.float32)
            q /= np.linalg.norm(q) + 1e-8
            gallery = self._gallery_cache.get(person_id)
            if gallery is not None and len(gallery) > 0:
                face_sim = float((gallery @ q).max())
            else:
                with self._lock:
                    row = self._db.execute(
                        'SELECT embedding FROM persons WHERE id=?', (person_id,)).fetchone()
                if row:
                    emb = np.frombuffer(row[0], dtype=np.float32).copy()
                    emb /= np.linalg.norm(emb) + 1e-8
                    face_sim = float(np.dot(q, emb))

        if voice_emb:
            q = np.array(voice_emb, dtype=np.float32)
            q /= np.linalg.norm(q) + 1e-8
            entries = self._voice_gallery_cache.get(person_id, [])
            if entries:
                has_voice_gallery = True
                valid = [e['emb'] for e in entries if e['emb'].shape[0] == q.shape[0]]
                if valid:
                    centroid = np.mean(np.stack(valid), axis=0)
                    centroid /= np.linalg.norm(centroid) + 1e-8
                    voice_sim = float(np.dot(centroid, q))
            else:
                emb = self._voice_cache.get(person_id)
                if emb is not None and emb.shape[0] == q.shape[0]:
                    has_voice_gallery = True
                    voice_sim = float(np.dot(emb, q))

        return {
            'face_sim':         round(face_sim,  4),
            'voice_sim':        round(voice_sim, 4),
            'has_voice_gallery': has_voice_gallery,
        }

    def _reload_gallery(self, req: dict) -> dict:
        self._load_gallery_cache()
        total = sum(len(v) for v in self._gallery_cache.values())
        return {'reloaded': True, 'persons': len(self._gallery_cache), 'total': total}

    # ── Voice embeddings ───────────────────────────────────────────────

    def _get_voice_embedding(self, req: dict) -> dict:
        pid = req['person_id']
        emb = self._voice_cache.get(pid)
        if emb is None:
            return {'person_id': pid, 'embedding': None, 'has_voice': False}
        return {'person_id': pid, 'embedding': emb.tolist(), 'has_voice': True}

    def _save_voice_embedding(self, req: dict) -> dict:
        pid = req['person_id']
        emb = as_embedding(req['embedding'], self._voice_emb_dim)
        with self._lock:
            row = self._db.execute('SELECT id FROM persons WHERE id=?', (pid,)).fetchone()
            if not row:
                return {'error': f'Person {pid} not found'}
            self._db.execute(
                'UPDATE persons SET voice_embedding=? WHERE id=?', (emb.tobytes(), pid))
            self._db.commit()
        self._voice_cache[pid] = emb
        self.get_logger().info(f'Voice embedding saved: pid={pid}')
        return {'saved': True, 'person_id': pid}

    def _update_voice_embedding(self, req: dict) -> dict:
        pid   = req['person_id']
        alpha = float(req.get('alpha', 0.3))
        new_emb = as_embedding(req['embedding'], self._voice_emb_dim)
        old_emb = self._voice_cache.get(pid)
        if old_emb is None:
            return self._save_voice_embedding(req)
        merged = (1 - alpha) * old_emb + alpha * new_emb
        merged /= np.linalg.norm(merged) + 1e-8
        with self._lock:
            self._db.execute(
                'UPDATE persons SET voice_embedding=? WHERE id=?', (merged.tobytes(), pid))
            self._db.commit()
        self._voice_cache[pid] = merged
        return {'updated': True, 'person_id': pid}

    def _lookup_by_voice(self, req: dict) -> dict:
        """Voice identification: compares the query against the gallery's normalized centroid.

        The centroid (mean_emb / ||mean_emb||) works better than the mean of
        pairwise similarities: it suppresses outliers and points to the
        cluster's overall direction.
        """
        voice_high      = float(req.get('high_threshold',      0.72))
        voice_uncertain = float(req.get('uncertain_threshold', 0.58))
        query = np.array(req['embedding'], dtype=np.float32)
        query /= np.linalg.norm(query) + 1e-8

        # Prefer the new gallery, fall back to the single-emb cache
        gallery = dict(self._voice_gallery_cache)
        single  = dict(self._voice_cache)
        all_pids = set(gallery) | set(single)
        if not all_pids:
            return {'person_id': None, 'similarity': 0.0, 'confidence': 'unknown'}

        best_pid, best_sim = None, -1.0
        for pid in all_pids:
            entries = gallery.get(pid)
            if entries:
                valid = [e['emb'] for e in entries if e['emb'].shape[0] == query.shape[0]]
                if not valid:
                    continue
                # Normalized centroid: average, normalize → compare
                centroid = np.mean(np.stack(valid), axis=0)
                norm = np.linalg.norm(centroid)
                if norm < 1e-8:
                    continue
                centroid /= norm
                sim = float(np.dot(centroid, query))
            else:
                emb = single.get(pid)
                if emb is None or emb.shape[0] != query.shape[0]:
                    continue
                sim = float(np.dot(emb, query))
            if sim > best_sim:
                best_sim, best_pid = sim, pid

        if best_pid is None:
            return {'person_id': None, 'similarity': 0.0, 'confidence': 'unknown'}
        with self._lock:
            row = self._db.execute('SELECT name FROM persons WHERE id=?', (best_pid,)).fetchone()
        best_name = row[0] if row else None
        if best_sim >= voice_high:
            return {'person_id': best_pid, 'name': best_name,
                    'similarity': best_sim, 'confidence': 'high'}
        if best_sim >= voice_uncertain:
            return {'person_id': None, 'name': best_name, 'similarity': best_sim,
                    'confidence': 'uncertain',
                    'best_candidate_id': best_pid, 'best_candidate_name': best_name}
        return {'person_id': None, 'similarity': best_sim, 'confidence': 'unknown'}

    def _get_voice_gallery(self, req: dict) -> dict:
        """Returns the full voice gallery for a person (up to 10 entries)."""
        pid = req['person_id']
        entries = self._voice_gallery_cache.get(pid, [])
        return {
            'person_id':  pid,
            'has_voice':  bool(entries),
            'gallery': [
                {'embedding': e['emb'].tolist(), 'timestamp': e['recorded_at']}
                for e in entries
            ],
        }

    _SV_QUALITY_MIN = 0.40  # minimum centroid similarity of a new embedding vs. the gallery

    def _add_voice_to_gallery(self, req: dict) -> dict:
        """Adds a voice embedding to the gallery.

        Rule: if < MAX entries — add it (if it passes the quality filter).
        If == MAX — replace the oldest entry only if it is older than REFRESH_DAYS.
        Quality filter: if the gallery already has entries, the new embedding must
        have centroid similarity >= _SV_QUALITY_MIN, otherwise it's noise/another
        person's voice — discard it.
        """
        pid         = req['person_id']
        try:
            new_emb = as_embedding(req['embedding'], self._voice_emb_dim)
        except ValueError as e:
            return {'added': False, 'reason': f'invalid_embedding: {e}'}
        recorded_at = float(req.get('timestamp', time.time()))

        with self._lock:
            if not self._db.execute('SELECT 1 FROM persons WHERE id=?', (pid,)).fetchone():
                return {'error': f'Person {pid} not found'}

            entries = self._voice_gallery_cache.get(pid, [])

            # Quality filter: discard noise by centroid similarity
            if entries:
                valid = [e['emb'] for e in entries
                         if e['emb'].shape[0] == new_emb.shape[0]]
                if valid:
                    centroid = np.mean(np.stack(valid), axis=0)
                    centroid /= np.linalg.norm(centroid) + 1e-8
                    sim = float(np.dot(centroid, new_emb))
                    if sim < self._SV_QUALITY_MIN:
                        self.get_logger().warn(
                            f'Voice gallery [{pid}]: noise rejected '
                            f'(centroid_sim={sim:.3f} < {self._SV_QUALITY_MIN})')
                        return {'added': False, 'reason': 'quality_too_low',
                                'centroid_sim': sim}

            if len(entries) < self._SV_GALLERY_MAX:
                self._db.execute(
                    'INSERT INTO voice_gallery (person_id, embedding, recorded_at) VALUES (?,?,?)',
                    (pid, new_emb.tobytes(), recorded_at))
                self._db.commit()
                self._voice_gallery_cache.setdefault(pid, []).append(
                    {'emb': new_emb, 'recorded_at': recorded_at})
                self.get_logger().info(
                    f'Voice gallery [{pid}]: entry added '
                    f'({len(self._voice_gallery_cache[pid])}/{self._SV_GALLERY_MAX})')
                return {'added': True, 'count': len(self._voice_gallery_cache[pid])}

            # Gallery full — check the age of the oldest entry
            oldest = min(entries, key=lambda e: e['recorded_at'])
            age_days = (time.time() - oldest['recorded_at']) / 86400
            if age_days < self._SV_REFRESH_DAYS:
                return {'added': False, 'reason': 'gallery_full_and_fresh',
                        'oldest_age_days': round(age_days, 1)}

            # Remove the oldest entry, add the new one
            # Find the row id to delete
            row = self._db.execute(
                'SELECT id FROM voice_gallery WHERE person_id=? ORDER BY recorded_at ASC LIMIT 1',
                (pid,)).fetchone()
            if row:
                self._db.execute('DELETE FROM voice_gallery WHERE id=?', (row[0],))
            self._db.execute(
                'INSERT INTO voice_gallery (person_id, embedding, recorded_at) VALUES (?,?,?)',
                (pid, new_emb.tobytes(), recorded_at))
            self._db.commit()

            # Update the cache
            entries = [e for e in entries if e['recorded_at'] != oldest['recorded_at']]
            entries.append({'emb': new_emb, 'recorded_at': recorded_at})
            self._voice_gallery_cache[pid] = entries
            self.get_logger().info(
                f'Voice gallery [{pid}]: replaced oldest entry '
                f'(age={age_days:.1f}d)')
            return {'added': True, 'count': len(entries), 'replaced_oldest': True}

    # ════════════════════════════════════════════════════════════════════
    # Shutdown (via on_shutdown/on_cleanup — not destroy_node)
    # ════════════════════════════════════════════════════════════════════


def main():
    rclpy.init()
    node = MemoryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Ctrl-C / SIGINT from launch skips on_shutdown: let the last conversation's
        # summary + reminder extraction finish before the DBs close.
        node._join_bg()
        node._close_db()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
