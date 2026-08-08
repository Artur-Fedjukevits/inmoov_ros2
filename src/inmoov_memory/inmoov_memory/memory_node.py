#!/usr/bin/env python3
"""
memory_node.py — Memory Manager Node (v2)
==========================================
Единый ROS2-узел, управляющий всей памятью робота:

  СОЦИАЛЬНАЯ ПАМЯТЬ (SQLite, /home/artur/inmoov_memory.db):
    persons, person_gallery, person_notes, robot_knowledge
    — распознавание лиц, голосов, заметки о людях

  РАБОЧАЯ ПАМЯТЬ  (RAM, WorkingMemory):
    время, место, режим, люди рядом — всегда в LLM context

  ЭПИЗОДИЧЕСКАЯ ПАМЯТЬ (SQLite, /home/artur/inmoov_episodic.db):
    краткосрочные воспоминания, диалоги — последние 5 в prompting

  СЕМАНТИЧЕСКАЯ ПАМЯТЬ (SQLite + ChromaDB, /home/artur/inmoov_semantic.db):
    долгосрочные факты, предпочтения — только через tool call

ROS2-интерфейсы:
  /memory/query   (srv MemoryQuery) — все операции (ops ниже)
  /memory/context (pub String, 30s) — рабочая память + эпизоды для LLM prompt
  /social_context (sub String)      — обновление рабочей памяти из identity_manager
  /conversation_end (sub String)    — после диалога: резюме + извлечение фактов
  /robot_sleep    (sub Bool, latched)

Операции /memory/query:
  Социальная память (существующие):
    lookup_person, save_person, update_embedding, get_context,
    update_seen, set_note, get_knowledge, set_knowledge,
    gallery_add, gallery_remove, gallery_rebuild_embedding,
    gallery_list, merge_persons, reload_gallery,
    get_voice_embedding, save_voice_embedding, update_voice_embedding, lookup_by_voice

  Новые (3-слойная память):
    search_semantic        — семантический поиск в долговременной памяти
    save_semantic_fact     — сохранить факт в долговременную память
    search_episodes        — поиск по эпизодической памяти
    get_recent_episodes    — последние N эпизодов (для LLM tool)
    save_episode           — сохранить эпизод вручную
    get_working_memory     — текущее состояние рабочей памяти
    update_working_memory  — обновить рабочую память
    get_memory_context     — собрать текстовый контекст для LLM
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime

import numpy as np
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String, Bool
from inmoov_msgs.srv import MemoryQuery

from inmoov_memory.memory_manager import MemoryManager
from inmoov_memory.reminder_db import ReminderDB

logger = logging.getLogger(__name__)


class MemoryNode(LifecycleNode):
    def __init__(self):
        super().__init__('memory_node')
        # Инстанс-переменные — будут заполнены в on_configure
        self._db              = None
        self._lock            = threading.Lock()
        self._gallery_cache: dict[int, np.ndarray] = {}
        self._voice_cache: dict[int, np.ndarray]   = {}
        self._voice_emb_dim   = 192
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

    # ── Lifecycle callbacks ────────────────────────────────────────────────

    def _dp(self, name, default=None):
        """Безопасный declare_parameter: игнорирует повторное объявление при re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        # ── Параметры ──────────────────────────────────────────────────────
        self._dp('db_path',                    '/home/artur/inmoov_memory.db')
        self._dp('episodic_db_path',           '/home/artur/inmoov_episodic.db')
        self._dp('semantic_db_path',           '/home/artur/inmoov_semantic.db')
        self._dp('chroma_path',                '/home/artur/inmoov_chroma')
        self._dp('reminder_db_path',           '/home/artur/inmoov_reminders.db')
        self._dp('ollama_url',                 'http://192.168.10.118:11434')
        self._dp('ollama_model',               'qwen3.6:27b')
        self._dp('similarity_threshold',       0.55)
        self._dp('uncertain_threshold',        0.40)
        self._dp('context_publish_rate',       30.0)
        self._dp('telegram_reminder_person_id', 5)
        self._dp('telegram_reminder_check_sec', 60.0)
        self._dp('reminder_default_time',      '07:00')

        db_path      = self.get_parameter('db_path').value
        episodic_db  = self.get_parameter('episodic_db_path').value
        semantic_db  = self.get_parameter('semantic_db_path').value
        chroma_path  = self.get_parameter('chroma_path').value
        reminder_db  = self.get_parameter('reminder_db_path').value
        ollama_url   = self.get_parameter('ollama_url').value
        ollama_model = self.get_parameter('ollama_model').value
        self.sim_threshold          = self.get_parameter('similarity_threshold').value
        self.uncertain_threshold    = self.get_parameter('uncertain_threshold').value
        self._tg_reminder_person_id = self.get_parameter('telegram_reminder_person_id').value
        self._reminder_default_time = self.get_parameter('reminder_default_time').value

        # ── Социальная память (SQLite) ─────────────────────────────────────
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._init_social_db()
        self._load_gallery_cache()
        self._load_voice_cache()
        self._load_voice_gallery_cache()

        # ── Напоминания ────────────────────────────────────────────────────
        self._reminder_db = ReminderDB(reminder_db)
        self.get_logger().info(f'ReminderDB: {reminder_db}')

        # ── 3-слойная память ───────────────────────────────────────────────
        self._mm = MemoryManager(
            db_path=episodic_db,
            semantic_db_path=semantic_db,
            chroma_path=chroma_path,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
        )
        self.get_logger().info(
            f'MemoryManager: episodic={episodic_db} semantic={semantic_db}')

        # ── Сервис — создаём в configure, доступен сразу после активации ───
        self.create_service(MemoryQuery, '/memory/query', self._handle)

        # ── Подписки ───────────────────────────────────────────────────────
        self.create_subscription(String, '/social_context',   self._social_context_cb, 10)
        self.create_subscription(String, '/conversation_end', self._conversation_end_cb, 10)
        _latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Bool, '/robot_sleep', self._robot_sleep_cb, _latched)

        # ── Lifecycle publishers (молчат пока не вызван on_activate) ───────
        self._ctx_pub     = self.create_lifecycle_publisher(String, '/memory/context', 10)
        self._tg_push_pub = self.create_lifecycle_publisher(String, '/telegram_push', 10)

        self.get_logger().info(
            f'MemoryNode configured. БД: {db_path} | порог: {self.sim_threshold}')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        # ОБЯЗАТЕЛЬНО активировать каждый lifecycle publisher, передавая state
        self._ctx_pub.on_activate(state)
        self._tg_push_pub.on_activate(state)

        ctx_rate     = self.get_parameter('context_publish_rate').value
        tg_check_sec = self.get_parameter('telegram_reminder_check_sec').value

        self._timers.append(self.create_timer(ctx_rate, self._publish_memory_context))
        if self._tg_reminder_person_id > 0:
            self._timers.append(
                self.create_timer(tg_check_sec, self._send_due_reminders_to_telegram))

        # Первая публикация контекста сразу
        self._publish_memory_context()

        self.get_logger().info(
            f'MemoryNode active. Контекст каждые {ctx_rate}с | '
            f'telegram reminders person_id={self._tg_reminder_person_id} '
            f'каждые {tg_check_sec:.0f}с')
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
        self._close_db()
        self._gallery_cache.clear()
        self._voice_cache.clear()
        self._voice_gallery_cache.clear()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self._close_db()
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        self._close_db()
        return TransitionCallbackReturn.SUCCESS

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
    # Инициализация социальной БД (бывший _init_db)
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
                self.get_logger().info('Миграция БД: добавлена колонка voice_embedding')
            except sqlite3.OperationalError:
                pass
            try:
                self._db.execute('ALTER TABLE persons ADD COLUMN telegram_id INTEGER DEFAULT NULL')
                self._db.commit()
                self.get_logger().info('Миграция БД: добавлена колонка telegram_id')
            except sqlite3.OperationalError:
                pass
            # Голосовая галерея (до 10 отпечатков на человека)
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
            # Миграция: перенести одиночный voice_embedding в voice_gallery (только если галерея пуста)
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
    # Галерея и голосовые кэши
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
            f'Галерея: {len(self._gallery_cache)} человек, {total} фото')

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
            f'Голосовые embeddings: {len(self._voice_cache)} человек'
            + (f' ({skipped} пропущено)' if skipped else ''))

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
            f'Голосовая галерея: {len(self._voice_gallery_cache)} человек, {total} записей')

    # ════════════════════════════════════════════════════════════════════
    # ROS2 коллбеки — новые
    # ════════════════════════════════════════════════════════════════════

    def _social_context_cb(self, msg: String):
        """Обновляет рабочую память из social_context identity_manager."""
        try:
            ctx = json.loads(msg.data)
            present = ctx.get('person_present', False)
            name    = ctx.get('name', '')
            emotion = ctx.get('emotion', 'neutral')

            # Список людей рядом
            people = [name] if present and name else []
            self._mm.working.update_environment(people=people)

            # На кого смотрит робот
            self._mm.working.update_robot_state(
                facing=name if present and name else None,
            )

            # Режим работы
            state_raw = ctx.get('state', '')
            mode_map = {
                'IDLE':         'idle',
                'INTERACTING':  'conversation',
                'INTRODUCING':  'conversation',
            }
            mode = mode_map.get(str(state_raw), 'idle')
            self._mm.working.update_robot_state(mode=mode)

        except Exception as e:
            self.get_logger().warn(f'social_context_cb: {e}')

    def _conversation_end_cb(self, msg: String):
        """Обрабатывает завершение диалога: резюме + извлечение фактов в фоне."""
        try:
            data         = json.loads(msg.data)
            transcript   = data.get('transcript', '')
            participants = data.get('participants', [])
            if not transcript.strip():
                return
            threading.Thread(
                target=self._run_after_conversation,
                args=(transcript, participants),
                daemon=True,
            ).start()
        except Exception as e:
            self.get_logger().warn(f'conversation_end_cb: {e}')

    def _run_after_conversation(self, transcript: str, participants: list):
        try:
            self._mm.after_conversation(transcript, participants=participants)
            self.get_logger().info(
                f'Эпизод сохранён: {len(transcript)} симв., '
                f'участники: {participants}')
        except Exception as e:
            self.get_logger().error(f'after_conversation error: {e}')
        finally:
            self._publish_memory_context()
        # Авто-извлечение напоминаний из диалога (в отдельном потоке, не блокирует)
        threading.Thread(
            target=self._extract_reminders,
            args=(transcript, participants),
            daemon=True,
        ).start()

    # Ключевые слова, при наличии которых вызываем LLM для извлечения напоминаний
    _REMINDER_KEYWORDS = [
        'рождени', 'выступлени', 'концерт', 'экзамен', 'спектакл',
        'годовщин', 'свидани', 'соревновани', 'дедлайн', 'визит',
        'операци', 'приём врач', 'праздни', 'юбилей',
        'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
        'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря',
    ]

    def _extract_reminders(self, transcript: str, participants: list):
        """Авто-извлечение напоминаний из диалога через LLM.
        Запускается только если транскрипт содержит ключевые слова с датами."""
        if not participants:
            return
        person_name = participants[0]
        transcript_lower = transcript.lower()
        if not any(kw in transcript_lower for kw in self._REMINDER_KEYWORDS):
            return

        with self._lock:
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
            raw = self._mm._call_ollama(prompt, system=system)
            raw = raw.strip().lstrip('```json').lstrip('```').rstrip('```').strip()
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
                    continue  # прошедшая дата — пропускаем
                rid = self._reminder_db.add_reminder(
                    person_id=person_id,
                    person_name=person_name,
                    message=message,
                    trigger_date=remind_date,
                    source='auto',
                )
                count += 1
                self.get_logger().info(
                    f'Авто-напоминание: id={rid} date={remind_date or "next_meeting"} '
                    f'"{message[:60]}"')
            if count:
                self.get_logger().info(
                    f'Авто-извлечено {count} напоминаний для {person_name}')
        except Exception as e:
            self.get_logger().warn(f'_extract_reminders: {e}')

    def _robot_sleep_cb(self, msg: Bool):
        mode = 'idle' if msg.data else 'idle'
        self._mm.working.update_robot_state(mode=mode)
        if msg.data:
            self._mm.working.update_environment(people=[])
            self._mm.working.update_robot_state(facing=None)

    # ── /memory/context publisher ──────────────────────────────────────

    def _send_due_reminders_to_telegram(self):
        """Таймер: проверяет просроченные напоминания (delivered=0) и отправляет в Telegram.

        После публикации помечает delivered=1 — повторно не отправляются.
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
            self.get_logger().warn(f'TG reminder timer: ошибка get_due: {e}')
            return

        for r in reminders:
            text = f'⏰ <b>Напоминание:</b> {_html.escape(r["message"])}'
            try:
                msg = String()
                msg.data = json.dumps(
                    {'text': text, 'parse_mode': 'HTML'}, ensure_ascii=False)
                self._tg_push_pub.publish(msg)
                self._reminder_db.mark_delivered(r['id'])
                self.get_logger().info(
                    f'TG reminder: id={r["id"]} отправлено и помечено delivered')
            except Exception as e:
                self.get_logger().warn(f'TG reminder: ошибка отправки id={r["id"]}: {e}')

    def _publish_memory_context(self):
        """Публикует рабочую память + последние эпизоды в /memory/context."""
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
    # Диспетчер /memory/query
    # ════════════════════════════════════════════════════════════════════

    def _handle(self, request, response):
        try:
            req = json.loads(request.request_json)
            op  = req.get('op', '')
            ops = {
                # ── Социальная память (оригинальные) ──────────────────
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
                # ── 3-слойная память (новые) ───────────────────────────
                'search_semantic':            self._op_search_semantic,
                'save_semantic_fact':         self._op_save_semantic_fact,
                'search_episodes':            self._op_search_episodes,
                'get_recent_episodes':        self._op_get_recent_episodes,
                'save_episode':               self._op_save_episode,
                'get_working_memory':         self._op_get_working_memory,
                'update_working_memory':      self._op_update_working_memory,
                'get_memory_context':         self._op_get_memory_context,
                # ── Напоминания ────────────────────────────────────────
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
    # Новые операции — 3-слойная память
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
        """Возвращает текстовый контекст для вставки в LLM system prompt."""
        working_text = self._mm.working.to_text()
        recent_text  = self._mm.episodic.get_recent_text(limit=int(req.get('limit', 5)))
        parts = ['== Текущий момент ==', working_text]
        if recent_text:
            parts += ['', '== Последние события ==', recent_text]
        return {'context': '\n'.join(parts)}

    # ════════════════════════════════════════════════════════════════════
    # Напоминания
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
            f'Напоминание создано: id={rid} person={person_name} '
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
        """Удаляет все показанные (delivered=1) напоминания — пользователь подтвердил."""
        person_id = req.get('person_id')
        if person_id is None:
            return {'error': 'person_id обязателен'}
        count = self._reminder_db.confirm_all_delivered(int(person_id))
        self.get_logger().info(
            f'Напоминания подтверждены: person_id={person_id}, удалено={count}')
        return {'confirmed': count}

    def _op_list_reminders(self, req: dict) -> dict:
        person_id = req.get('person_id')
        reminders = self._reminder_db.list_reminders(
            person_id=int(person_id) if person_id is not None else None)
        return {'reminders': reminders}

    # ════════════════════════════════════════════════════════════════════
    # Социальная память — оригинальные операции (без изменений)
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
            self.get_logger().info(f'Распознан: {best_name} (sim={best_sim:.3f})')
            return {'person_id': best_pid, 'name': best_name,
                    'similarity': best_sim, 'confidence': 'high'}

        if best_sim >= self.uncertain_threshold:
            self.get_logger().info(f'Вероятно {best_name} (sim={best_sim:.3f}) — неуверенно')
            return {'person_id': None, 'similarity': best_sim, 'confidence': 'uncertain',
                    'best_candidate_id': best_pid, 'best_candidate_name': best_name}

        self.get_logger().info(f'Неизвестный (best_sim={best_sim:.3f})')
        return {'person_id': None, 'similarity': best_sim, 'confidence': 'unknown'}

    def _save_person(self, req: dict) -> dict:
        name = req['name'].strip()
        emb  = np.array(req['embedding'], dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        now = datetime.now().isoformat()
        with self._lock:
            cur = self._db.execute(
                'INSERT INTO persons (name, embedding, first_seen, last_seen) VALUES (?, ?, ?, ?)',
                (name, emb.tobytes(), now, now),
            )
            self._db.commit()
            pid = cur.lastrowid
        self.get_logger().info(f'Новый человек: {name} (id={pid})')
        return {'person_id': pid, 'name': name}

    def _update_embedding(self, req: dict) -> dict:
        pid   = req['person_id']
        alpha = float(req.get('alpha', 0.3))
        new_emb = np.array(req['embedding'], dtype=np.float32)
        new_emb /= np.linalg.norm(new_emb) + 1e-8
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

    # ── Галерея ────────────────────────────────────────────────────────

    def _gallery_add(self, req: dict) -> dict:
        pid        = req['person_id']
        photo_path = req['photo_path']
        quality    = float(req.get('quality', 1.0))
        source     = req.get('source', 'auto')
        now        = datetime.now().isoformat()

        emb = np.array(req['embedding'], dtype=np.float32)
        emb /= np.linalg.norm(emb) + 1e-8

        existing = self._gallery_cache.get(pid)
        if existing is not None and len(existing) > 0:
            sims    = existing @ emb
            max_sim = float(sims.max())
            if max_sim > 0.95:
                return {'added': False, 'reason': 'duplicate'}
            if len(existing) >= 3 and max_sim < self.uncertain_threshold:
                self.get_logger().warn(
                    f'gallery_add ОТКЛОНЕНО pid={pid}: sim={max_sim:.3f} — другой человек')
                return {'added': False, 'reason': 'embedding_mismatch'}

        with self._lock:
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
            f'gallery_add: pid={pid} | {photo_path} | q={quality:.2f} | итого={count}')
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
            f'gallery_rebuild_embedding: pid={pid} | {len(embs)} фото усреднено')
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
        import os, shutil, glob, json as _json
        from_id = req['from_id']
        to_id   = req['to_id']
        GALLERY_LIMIT = 30
        VOICE_LIMIT   = self._SV_GALLERY_MAX

        with self._lock:
            # Имена до удаления — нужны для episodic DB и логов
            from_row = self._db.execute('SELECT name FROM persons WHERE id=?', (from_id,)).fetchone()
            to_row   = self._db.execute('SELECT name FROM persons WHERE id=?', (to_id,)).fetchone()
            if not from_row or not to_row:
                return {'error': f'person not found: from={from_id}, to={to_id}'}
            from_name = from_row[0]
            to_name   = to_row[0]

            # Опциональная проверка сходства галерей перед слиянием
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

            # ── Фото-галерея: сливаем, соблюдаем лимит (берём лучшие по quality) ──
            all_photos = self._db.execute(
                'SELECT id, person_id, photo_path, quality FROM person_gallery '
                'WHERE person_id IN (?,?) ORDER BY quality DESC',
                (from_id, to_id)
            ).fetchall()

            to_count   = sum(1 for p in all_photos if p[1] == to_id)
            available  = max(0, GALLERY_LIMIT - to_count)
            from_photos = [(p[0], p[2]) for p in all_photos if p[1] == from_id]
            to_transfer = from_photos[:available]
            to_discard  = from_photos[available:]

            if to_transfer:
                ids = tuple(p[0] for p in to_transfer)
                ph  = ','.join('?' * len(ids))
                self._db.execute(f'UPDATE person_gallery SET person_id=? WHERE id IN ({ph})',
                                 (to_id, *ids))
            for _, path in to_discard:
                self._db.execute('DELETE FROM person_gallery WHERE photo_path=?', (path,))
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass

            # ── Голосовая галерея: переносим до свободных слотов (новейшие) ──
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

            # ── Заметки, счётчик встреч, удаление источника ──────────────────
            self._db.execute(
                'INSERT OR IGNORE INTO person_notes (person_id, key, value) '
                'SELECT ?, key, value FROM person_notes WHERE person_id=?', (to_id, from_id))
            self._db.execute('DELETE FROM person_notes WHERE person_id=?', (from_id,))
            self._db.execute(
                'UPDATE persons SET meet_count=meet_count+'
                '(SELECT COALESCE(meet_count,0) FROM persons WHERE id=?) WHERE id=?',
                (from_id, to_id))
            self._db.execute('DELETE FROM persons WHERE id=?', (from_id,))
            self._db.commit()

        # ── Обновляем кэши ───────────────────────────────────────────────
        if from_id in self._gallery_cache:
            old = self._gallery_cache.pop(from_id)
            if to_id in self._gallery_cache:
                self._gallery_cache[to_id] = np.vstack([self._gallery_cache[to_id], old])
            else:
                self._gallery_cache[to_id] = old

        if from_id in self._voice_gallery_cache:
            src = self._voice_gallery_cache.pop(from_id)
            dst = self._voice_gallery_cache.setdefault(to_id, [])
            dst.extend(src)
            dst.sort(key=lambda e: e.get('recorded_at', 0), reverse=True)
            self._voice_gallery_cache[to_id] = dst[:VOICE_LIMIT]

        # ── Физическая директория источника ─────────────────────────────
        for d in glob.glob(f'/home/artur/inmoov_faces/persons/{from_id}_*'):
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)

        # ── Эпизодическая память: переименовываем участника ─────────────
        try:
            import sqlite3 as _sq
            ep_conn = _sq.connect('/home/artur/inmoov_episodic.db')
            ep_cur  = ep_conn.cursor()
            ep_cur.execute('SELECT id, participants FROM episodes')
            for ep_id, parts_json in ep_cur.fetchall():
                try:
                    parts = _json.loads(parts_json) if parts_json else []
                except Exception:
                    continue
                if from_name in parts:
                    new_parts = [to_name if p == from_name else p for p in parts]
                    ep_cur.execute('UPDATE episodes SET participants=? WHERE id=?',
                                   (_json.dumps(new_parts, ensure_ascii=False), ep_id))
            ep_conn.commit()
            ep_conn.close()
        except Exception as e:
            self.get_logger().warn(f'merge_persons: episodic update failed: {e}')

        self.get_logger().info(f'merge_persons: {from_name}({from_id}) → {to_name}({to_id})')
        return {'merged': True, 'from_id': from_id, 'to_id': to_id,
                'from_name': from_name, 'to_name': to_name}

    def _lookup_by_name(self, req: dict) -> dict:
        """Поиск человека по точному имени (регистронезависимый)."""
        name = req.get('name', '').strip()
        with self._lock:
            row = self._db.execute(
                'SELECT id, name FROM persons WHERE LOWER(name)=LOWER(?)', (name,)
            ).fetchone()
        if row:
            return {'person_id': row[0], 'name': row[1]}
        return {'person_id': None}

    def _verify_person_claim(self, req: dict) -> dict:
        """Проверяет, насколько текущие embeddings соответствуют заявленному person_id.

        Возвращает face_sim и voice_sim (0.0 если данных нет).
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

    # ── Голосовые embeddings ───────────────────────────────────────────

    def _get_voice_embedding(self, req: dict) -> dict:
        pid = req['person_id']
        emb = self._voice_cache.get(pid)
        if emb is None:
            return {'person_id': pid, 'embedding': None, 'has_voice': False}
        return {'person_id': pid, 'embedding': emb.tolist(), 'has_voice': True}

    def _save_voice_embedding(self, req: dict) -> dict:
        pid = req['person_id']
        emb = np.array(req['embedding'], dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        with self._lock:
            row = self._db.execute('SELECT id FROM persons WHERE id=?', (pid,)).fetchone()
            if not row:
                return {'error': f'Person {pid} not found'}
            self._db.execute(
                'UPDATE persons SET voice_embedding=? WHERE id=?', (emb.tobytes(), pid))
            self._db.commit()
        self._voice_cache[pid] = emb
        self.get_logger().info(f'Голосовой embedding сохранён: pid={pid}')
        return {'saved': True, 'person_id': pid}

    def _update_voice_embedding(self, req: dict) -> dict:
        pid   = req['person_id']
        alpha = float(req.get('alpha', 0.3))
        new_emb = np.array(req['embedding'], dtype=np.float32)
        new_emb /= np.linalg.norm(new_emb) + 1e-8
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
        """Идентификация по голосу: сравниваем запрос с нормализованным центроидом галереи.

        Центроид (mean_emb / ||mean_emb||) лучше чем среднее попарных сходств:
        он подавляет выбросы и указывает на общее направление кластера.
        """
        voice_high      = float(req.get('high_threshold',      0.72))
        voice_uncertain = float(req.get('uncertain_threshold', 0.58))
        query = np.array(req['embedding'], dtype=np.float32)
        query /= np.linalg.norm(query) + 1e-8

        # Предпочитаем новую галерею, fallback на single-emb кэш
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
                # Нормализованный центроид: усредняем, нормализуем → сравниваем
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
        """Возвращает полную голосовую галерею человека (до 10 записей)."""
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

    _SV_QUALITY_MIN = 0.40  # минимальное centroid-сходство нового embedding с галереей

    def _add_voice_to_gallery(self, req: dict) -> dict:
        """Добавляет голосовой embedding в галерею.

        Правило: если < MAX записей — добавляем (если проходит quality-фильтр).
        Если == MAX — заменяем самую старую запись только если ей > REFRESH_DAYS дней.
        Quality-фильтр: если в галерее уже есть записи, новый embedding должен иметь
        centroid-сходство >= _SV_QUALITY_MIN, иначе это шум/чужой голос — отбрасываем.
        """
        import time as _time
        pid         = req['person_id']
        new_emb     = np.array(req['embedding'], dtype=np.float32)
        norm        = np.linalg.norm(new_emb)
        if norm < 1e-8:
            return {'added': False, 'reason': 'zero_norm'}
        new_emb /= norm
        recorded_at = float(req.get('timestamp', _time.time()))

        with self._lock:
            if not self._db.execute('SELECT 1 FROM persons WHERE id=?', (pid,)).fetchone():
                return {'error': f'Person {pid} not found'}

            entries = self._voice_gallery_cache.get(pid, [])

            # Quality-фильтр: отсеиваем мусор по centroid-сходству
            if entries:
                valid = [e['emb'] for e in entries
                         if e['emb'].shape[0] == new_emb.shape[0]]
                if valid:
                    centroid = np.mean(np.stack(valid), axis=0)
                    centroid /= np.linalg.norm(centroid) + 1e-8
                    sim = float(np.dot(centroid, new_emb))
                    if sim < self._SV_QUALITY_MIN:
                        self.get_logger().warn(
                            f'Голосовая галерея [{pid}]: отброшен шум '
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
                    f'Голосовая галерея [{pid}]: добавлена запись '
                    f'({len(self._voice_gallery_cache[pid])}/{self._SV_GALLERY_MAX})')
                return {'added': True, 'count': len(self._voice_gallery_cache[pid])}

            # Галерея полная — проверяем возраст самой старой записи
            oldest = min(entries, key=lambda e: e['recorded_at'])
            age_days = (_time.time() - oldest['recorded_at']) / 86400
            if age_days < self._SV_REFRESH_DAYS:
                return {'added': False, 'reason': 'gallery_full_and_fresh',
                        'oldest_age_days': round(age_days, 1)}

            # Удаляем самую старую запись, добавляем новую
            # Определяем id строки для удаления
            row = self._db.execute(
                'SELECT id FROM voice_gallery WHERE person_id=? ORDER BY recorded_at ASC LIMIT 1',
                (pid,)).fetchone()
            if row:
                self._db.execute('DELETE FROM voice_gallery WHERE id=?', (row[0],))
            self._db.execute(
                'INSERT INTO voice_gallery (person_id, embedding, recorded_at) VALUES (?,?,?)',
                (pid, new_emb.tobytes(), recorded_at))
            self._db.commit()

            # Обновляем кэш
            entries = [e for e in entries if e['recorded_at'] != oldest['recorded_at']]
            entries.append({'emb': new_emb, 'recorded_at': recorded_at})
            self._voice_gallery_cache[pid] = entries
            self.get_logger().info(
                f'Голосовая галерея [{pid}]: заменена старая запись '
                f'(возраст={age_days:.1f}д)')
            return {'added': True, 'count': len(entries), 'replaced_oldest': True}

    # ════════════════════════════════════════════════════════════════════
    # Завершение (через on_shutdown/on_cleanup — не destroy_node)
    # ════════════════════════════════════════════════════════════════════


def main():
    rclpy.init()
    node = MemoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
