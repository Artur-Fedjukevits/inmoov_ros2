"""
reminder_db.py — Хранилище напоминаний (SQLite)

Схема:
    reminders(id, person_id, person_name, trigger_date, trigger_time, message, source, delivered, created_at)

trigger_date:
    NULL  → показываем при каждой встрече пока не подтвердят
    YYYY-MM-DD → показываем начиная с этой даты

trigger_time:
    NULL  → используется default_time при проверке (обычно 07:00)
    HH:MM → показываем начиная с этого времени в день trigger_date

Жизненный цикл:
    добавлен (delivered=0) → показан при встрече (delivered=1) → подтверждён пользователем → удалён
"""

import sqlite3
import threading
from datetime import datetime


class ReminderDB:
    def __init__(self, db_path: str):
        self._db   = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            self._db.executescript('''
                CREATE TABLE IF NOT EXISTS reminders (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id    INTEGER NOT NULL,
                    person_name  TEXT    NOT NULL DEFAULT '',
                    trigger_date TEXT,
                    trigger_time TEXT,
                    message      TEXT    NOT NULL,
                    source       TEXT    NOT NULL DEFAULT 'manual',
                    delivered    INTEGER NOT NULL DEFAULT 0,
                    created_at   TEXT    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rem_person ON reminders(person_id);
                CREATE INDEX IF NOT EXISTS idx_rem_date   ON reminders(trigger_date);
            ''')
            # Migration: add trigger_time column if it doesn't exist yet
            cols = {row[1] for row in self._db.execute("PRAGMA table_info(reminders)")}
            if 'trigger_time' not in cols:
                self._db.execute('ALTER TABLE reminders ADD COLUMN trigger_time TEXT')
            self._db.commit()

    # ── Запись ────────────────────────────────────────────────────────────

    def add_reminder(self, person_id: int, person_name: str, message: str,
                     trigger_date: str | None = None,
                     trigger_time: str | None = None,
                     source: str = 'manual') -> int:
        """Создать напоминание. trigger_date=None → при следующей встрече.

        trigger_time: 'HH:MM' или None (используется default_time при get_due).

        Возвращает id существующего напоминания если точный дубликат уже есть
        (совпадают person_id, trigger_date, trigger_time и нормализованный текст, delivered=0).
        """
        msg_norm = message.strip().lower()
        now = datetime.now().isoformat()
        with self._lock:
            if trigger_date is None:
                existing = self._db.execute(
                    'SELECT id FROM reminders '
                    'WHERE person_id=? AND trigger_date IS NULL '
                    '  AND LOWER(TRIM(message))=? AND delivered=0 LIMIT 1',
                    (person_id, msg_norm),
                ).fetchone()
            else:
                existing = self._db.execute(
                    'SELECT id FROM reminders '
                    'WHERE person_id=? AND trigger_date=? AND trigger_time IS ? '
                    '  AND LOWER(TRIM(message))=? AND delivered=0 LIMIT 1',
                    (person_id, trigger_date, trigger_time, msg_norm),
                ).fetchone()
            if existing:
                return existing[0]
            cur = self._db.execute(
                'INSERT INTO reminders '
                '(person_id, person_name, trigger_date, trigger_time, message, source, created_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?)',
                (person_id, person_name, trigger_date, trigger_time, message, source, now),
            )
            self._db.commit()
        return cur.lastrowid

    def mark_delivered(self, reminder_id: int) -> bool:
        """Отметить как показанное (ждём подтверждения пользователя)."""
        with self._lock:
            self._db.execute(
                'UPDATE reminders SET delivered=1 WHERE id=?', (reminder_id,))
            self._db.commit()
        return True

    def delete_reminder(self, reminder_id: int) -> bool:
        """Удалить конкретное напоминание (по id)."""
        with self._lock:
            self._db.execute('DELETE FROM reminders WHERE id=?', (reminder_id,))
            self._db.commit()
        return True

    def confirm_all_delivered(self, person_id: int) -> int:
        """Удалить показанные (delivered=1) напоминания для этого человека.
        Возвращает кол-во удалённых."""
        with self._lock:
            cur = self._db.execute(
                'DELETE FROM reminders WHERE person_id=? AND delivered=1',
                (person_id,))
            self._db.commit()
        return cur.rowcount

    # ── Запросы ───────────────────────────────────────────────────────────

    def get_due(self, person_id: int, today: str,
                now_time: str | None = None,
                default_time: str = '07:00') -> list[dict]:
        """Напоминания для показа: trigger_date IS NULL или (trigger_date < today)
        или (trigger_date = today И effective_time <= now_time).

        now_time: текущее время 'HH:MM'. Если None — используется '23:59' (доставить всё за сегодня).
        default_time: время по умолчанию когда trigger_time IS NULL (обычно '07:00').
        """
        if now_time is None:
            now_time = '23:59'
        with self._lock:
            rows = self._db.execute(
                '''SELECT id, person_id, person_name, trigger_date, trigger_time,
                          message, source, delivered, created_at
                   FROM reminders
                   WHERE person_id = ? AND delivered = 0
                     AND (
                       trigger_date IS NULL
                       OR trigger_date < ?
                       OR (trigger_date = ? AND COALESCE(trigger_time, ?) <= ?)
                     )
                   ORDER BY (trigger_date IS NULL) DESC, trigger_date ASC,
                             COALESCE(trigger_time, ?) ASC''',
                (person_id, today, today, default_time, now_time, default_time),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_reminders(self, person_id: int | None = None) -> list[dict]:
        """Все напоминания — для конкретного человека или для всех."""
        with self._lock:
            if person_id is not None:
                rows = self._db.execute(
                    'SELECT id, person_id, person_name, trigger_date, trigger_time, '
                    'message, source, delivered, created_at '
                    'FROM reminders WHERE person_id=? '
                    'ORDER BY trigger_date IS NULL DESC, trigger_date, '
                    'COALESCE(trigger_time, "07:00"), created_at',
                    (person_id,),
                ).fetchall()
            else:
                rows = self._db.execute(
                    'SELECT id, person_id, person_name, trigger_date, trigger_time, '
                    'message, source, delivered, created_at '
                    'FROM reminders '
                    'ORDER BY person_id, trigger_date IS NULL DESC, trigger_date, '
                    'COALESCE(trigger_time, "07:00"), created_at',
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ── Утилиты ───────────────────────────────────────────────────────────

    def _row_to_dict(self, row) -> dict:
        return {
            'id':           row[0],
            'person_id':    row[1],
            'person_name':  row[2],
            'trigger_date': row[3],
            'trigger_time': row[4],
            'message':      row[5],
            'source':       row[6],
            'delivered':    bool(row[7]),
            'created_at':   row[8],
        }

    def close(self):
        with self._lock:
            self._db.close()
