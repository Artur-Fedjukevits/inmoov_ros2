"""
reminder_db.py — Reminder storage (SQLite)

Schema:
    reminders(id, person_id, person_name, trigger_date, trigger_time, message, source, delivered, created_at)

trigger_date:
    NULL  → shown at every meeting until the user confirms
    YYYY-MM-DD → shown starting from this date

trigger_time:
    NULL  → default_time is used when checking (normally 07:00)
    HH:MM → shown starting from this time on the trigger_date day

Lifecycle:
    added (delivered=0) → shown at a meeting (delivered=1) → confirmed by the user → deleted

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
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

    # ── Writing ────────────────────────────────────────────────────────────

    def add_reminder(self, person_id: int, person_name: str, message: str,
                     trigger_date: str | None = None,
                     trigger_time: str | None = None,
                     source: str = 'manual') -> int:
        """Create a reminder. trigger_date=None → at the next meeting.

        trigger_time: 'HH:MM' or None (default_time is used in get_due).

        Returns the id of the existing reminder if an exact duplicate already exists
        (same person_id, trigger_date, trigger_time and normalized text, delivered=0).
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
        """Mark as shown (awaiting user confirmation)."""
        with self._lock:
            self._db.execute(
                'UPDATE reminders SET delivered=1 WHERE id=?', (reminder_id,))
            self._db.commit()
        return True

    def delete_reminder(self, reminder_id: int) -> bool:
        """Delete a specific reminder (by id)."""
        with self._lock:
            self._db.execute('DELETE FROM reminders WHERE id=?', (reminder_id,))
            self._db.commit()
        return True

    def confirm_all_delivered(self, person_id: int) -> int:
        """Delete the shown (delivered=1) reminders for this person.
        Returns the number deleted."""
        with self._lock:
            cur = self._db.execute(
                'DELETE FROM reminders WHERE person_id=? AND delivered=1',
                (person_id,))
            self._db.commit()
        return cur.rowcount

    # ── Queries ───────────────────────────────────────────────────────────

    def get_due(self, person_id: int, today: str,
                now_time: str | None = None,
                default_time: str = '07:00') -> list[dict]:
        """Reminders to show: trigger_date IS NULL or (trigger_date < today)
        or (trigger_date = today AND effective_time <= now_time).

        now_time: current time 'HH:MM'. If None, '23:59' is used (deliver everything due today).
        default_time: default time when trigger_time IS NULL (normally '07:00').
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
        """All reminders — for a specific person or for everyone."""
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

    # ── Utilities ───────────────────────────────────────────────────────────

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
