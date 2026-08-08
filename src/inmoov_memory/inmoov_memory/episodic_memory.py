"""
EpisodicMemory — слой эпизодической памяти
============================================
Хранит краткосрочные события и диалоги в SQLite.
Скользящее окно: 7 дней / 200 записей.
Важные эпизоды мигрируют в семантическую память.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class EpisodicMemory:
    """Краткосрочная эпизодическая память на базе SQLite."""

    CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS episodes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT    NOT NULL,
        date        TEXT    NOT NULL,      -- YYYY-MM-DD для быстрой фильтрации
        type        TEXT    NOT NULL DEFAULT 'conversation',
        participants TEXT   NOT NULL DEFAULT '[]',  -- JSON-массив имён
        summary     TEXT    NOT NULL,
        raw_text    TEXT,
        location    TEXT,
        emotion_tag TEXT    NOT NULL DEFAULT 'neutral',
        importance  REAL    NOT NULL DEFAULT 0.3,
        migrated    INTEGER NOT NULL DEFAULT 0  -- 1 = перенесён в semantic
    );
    CREATE INDEX IF NOT EXISTS idx_ep_date ON episodes(date);
    CREATE INDEX IF NOT EXISTS idx_ep_importance ON episodes(importance);
    """

    def __init__(self, db_path: str = "memory.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(self.CREATE_SQL)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Запись
    # ------------------------------------------------------------------

    def save(
        self,
        summary: str,
        raw_text: str = "",
        participants: list[str] | None = None,
        location: str = "unknown",
        importance: float = 0.3,
        ep_type: str = "conversation",
        emotion_tag: str = "neutral",
    ) -> int:
        """Сохраняет новый эпизод. Возвращает ID."""
        now = datetime.now()
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO episodes
                   (timestamp, date, type, participants, summary, raw_text,
                    location, emotion_tag, importance)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now.isoformat(timespec="seconds"),
                    now.strftime("%Y-%m-%d"),
                    ep_type,
                    json.dumps(participants or [], ensure_ascii=False),
                    summary,
                    raw_text,
                    location,
                    emotion_tag,
                    importance,
                ),
            )
            return cur.lastrowid

    # ------------------------------------------------------------------
    # Чтение
    # ------------------------------------------------------------------

    def get_recent(self, limit: int = 5, date: Optional[str] = None) -> list[dict]:
        """Последние N эпизодов. Если date задан — только за этот день; иначе — любые последние."""
        with self._conn() as conn:
            if date:
                rows = conn.execute(
                    """SELECT timestamp, summary, participants, location, emotion_tag, importance
                       FROM episodes WHERE date = ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (date, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT timestamp, summary, participants, location, emotion_tag, importance
                       FROM episodes
                       ORDER BY timestamp DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_text(self, limit: int = 5) -> str:
        """Форматированный текст для вставки в системный промпт."""
        episodes = self.get_recent(limit=limit)
        if not episodes:
            return ""
        lines = []
        for ep in reversed(episodes):  # хронологический порядок
            ts = ep["timestamp"][:16]  # YYYY-MM-DDTHH:MM → берём дату+время
            date_part = ts[:10]
            time_part = ts[11:16]
            today = datetime.now().strftime("%Y-%m-%d")
            label = time_part if date_part == today else f"{date_part} {time_part}"
            people = json.loads(ep.get("participants") or "[]")
            ppl_str = f" [{', '.join(people)}]" if people else ""
            emotion = ep.get("emotion_tag", "neutral")
            emotion_str = f" ({emotion})" if emotion and emotion != "neutral" else ""
            lines.append(f"• {label}{ppl_str}{emotion_str} — {ep['summary']}")
        return "\n".join(lines)

    def search(self, keyword: str, date: Optional[str] = None, limit: int = 10) -> list[dict]:
        """Полнотекстовый поиск по summary и raw_text."""
        like = f"%{keyword}%"
        with self._conn() as conn:
            if date:
                rows = conn.execute(
                    """SELECT timestamp, date, summary, participants, location, importance
                       FROM episodes
                       WHERE date = ? AND (summary LIKE ? OR raw_text LIKE ?)
                       ORDER BY timestamp DESC LIMIT ?""",
                    (date, like, like, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT timestamp, date, summary, participants, location, importance
                       FROM episodes
                       WHERE summary LIKE ? OR raw_text LIKE ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (like, like, limit),
                ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    # ------------------------------------------------------------------
    # Обслуживание
    # ------------------------------------------------------------------

    def cleanup(self, older_than_days: int = 7, max_importance: float = 0.5) -> int:
        """Удаляет старые малозначимые эпизоды. Возвращает кол-во удалённых."""
        cutoff = (datetime.now() - timedelta(days=older_than_days)).strftime("%Y-%m-%d")
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM episodes WHERE date < ? AND importance <= ? AND migrated = 1",
                (cutoff, max_importance),
            )
            return cur.rowcount

    def get_unmigrated_important(self, min_importance: float = 0.7) -> list[dict]:
        """Эпизоды с высокой важностью, ещё не перенесённые в semantic."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT id, timestamp, summary, raw_text
                   FROM episodes
                   WHERE importance >= ? AND migrated = 0
                   ORDER BY importance DESC""",
                (min_importance,),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_migrated(self, episode_id: int) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE episodes SET migrated=1 WHERE id=?", (episode_id,))
