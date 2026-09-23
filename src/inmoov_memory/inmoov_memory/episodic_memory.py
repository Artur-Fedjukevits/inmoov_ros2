"""
EpisodicMemory — episodic memory layer
========================================
Stores short-term events and dialogues in SQLite.
Sliding window: low-importance episodes (<= 0.5) older than 7 days are purged
by memory_node's periodic cleanup(); important episodes are kept.
Facts from important episodes are extracted into semantic memory
(the episode is then marked migrated=1).

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class EpisodicMemory:
    """Short-term episodic memory backed by SQLite."""

    CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS episodes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT    NOT NULL,
        date        TEXT    NOT NULL,      -- YYYY-MM-DD for fast filtering
        type        TEXT    NOT NULL DEFAULT 'conversation',
        participants TEXT   NOT NULL DEFAULT '[]',  -- JSON array of names
        summary     TEXT    NOT NULL,
        raw_text    TEXT,
        location    TEXT,
        emotion_tag TEXT    NOT NULL DEFAULT 'neutral',
        importance  REAL    NOT NULL DEFAULT 0.3,
        migrated    INTEGER NOT NULL DEFAULT 0  -- 1 = migrated to semantic
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
    # Writing
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
        """Saves a new episode. Returns its ID."""
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
    # Reading
    # ------------------------------------------------------------------

    def get_recent(self, limit: int = 5, date: Optional[str] = None) -> list[dict]:
        """Last N episodes. If date is given — only that day; otherwise the latest overall."""
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
        """Formatted text for insertion into the system prompt."""
        episodes = self.get_recent(limit=limit)
        if not episodes:
            return ""
        lines = []
        for ep in reversed(episodes):  # chronological order
            ts = ep["timestamp"][:16]  # YYYY-MM-DDTHH:MM → take date+time
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
        """Full-text search across summary and raw_text."""
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
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup(self, older_than_days: int = 7, max_importance: float = 0.5) -> int:
        """Deletes old, low-importance episodes. Returns the number deleted.

        Not restricted to migrated=1: low-importance episodes are below the fact
        extraction threshold, so they never get migrated and would never be purged.
        """
        cutoff = (datetime.now() - timedelta(days=older_than_days)).strftime("%Y-%m-%d")
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM episodes WHERE date < ? AND importance <= ?",
                (cutoff, max_importance),
            )
            return cur.rowcount

    def get_unmigrated_important(self, min_importance: float = 0.7) -> list[dict]:
        """High-importance episodes not yet migrated to semantic memory."""
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
