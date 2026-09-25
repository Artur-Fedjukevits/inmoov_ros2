"""
SemanticMemory — long-term semantic memory layer
==================================================
Stores facts in two ways:
  - SQLite (facts): structured records for exact lookup
  - ChromaDB:       vector index for semantic search

Search combines results from both sources.
Accessible only via tool call (not injected into the system prompt automatically).

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from inmoov_memory.sqlite_util import session

logger = logging.getLogger(__name__)


def _chroma_id(subject: str, predicate: str) -> str:
    """Stable Chroma ID for a fact (independent of fact_id, which changes on UPDATE)."""
    return hashlib.md5(f"{subject}::{predicate}".encode()).hexdigest()


def _chroma_doc(subject: str, predicate: str, value: str) -> str:
    return f"{subject} — {predicate}: {value}"


class SemanticMemory:
    """Long-term semantic memory: facts, preferences, events."""

    CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS facts (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        category     TEXT    NOT NULL,   -- person | preference | event | rule
        subject      TEXT    NOT NULL,
        predicate    TEXT    NOT NULL,
        value        TEXT    NOT NULL,
        confidence   REAL    NOT NULL DEFAULT 1.0,
        source       TEXT    NOT NULL DEFAULT 'conversation',
        created_at   TEXT    NOT NULL,
        updated_at   TEXT    NOT NULL,
        UNIQUE(subject, predicate)
    );
    CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
    CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
    """

    def __init__(self, db_path: str = "memory.db", chroma_path: str = "./chroma") -> None:
        self.db_path = db_path
        self.chroma_path = chroma_path
        self._chroma = None
        self._init_db()
        self._init_chroma()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(self.CREATE_SQL)

    def _conn(self):
        """Short-lived connection (WAL, busy_timeout): commit + close on exit."""
        return session(self.db_path, row_factory=sqlite3.Row)

    def _init_chroma(self) -> None:
        """Initializes ChromaDB. If it isn't installed — falls back to SQLite only."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.chroma_path)
            self._chroma = client.get_or_create_collection(
                name="robot_memory",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB initialized: %s", self.chroma_path)
        except ImportError:
            logger.warning("chromadb not installed — semantic search disabled (SQLite only)")
            self._chroma = None
        except Exception as exc:
            logger.warning("ChromaDB unavailable: %s", exc)
            self._chroma = None

    # ------------------------------------------------------------------
    # Writing facts
    # ------------------------------------------------------------------

    def save_fact(
        self,
        subject: str,
        predicate: str,
        value: str,
        category: str = "preference",
        confidence: float = 1.0,
        source: str = "conversation",
    ) -> int:
        """Saves or updates a fact. Returns the record ID."""
        subject   = str(subject   or "").strip()
        predicate = str(predicate or "").strip()
        value     = str(value     or "").strip()
        category  = str(category  or "preference").strip()
        if not subject or not predicate or not value:
            logger.warning("save_fact: empty field — skipping (subj=%r pred=%r val=%r)",
                           subject, predicate, value)
            return -1
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            # Check whether a record already exists for (subject, predicate)
            existing = conn.execute(
                "SELECT id FROM facts WHERE subject=? AND predicate=?",
                (subject, predicate),
            ).fetchone()
            if existing:
                # Update only value, confidence, source, updated_at — leave created_at untouched
                conn.execute(
                    """UPDATE facts SET value=?, confidence=?, source=?, category=?, updated_at=?
                       WHERE subject=? AND predicate=?""",
                    (value, confidence, source, category, now, subject, predicate),
                )
                fact_id = existing[0]
            else:
                cur = conn.execute(
                    """INSERT INTO facts
                       (category, subject, predicate, value, confidence, source, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (category, subject, predicate, value, confidence, source, now, now),
                )
                fact_id = cur.lastrowid

        # Sync with ChromaDB
        if self._chroma is not None:
            try:
                self._chroma.upsert(
                    ids=[_chroma_id(subject, predicate)],
                    documents=[_chroma_doc(subject, predicate, value)],
                    metadatas=[{
                        "category": category,
                        "subject": subject,
                        "predicate": predicate,
                        "value": value,
                        "fact_id": fact_id,
                    }],
                )
            except Exception as exc:
                logger.warning("ChromaDB upsert error: %s", exc)

        logger.debug("Fact saved: %s — %s: %s", subject, predicate, value)
        return fact_id

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict]:
        """
        Semantic search:
          1. ChromaDB (if available) — vector similarity
          2. SQLite fallback — LIKE over subject/predicate/value
        Results are deduplicated and capped at limit.
        """
        results: list[dict] = []

        # --- ChromaDB ---
        if self._chroma is not None:
            try:
                where = {"category": category} if category else None
                chroma_res = self._chroma.query(
                    query_texts=[query],
                    n_results=min(limit, self._chroma.count() or 1),
                    where=where,
                )
                for meta, dist in zip(
                    chroma_res["metadatas"][0],
                    chroma_res["distances"][0],
                ):
                    results.append({
                        "subject": meta.get("subject", ""),
                        "predicate": meta.get("predicate", ""),
                        "value": meta.get("value", ""),
                        "category": meta.get("category", ""),
                        "score": round(1.0 - dist, 3),
                        "source": "vector",
                    })
            except Exception as exc:
                logger.warning("ChromaDB search: %s", exc)

        # --- SQLite fallback / supplement ---
        if len(results) < limit:
            like = f"%{query}%"
            with self._conn() as conn:
                if category:
                    rows = conn.execute(
                        """SELECT subject, predicate, value, category, confidence, source
                           FROM facts
                           WHERE category = ? AND (
                               subject LIKE ? OR predicate LIKE ? OR value LIKE ?
                           )
                           ORDER BY updated_at DESC LIMIT ?""",
                        (category, like, like, like, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT subject, predicate, value, category, confidence, source
                           FROM facts
                           WHERE subject LIKE ? OR predicate LIKE ? OR value LIKE ?
                           ORDER BY updated_at DESC LIMIT ?""",
                        (like, like, like, limit),
                    ).fetchall()

            seen = {(r["subject"], r["predicate"]) for r in results}
            for row in rows:
                key = (row["subject"], row["predicate"])
                if key not in seen:
                    results.append({
                        "subject": row["subject"],
                        "predicate": row["predicate"],
                        "value": row["value"],
                        "category": row["category"],
                        "score": row["confidence"],
                        "source": "sqlite",
                    })
                    seen.add(key)

        return results[:limit]

    def get_by_subject(self, subject: str) -> list[dict]:
        """All facts about a specific subject."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM facts WHERE subject = ? ORDER BY category, predicate",
                (subject,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_fact(self, subject: str, predicate: str) -> bool:
        """Deletes the fact from SQLite and its vector from ChromaDB."""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM facts WHERE subject=? AND predicate=?",
                (subject, predicate),
            )
            deleted = cur.rowcount > 0
        if self._chroma is not None:
            try:
                self._chroma.delete(ids=[_chroma_id(subject, predicate)])
            except Exception as exc:
                logger.warning("ChromaDB delete failed (%s::%s): %s", subject, predicate, exc)
        return deleted

    # ------------------------------------------------------------------
    # SQLite → ChromaDB reconciliation
    # ------------------------------------------------------------------

    def reconcile_chroma(self) -> dict:
        """Brings the Chroma index in line with SQLite (the source of truth).

        save_fact/delete_fact write SQLite first and only log a Chroma failure, so
        the index can miss facts, hold stale values or keep deleted facts. Facts
        whose vector is missing or whose metadata differs are re-upserted; Chroma
        IDs with no SQLite fact are deleted. Unchanged facts aren't re-embedded.
        """
        if self._chroma is None:
            return {"upserted": 0, "deleted": 0}
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, category, subject, predicate, value FROM facts").fetchall()
        wanted = {}
        for r in rows:
            wanted[_chroma_id(r["subject"], r["predicate"])] = {
                "category": r["category"], "subject": r["subject"],
                "predicate": r["predicate"], "value": r["value"], "fact_id": r["id"],
            }

        have = self._chroma.get(include=["metadatas"])
        indexed = dict(zip(have["ids"], have["metadatas"]))

        stale = [cid for cid, meta in wanted.items() if indexed.get(cid) != meta]
        extra = [cid for cid in indexed if cid not in wanted]

        for i in range(0, len(stale), 100):
            batch = stale[i:i + 100]
            self._chroma.upsert(
                ids=batch,
                documents=[_chroma_doc(wanted[c]["subject"], wanted[c]["predicate"],
                                       wanted[c]["value"]) for c in batch],
                metadatas=[wanted[c] for c in batch],
            )
        if extra:
            self._chroma.delete(ids=extra)
        return {"upserted": len(stale), "deleted": len(extra)}
