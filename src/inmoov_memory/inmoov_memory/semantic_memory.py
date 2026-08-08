"""
SemanticMemory — слой долговременной семантической памяти
==========================================================
Хранит факты двумя способами:
  - SQLite (facts): структурированные записи для точного поиска
  - ChromaDB:       векторный индекс для семантического поиска

При поиске объединяет результаты из обоих источников.
Доступ только через tool call (не попадает в системный промпт автоматически).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SemanticMemory:
    """Долговременная семантическая память: факты, предпочтения, события."""

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

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_chroma(self) -> None:
        """Инициализация ChromaDB. Если не установлен — работаем только на SQLite."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.chroma_path)
            self._chroma = client.get_or_create_collection(
                name="robot_memory",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB инициализирован: %s", self.chroma_path)
        except ImportError:
            logger.warning("chromadb не установлен — семантический поиск отключён (только SQLite)")
            self._chroma = None
        except Exception as exc:
            logger.warning("ChromaDB недоступен: %s", exc)
            self._chroma = None

    # ------------------------------------------------------------------
    # Запись фактов
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
        """Сохраняет или обновляет факт. Возвращает ID записи."""
        subject   = str(subject   or "").strip()
        predicate = str(predicate or "").strip()
        value     = str(value     or "").strip()
        category  = str(category  or "preference").strip()
        if not subject or not predicate or not value:
            logger.warning("save_fact: пустое поле — пропускаю (subj=%r pred=%r val=%r)",
                           subject, predicate, value)
            return -1
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            # Проверяем, существует ли запись (subject, predicate)
            existing = conn.execute(
                "SELECT id FROM facts WHERE subject=? AND predicate=?",
                (subject, predicate),
            ).fetchone()
            if existing:
                # Обновляем только value, confidence, source, updated_at — created_at не трогаем
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

        # Синхронизируем с ChromaDB
        if self._chroma is not None:
            doc_text = f"{subject} — {predicate}: {value}"
            # ID стабильный (не зависит от fact_id, который меняется при UPDATE)
            import hashlib
            chroma_id = hashlib.md5(f"{subject}::{predicate}".encode()).hexdigest()
            try:
                self._chroma.upsert(
                    ids=[chroma_id],
                    documents=[doc_text],
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

        logger.debug("Факт сохранён: %s — %s: %s", subject, predicate, value)
        return fact_id

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict]:
        """
        Семантический поиск:
          1. ChromaDB (если доступен) — векторное сходство
          2. SQLite fallback — LIKE по subject/predicate/value
        Результаты дедуплицируются и ограничиваются limit.
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
                logger.warning("ChromaDB поиск: %s", exc)

        # --- SQLite fallback / дополнение ---
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
        """Все факты о конкретном субъекте."""
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
    # Удаление
    # ------------------------------------------------------------------

    def delete_fact(self, subject: str, predicate: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM facts WHERE subject=? AND predicate=?",
                (subject, predicate),
            )
            return cur.rowcount > 0
