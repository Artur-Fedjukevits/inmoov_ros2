"""
Robot Memory Manager
====================
Управляет тремя слоями памяти робота:
  - WorkingMemory  : RAM, текущее состояние (время, место, режим)
  - EpisodicMemory : SQLite, краткосрочные эпизоды (диалоги, события)
  - SemanticMemory : ChromaDB + SQLite, долговременные факты и знания

Использование:
    mm = MemoryManager(db_path="memory.db", chroma_path="./chroma")
    system_prompt = mm.build_system_prompt()
    mm.after_conversation("Артур сказал что не любит громкую музыку")
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from inmoov_memory.working_memory import WorkingMemory
from inmoov_memory.episodic_memory import EpisodicMemory
from inmoov_memory.semantic_memory import SemanticMemory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persona — редактируй под своего робота
# ---------------------------------------------------------------------------
ROBOT_PERSONA = """Ты InMoov — гуманоидный робот-ассистент.
Ты дружелюбен, внимателен и стараешься быть полезным.
Отвечай на языке собеседника (русский или английский).
Не выдумывай факты, которых не знаешь — лучше честно скажи об этом."""


class MemoryManager:
    """Центральный менеджер памяти робота."""

    def __init__(
        self,
        db_path: str = "memory.db",
        chroma_path: str = "./chroma",
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:14b",
        semantic_db_path: Optional[str] = None,
    ) -> None:
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

        # Три слоя памяти (episodic и semantic могут использовать разные БД)
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(db_path=db_path)
        self.semantic = SemanticMemory(
            db_path=semantic_db_path or db_path,
            chroma_path=chroma_path,
        )

        logger.info(
            "MemoryManager инициализирован. episodic=%s semantic=%s chroma=%s",
            db_path, semantic_db_path or db_path, chroma_path,
        )

    # ------------------------------------------------------------------
    # Построение системного промпта (вызывается перед каждым LLM-запросом)
    # ------------------------------------------------------------------

    def build_system_prompt(self) -> str:
        """
        Собирает системный промпт из:
          - персонажа робота
          - рабочей памяти (время, место, состояние)
          - последних 5 эпизодов сегодняшнего дня
        """
        working_text = self.working.to_text()
        recent_text = self.episodic.get_recent_text(limit=5)

        parts = [
            ROBOT_PERSONA,
            "",
            "== Текущий момент ==",
            working_text,
        ]
        if recent_text:
            parts += ["", "== Что происходило сегодня ==", recent_text]

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Tool call-функции (регистрируются в LLM как tools)
    # ------------------------------------------------------------------

    def get_tool_definitions(self) -> list[dict]:
        """Возвращает описание инструментов для передачи в LLM (Ollama format)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": (
                        "Поиск фактов в долговременной (семантической) памяти. "
                        "Используй когда нужно вспомнить предпочтения, имена, события."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Запрос для семантического поиска",
                            },
                            "category": {
                                "type": "string",
                                "enum": ["person", "preference", "event", "rule", ""],
                                "description": "Фильтр по категории (опционально)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Максимум результатов (по умолчанию 5)",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "save_fact",
                    "description": (
                        "Сохранить новый факт в долговременную память. "
                        "Используй когда узнаёшь что-то важное о человеке или ситуации."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string", "description": "О ком/о чём факт"},
                            "predicate": {"type": "string", "description": "Свойство или отношение"},
                            "value": {"type": "string", "description": "Значение"},
                            "category": {
                                "type": "string",
                                "enum": ["person", "preference", "event", "rule"],
                            },
                        },
                        "required": ["subject", "predicate", "value", "category"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_episode",
                    "description": "Получить конкретный эпизод из краткосрочной памяти по дате или ключевому слову.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string", "description": "Ключевое слово для поиска"},
                            "date": {
                                "type": "string",
                                "description": "Дата в формате YYYY-MM-DD (опционально)",
                            },
                        },
                        "required": ["keyword"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_location",
                    "description": "Обновить текущее местоположение робота.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "room": {"type": "string"},
                            "landmark": {"type": "string"},
                        },
                        "required": ["room"],
                    },
                },
            },
        ]

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Диспетчер вызовов инструментов от LLM."""
        handlers = {
            "search_memory": self._tool_search_memory,
            "save_fact": self._tool_save_fact,
            "get_episode": self._tool_get_episode,
            "update_location": self._tool_update_location,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Неизвестный инструмент: {tool_name}"})
        try:
            return handler(**arguments)
        except Exception as exc:
            logger.exception("Ошибка в tool %s", tool_name)
            return json.dumps({"error": str(exc)})

    def _tool_search_memory(self, query: str, category: str = "", limit: int = 5) -> str:
        results = self.semantic.search(query, category=category or None, limit=limit)
        if not results:
            return json.dumps({"result": "Ничего не найдено в памяти."})
        return json.dumps({"facts": results})

    def _tool_save_fact(self, subject: str, predicate: str, value: str, category: str) -> str:
        self.semantic.save_fact(subject=subject, predicate=predicate, value=value, category=category)
        return json.dumps({"status": "ok", "message": f"Факт сохранён: {subject} — {predicate}: {value}"})

    def _tool_get_episode(self, keyword: str, date: Optional[str] = None) -> str:
        episodes = self.episodic.search(keyword=keyword, date=date)
        if not episodes:
            return json.dumps({"result": "Эпизоды не найдены."})
        return json.dumps({"episodes": episodes})

    def _tool_update_location(self, room: str, landmark: str = "") -> str:
        self.working.update_location(room=room, landmark=landmark)
        return json.dumps({"status": "ok", "location": room})

    # ------------------------------------------------------------------
    # Постобработка диалога (вызывается после каждого завершённого разговора)
    # ------------------------------------------------------------------

    def after_conversation(self, dialogue_text: str, participants: list[str] | None = None) -> None:
        """
        Сохраняет эпизод и извлекает факты из завершённого диалога.
        Использует Ollama для генерации резюме и извлечения фактов.
        """
        participants = participants or []

        # 1. Генерируем краткое резюме
        summary = self._summarize(dialogue_text)
        importance = self._score_importance(dialogue_text, summary)

        # 2. Сохраняем в эпизодическую память
        self.episodic.save(
            summary=summary,
            raw_text=dialogue_text,
            participants=participants,
            location=self.working.data["location"]["room"],
            importance=importance,
        )
        logger.info("Эпизод сохранён (importance=%.2f): %s", importance, summary[:60])

        # 3. Важные эпизоды → извлекаем факты в семантическую память
        if importance >= 0.6:
            facts = self._extract_facts(dialogue_text)
            saved = 0
            for fact in facts:
                try:
                    self.semantic.save_fact(**fact)
                    saved += 1
                except Exception as exc:
                    logger.warning("save_fact пропущен (%s): %s", fact, exc)
            logger.info("Извлечено %d фактов, сохранено %d", len(facts), saved)

    # ------------------------------------------------------------------
    # Вспомогательные LLM-вызовы (Ollama)
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str, system: str = "") -> str:
        """Синхронный вызов через /api/chat (совместим с llama.cpp и Ollama)."""
        import urllib.request

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0.1, "num_predict": 512},
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        parts = []
        with urllib.request.urlopen(req, timeout=30) as resp:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    parts.append(content)
                if chunk.get("done"):
                    break
        return "".join(parts).strip()

    def _summarize(self, dialogue_text: str) -> str:
        """Генерирует краткое (1–2 предложения) резюме диалога."""
        prompt = (
            f"Сделай краткое резюме (1-2 предложения) следующего диалога робота.\n"
            f"Отвечай только резюме, без лишних слов.\n\n{dialogue_text}"
        )
        try:
            return self._call_ollama(prompt)
        except Exception:
            # Fallback: первые 200 символов
            return dialogue_text[:200].replace("\n", " ")

    def _score_importance(self, text: str, summary: str) -> float:
        """
        Эвристическая оценка важности (0.0–1.0).
        Можно заменить на LLM-вызов для точности.
        """
        high_keywords = [
            "не нравится", "нравится", "люблю", "не люблю", "аллергия",
            "важно", "помни", "запомни", "зовут", "мой", "моя",
            "prefer", "hate", "love", "remember", "important", "name",
        ]
        text_lower = (text + " " + summary).lower()
        hits = sum(1 for kw in high_keywords if kw in text_lower)
        # базовая оценка + бонус за ключевые слова
        score = min(0.3 + hits * 0.15, 1.0)
        return round(score, 2)

    def _extract_facts(self, dialogue_text: str) -> list[dict]:
        """Извлекает структурированные факты из текста диалога через LLM."""
        system = (
            "Ты — система извлечения фактов. "
            "Отвечай ТОЛЬКО валидным JSON-массивом, без пояснений и markdown-блоков."
        )
        prompt = (
            "Извлеки все важные факты из этого диалога.\n"
            "Формат ответа — JSON-массив объектов:\n"
            '[{"subject":"...","predicate":"...","value":"...","category":"person|preference|event|rule"}]\n'
            "Если фактов нет — верни пустой массив [].\n\n"
            f"{dialogue_text}"
        )
        try:
            raw = self._call_ollama(prompt, system=system)
            # Убираем возможные markdown-обёртки
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            facts = json.loads(raw)
            return [
                f for f in facts
                if (isinstance(f, dict)
                    and str(f.get("subject",   "") or "").strip()
                    and str(f.get("predicate", "") or "").strip()
                    and str(f.get("value",     "") or "").strip()
                    and str(f.get("category",  "") or "").strip())
            ]
        except Exception as exc:
            logger.warning("Не удалось извлечь факты: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------

    def wipe_episodic(self, older_than_days: int = 30, min_importance: float = 0.0) -> int:
        """Удаляет старые малозначимые эпизоды. Возвращает кол-во удалённых."""
        return self.episodic.cleanup(older_than_days=older_than_days, max_importance=min_importance)

    def status(self) -> dict:
        """Краткая статистика состояния памяти."""
        return {
            "working": self.working.data,
            "episodic_count": self.episodic.count(),
            "semantic_count": self.semantic.count(),
        }
