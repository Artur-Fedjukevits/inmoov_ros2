"""
Robot Memory Manager
====================
Manages the robot's three memory layers:
  - WorkingMemory  : RAM, current state (time, location, mode)
  - EpisodicMemory : SQLite, short-term episodes (dialogues, events)
  - SemanticMemory : ChromaDB + SQLite, long-term facts and knowledge

Usage:
    mm = MemoryManager(db_path="memory.db", chroma_path="./chroma")
    system_prompt = mm.build_system_prompt()
    mm.after_conversation("Artur said he does not like loud music")

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
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
# Persona — edit to match your robot
# ---------------------------------------------------------------------------
ROBOT_PERSONA = """Ты InMoov — гуманоидный робот-ассистент.
Ты дружелюбен, внимателен и стараешься быть полезным.
Отвечай на языке собеседника (русский или английский).
Не выдумывай факты, которых не знаешь — лучше честно скажи об этом."""


class MemoryManager:
    """The robot's central memory manager."""

    def __init__(
        self,
        db_path: str = "memory.db",
        chroma_path: str = "./chroma",
        llm_url: str = "http://192.168.10.118:18020/v1/chat/completions",
        llm_model: str = "qwen3.8-27b",
        bearer_token: str = "",
        semantic_db_path: Optional[str] = None,
    ) -> None:
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.bearer_token = bearer_token

        # Three memory layers (episodic and semantic may use different DBs)
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(db_path=db_path)
        self.semantic = SemanticMemory(
            db_path=semantic_db_path or db_path,
            chroma_path=chroma_path,
        )

        logger.info(
            "MemoryManager initialized. episodic=%s semantic=%s chroma=%s",
            db_path, semantic_db_path or db_path, chroma_path,
        )

    # ------------------------------------------------------------------
    # System prompt assembly (called before every LLM request)
    # ------------------------------------------------------------------

    def build_system_prompt(self) -> str:
        """
        Assembles the system prompt from:
          - the robot's persona
          - working memory (time, location, state)
          - the last 5 episodes from today
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
    # Tool-call functions (registered with the LLM as tools)
    # ------------------------------------------------------------------

    def get_tool_definitions(self) -> list[dict]:
        """Returns the tool descriptions to pass to the LLM (OpenAI tools format)."""
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
        """Dispatches tool calls coming from the LLM."""
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
            logger.exception("Error in tool %s", tool_name)
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
    # Post-processing a dialogue (called after every completed conversation)
    # ------------------------------------------------------------------

    def after_conversation(self, dialogue_text: str, participants: list[str] | None = None) -> None:
        """
        Saves an episode and extracts facts from a completed dialogue.
        Uses the LLM (OpenAI chat.completions API) to generate the summary and extract facts.
        """
        participants = participants or []

        # 1. Generate a short summary
        summary = self._summarize(dialogue_text)
        importance = self._score_importance(dialogue_text, summary)

        # 2. Save to episodic memory
        self.episodic.save(
            summary=summary,
            raw_text=dialogue_text,
            participants=participants,
            location=self.working.data["location"]["room"],
            importance=importance,
        )
        logger.info("Episode saved (importance=%.2f): %s", importance, summary[:60])

        # 3. Important episodes → extract facts into semantic memory
        if importance >= 0.6:
            facts = self._extract_facts(dialogue_text)
            saved = 0
            for fact in facts:
                try:
                    self.semantic.save_fact(**fact)
                    saved += 1
                except Exception as exc:
                    logger.warning("save_fact skipped (%s): %s", fact, exc)
            logger.info("Extracted %d facts, saved %d", len(facts), saved)

    # ------------------------------------------------------------------
    # Helper LLM calls (OpenAI chat.completions API — vLLM)
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, system: str = "") -> str:
        """Synchronous, non-streaming call to an OpenAI-compatible chat.completions endpoint."""
        import urllib.request

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False,
            "temperature": 0.1,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        req = urllib.request.Request(self.llm_url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        choices = result.get("choices") or []
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        return content.strip()

    def _summarize(self, dialogue_text: str) -> str:
        """Generates a short (1-2 sentence) summary of the dialogue."""
        prompt = (
            f"Сделай краткое резюме (1-2 предложения) следующего диалога робота.\n"
            f"Отвечай только резюме, без лишних слов.\n\n{dialogue_text}"
        )
        try:
            return self._call_llm(prompt)
        except Exception:
            # Fallback: first 200 characters
            return dialogue_text[:200].replace("\n", " ")

    def _score_importance(self, text: str, summary: str) -> float:
        """
        Heuristic importance score (0.0-1.0).
        Could be replaced with an LLM call for better accuracy.
        """
        high_keywords = [
            "не нравится", "нравится", "люблю", "не люблю", "аллергия",
            "важно", "помни", "запомни", "зовут", "мой", "моя",
            "prefer", "hate", "love", "remember", "important", "name",
        ]
        text_lower = (text + " " + summary).lower()
        hits = sum(1 for kw in high_keywords if kw in text_lower)
        # base score + bonus per keyword hit
        score = min(0.3 + hits * 0.15, 1.0)
        return round(score, 2)

    def _extract_facts(self, dialogue_text: str) -> list[dict]:
        """Extracts structured facts from the dialogue text via the LLM."""
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
            raw = self._call_llm(prompt, system=system)
            # Strip possible markdown fences
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
            logger.warning("Failed to extract facts: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def wipe_episodic(self, older_than_days: int = 30, min_importance: float = 0.0) -> int:
        """Deletes old, low-importance episodes. Returns the number deleted."""
        return self.episodic.cleanup(older_than_days=older_than_days, max_importance=min_importance)

    def status(self) -> dict:
        """Brief memory status summary."""
        return {
            "working": self.working.data,
            "episodic_count": self.episodic.count(),
            "semantic_count": self.semantic.count(),
        }
