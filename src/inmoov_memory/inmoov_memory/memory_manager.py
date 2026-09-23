"""
Robot Memory Manager
====================
Manages the robot's three memory layers:
  - WorkingMemory  : RAM, current state (time, location, mode)
  - EpisodicMemory : SQLite, short-term episodes (dialogues, events)
  - SemanticMemory : ChromaDB + SQLite, long-term facts and knowledge

Usage (memory_node builds the LLM context itself, see _publish_memory_context):
    mm = MemoryManager(db_path="memory.db", chroma_path="./chroma")
    mm.after_conversation("Artur said he does not like loud music")

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from inmoov_memory.working_memory import WorkingMemory
from inmoov_memory.episodic_memory import EpisodicMemory
from inmoov_memory.semantic_memory import SemanticMemory

logger = logging.getLogger(__name__)


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```$", re.S)


def strip_code_fence(raw: str) -> str:
    """Removes a surrounding markdown code fence (```json ... ```), if any.

    (str.lstrip('```json') strips a *character set*, not a prefix, and could eat
    leading 'j'/'s'/'o'/'n' characters of the payload.)
    """
    raw = raw.strip()
    m = _FENCE_RE.match(raw)
    return m.group(1).strip() if m else raw


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
        episode_id = self.episodic.save(
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
            self.episodic.mark_migrated(episode_id)

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
            raw = strip_code_fence(raw)
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
