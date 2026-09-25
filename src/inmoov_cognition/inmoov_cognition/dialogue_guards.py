"""
dialogue_guards.py — small, ROS-free pieces of llm_node's dialogue policy.

  PendingUtterance  — one-slot buffer for a voice command that arrives while the
                      LLM is busy (newest wins, expires after max_age_sec)
  parse_tool_list / filter_tools
                    — per-source tool policy (e.g. Telegram can't move the robot)
  ToolAuditLog      — append-only JSONL record of every tool call, size-rotated

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import json
import os
import threading
import time


class PendingUtterance:
    """Holds at most one utterance; put() replaces, take() returns it if still fresh."""

    def __init__(self, max_age_sec: float = 30.0, clock=time.monotonic):
        self._max_age = max_age_sec
        self._clock   = clock
        self._text    = None
        self._t       = 0.0
        self._lock    = threading.Lock()

    def put(self, text: str) -> bool:
        """Returns True if it replaced an older pending utterance."""
        with self._lock:
            replaced = self._text is not None
            self._text, self._t = text, self._clock()
            return replaced

    def take(self):
        """The pending utterance (and clears the slot), or None if empty/expired."""
        with self._lock:
            text, t = self._text, self._t
            self._text = None
        if text is None or self._clock() - t > self._max_age:
            return None
        return text

    def clear(self):
        with self._lock:
            self._text = None


def parse_tool_list(raw: str) -> frozenset:
    """'a, b,,c' → frozenset({'a', 'b', 'c'})."""
    return frozenset(t.strip() for t in (raw or '').split(',') if t.strip())


def filter_tools(tools: list, denied: frozenset) -> list:
    """OpenAI tool schemas minus the denied function names."""
    if not denied:
        return tools
    return [t for t in tools if t.get('function', {}).get('name') not in denied]


class ToolAuditLog:
    """One JSON line per tool call: ts, source, tool, args, ok, result (truncated), ms.

    Rotates to <path>.1 when the file exceeds max_bytes. Never raises — auditing
    must not break the dialogue.
    """

    def __init__(self, path: str, max_bytes: int = 5 * 1024 * 1024, result_chars: int = 500):
        self._path         = os.path.expanduser(path) if path else ''
        self._max_bytes    = max_bytes
        self._result_chars = result_chars
        self._lock         = threading.Lock()
        if self._path:
            os.makedirs(os.path.dirname(self._path) or '.', exist_ok=True)

    def record(self, source: str, tool: str, args: dict, result, duration_sec: float):
        if not self._path:
            return
        ok = not (isinstance(result, dict)
                  and (result.get('success') is False or 'error' in result))
        entry = {
            'ts':     time.strftime('%Y-%m-%dT%H:%M:%S'),
            'source': source,
            'tool':   tool,
            'args':   args,
            'ok':     ok,
            'result': json.dumps(result, ensure_ascii=False, default=str)[:self._result_chars],
            'ms':     int(duration_sec * 1000),
        }
        line = json.dumps(entry, ensure_ascii=False, default=str) + '\n'
        try:
            with self._lock:
                if (os.path.exists(self._path)
                        and os.path.getsize(self._path) + len(line) > self._max_bytes):
                    os.replace(self._path, self._path + '.1')
                with open(self._path, 'a', encoding='utf-8') as f:
                    f.write(line)
        except OSError:
            pass
