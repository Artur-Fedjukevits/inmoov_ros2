"""
test_dialogue_guards.py — llm_node dialogue policy without ROS or an LLM.

  - PendingUtterance: newest wins, expiry, take() clears
  - tool policy: parse/filter
  - ToolAuditLog: JSONL fields, ok flag, rotation, never raises
  - llm_node busy flow: a voice command during processing is queued and run
    after the current reply (no TTS cancel), dropped on /introducing

Run:
  cd /home/artur/ros2_ws && source install/setup.bash
  python3 -m pytest src/inmoov_cognition/test/test_dialogue_guards.py -v
"""

import json
import os
import sys
import threading
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inmoov_cognition.dialogue_guards import (  # noqa: E402
    PendingUtterance, ToolAuditLog, filter_tools, parse_tool_list)
from inmoov_cognition.llm_node import LLMNode, TOOLS  # noqa: E402
from std_msgs.msg import String  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


# ─────────────────────────────────────────────────────────────────────────────
# PendingUtterance

def test_pending_newest_wins_and_take_clears():
    p = PendingUtterance()
    assert p.put('раз') is False
    assert p.put('два') is True
    assert p.take() == 'два'
    assert p.take() is None


def test_pending_expires():
    clock = FakeClock()
    p = PendingUtterance(max_age_sec=30.0, clock=clock)
    p.put('старое')
    clock.t += 31
    assert p.take() is None


# ─────────────────────────────────────────────────────────────────────────────
# Tool policy

def test_parse_and_filter_tools():
    denied = parse_tool_list(' robot_control, ,merge_persons,')
    assert denied == {'robot_control', 'merge_persons'}
    names = {t['function']['name'] for t in filter_tools(TOOLS, denied)}
    assert 'robot_control' not in names and 'merge_persons' not in names
    assert 'get_weather' in names
    assert filter_tools(TOOLS, frozenset()) is TOOLS


def test_default_telegram_denylist_names_exist():
    """A typo in the default would silently allow the tool."""
    all_names = {t['function']['name'] for t in TOOLS}
    assert parse_tool_list('robot_control,look_direction,merge_persons') <= all_names


# ─────────────────────────────────────────────────────────────────────────────
# ToolAuditLog

def test_audit_log_records(tmp_path):
    path = tmp_path / 'sub' / 'audit.jsonl'
    log = ToolAuditLog(str(path))
    log.record('telegram', 'items_control', {'item': 'Lamp', 'state': 'ON'},
               {'success': True}, 0.123)
    log.record('voice', 'robot_control', {'action': 'head'}, {'error': 'boom'}, 0.0)
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert lines[0]['source'] == 'telegram' and lines[0]['tool'] == 'items_control'
    assert lines[0]['ok'] is True and lines[0]['ms'] == 123
    assert lines[0]['args'] == {'item': 'Lamp', 'state': 'ON'}
    assert lines[1]['ok'] is False


def test_audit_log_rotates(tmp_path):
    path = tmp_path / 'audit.jsonl'
    log = ToolAuditLog(str(path), max_bytes=400)
    for i in range(10):
        log.record('voice', 'get_weather', {'i': i}, {'success': True}, 0.0)
    assert (tmp_path / 'audit.jsonl.1').exists()
    assert path.stat().st_size <= 400


def test_audit_log_never_raises(tmp_path):
    log = ToolAuditLog(str(tmp_path / 'audit.jsonl'))
    os.chmod(tmp_path, 0o500)                  # read-only dir
    try:
        log.record('voice', 'x', {}, {}, 0.0)  # must not raise
    finally:
        os.chmod(tmp_path, 0o700)
    ToolAuditLog('').record('voice', 'x', {}, {}, 0.0)   # disabled


# ─────────────────────────────────────────────────────────────────────────────
# llm_node busy → queue → dispatch (methods run on a stub, no ROS graph)

class _Pub:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)


def _stub():
    queried = []
    s = types.SimpleNamespace(
        _lock=threading.Lock(), _introducing=False, _processing=False,
        _pending=PendingUtterance(), _direction_hint_pub=_Pub(), _tts_cancel_pub=_Pub(),
        _addressed_to_robot=lambda text: True,
        get_logger=lambda: types.SimpleNamespace(
            info=lambda *a, **k: None, debug=lambda *a, **k: None, warn=lambda *a, **k: None),
        _queried=queried,
    )
    s._query_llm = lambda text, **kw: queried.append(text)
    s._dispatch_pending = lambda: LLMNode._dispatch_pending(s)
    return s


def _run_threads_now(monkeypatch):
    """threading.Thread(...).start() → run the target inline (deterministic)."""
    class Inline:
        def __init__(self, target, args=(), kwargs=None, daemon=None):
            self._t, self._a, self._k = target, args, kwargs or {}

        def start(self):
            self._t(*self._a, **self._k)
    monkeypatch.setattr('inmoov_cognition.llm_node.threading.Thread', Inline)


def test_busy_command_is_queued_then_run(monkeypatch):
    _run_threads_now(monkeypatch)
    s = _stub()
    s._processing = True                                  # a reply is being generated
    LLMNode.command_callback(s, String(data='который час'))
    LLMNode.command_callback(s, String(data='какая погода'))   # newest wins
    assert s._queried == []

    cancels_before = len(s._tts_cancel_pub.msgs)
    s._processing = False                                 # what _query_llm's finally does…
    LLMNode._dispatch_pending(s)                          # …followed by this
    assert s._queried == ['какая погода']
    assert s._processing is True
    assert len(s._tts_cancel_pub.msgs) == cancels_before, 'queued reply must not cut the TTS'


def test_queued_command_dropped_when_introducing(monkeypatch):
    _run_threads_now(monkeypatch)
    s = _stub()
    s._processing = True
    LLMNode.command_callback(s, String(data='привет'))
    s._processing = False
    s._introducing = True
    LLMNode._dispatch_pending(s)
    assert s._queried == [] and s._processing is False
