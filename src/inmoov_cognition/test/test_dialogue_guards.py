"""
test_dialogue_guards.py — llm_node dialogue policy without ROS or an LLM.

  - PendingUtterance: newest wins, expiry, take() clears
  - tool policy: parse/filter
  - ToolAuditLog: JSONL fields, ok flag, rotation, never raises
  - llm_node busy flow: a voice command during processing is queued and run
    after the current reply (no TTS cancel), dropped on /introducing

Run:
  cd ~/ros2_ws && source install/setup.bash
  python3 -m pytest src/inmoov_cognition/test/test_dialogue_guards.py -v
"""

import json
import os
import sys
import threading
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inmoov_cognition import llm_node  # noqa: E402
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
        _addressing_reason=lambda text: 'gaze',
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


# ─────────────────────────────────────────────────────────────────────────────
# Addressee gate: name anywhere / gaze now / wake word while no face

def _gate(looking=None, ctx_age=0.1, since_wake=1e6):
    now = time.monotonic()
    return types.SimpleNamespace(
        _lock=threading.Lock(), _looking_at_robot=looking,
        _social_ctx_ts=now - ctx_age, _wake_ts=now - since_wake, wake_grace_sec=30.0,
        _ROBOT_NAMES=LLMNode._ROBOT_NAMES,
        _ROBOT_NAMES_FIRST_WORD=LLMNode._ROBOT_NAMES_FIRST_WORD,
        _UNK_NAME_RE=LLMNode._UNK_NAME_RE,
        _GAZE_CTX_STALE_SEC=LLMNode._GAZE_CTX_STALE_SEC,
    )


def test_gate_name_anywhere():
    r = LLMNode._addressing_reason
    assert r(_gate(False), 'Ясно, Леня, спасибо давай пока.') == 'name'
    assert r(_gate(False), 'Лена, который час?') == 'name'
    assert r(_gate(False), 'скажи Лене, что я позвоню') == 'name'   # no Лена at home
    assert r(_gate(False), 'мне лень туда идти') is None
    # What Parakeet actually writes for "Лёня"
    for text in ('Ты глухой? Леона, ты глухой?', 'Л<unk>ня, посмотри налево.',
                 'Молодец, Ленин.', 'Лень, ты глухой?', 'Слушай, лення, напомни',
                 'Привет, Леоня!', 'Легин, сколько сейчас время?', 'Юля, повернись',
                 'Да, Лень, это я.'):
        assert r(_gate(False), text) == 'name', text
    assert r(_gate(False), 'позови Юлю ужинать') is None
    assert r(_gate(False), 'мне лень, давай завтра') is None
    assert r(_gate(False), 'там <unk> что-то') is None


def test_gate_gaze_must_be_true_and_fresh():
    r = LLMNode._addressing_reason
    assert r(_gate(True), 'который час') == 'gaze'
    assert r(_gate(False), 'который час') is None
    assert r(_gate(True, ctx_age=5.0), 'который час') is None   # identity_manager silent


def test_gate_wake_grace_only_without_face():
    r = LLMNode._addressing_reason
    assert r(_gate(None, since_wake=10), 'включи свет') == 'wake'
    assert r(_gate(None, since_wake=60), 'иди домой') is None     # walked away mid-dialogue
    assert r(_gate(False, since_wake=5), 'иди домой') is None     # face found, looking away


def test_social_ctx_none_overwrites_stale_true():
    s = types.SimpleNamespace(_lock=threading.Lock(), _looking_at_robot=True, _social_ctx_ts=0.0)
    LLMNode._social_context_cb(s, String(data=json.dumps({'looking_at_robot': None})))
    assert s._looking_at_robot is None


def test_ignore_marker_never_reaches_tts():
    sent = []
    s = types.SimpleNamespace(
        _lock=threading.Lock(), _voice_style={'emotion': ''}, _tg_req_id='',
        timeout_sec=1.0, _send_tts_chunk=lambda text, style: sent.append(text),
        get_logger=lambda: types.SimpleNamespace(
            info=lambda *a, **k: None, debug=lambda *a, **k: None, warn=lambda *a, **k: None),
    )
    deltas = ['[ig', 'nore', ']', ' Ладно, я тут ни при чём. Иду домой.']
    s._stream_llm = lambda payload, t: iter(
        [(d, False, None) for d in deltas] + [('', True, None)])
    content, _ = LLMNode._stream_with_tts(s, {})
    assert llm_node._is_not_addressed(content) and sent == []

    deltas = ['Сейчас десять часов вечера. ', 'Что-нибудь ещё?']
    content, _ = LLMNode._stream_with_tts(s, {})
    assert sent and not llm_node._is_not_addressed(content)


def test_addressing_block():
    assert llm_node._build_addressing_block(None) == ''
    assert '[ignore]' not in llm_node._build_addressing_block('name')
    assert '[ignore]' in llm_node._build_addressing_block('gaze')


# ─────────────────────────────────────────────────────────────────────────────
# telegram_bridge push ACK

def test_bridge_push_ack():
    from inmoov_cognition.telegram_bridge_node import TelegramBridgeNode
    pub = _Pub()
    stub = types.SimpleNamespace(_ack_pub=pub, _allowed_chat_id=0, _loop=None,
                                 get_logger=lambda: types.SimpleNamespace(warn=lambda *a: None))
    stub._ack = lambda data, ok, error='': TelegramBridgeNode._ack(stub, data, ok, error)

    TelegramBridgeNode._ack(stub, {'text': 'x', 'id': 'reminder:7'}, True)
    TelegramBridgeNode._ack(stub, {'text': 'no id'}, True)          # no id → no ACK
    assert [json.loads(m.data) for m in pub.msgs] == [{'id': 'reminder:7', 'ok': True}]

    # No chat configured → immediate negative ACK instead of a silent drop
    TelegramBridgeNode._telegram_push_cb(stub, String(data=json.dumps({'text': 'hi', 'id': 'reminder:8'})))
    assert json.loads(pub.msgs[-1].data) == {'id': 'reminder:8', 'ok': False,
                                             'error': 'no allowed_chat_id'}


# ─────────────────────────────────────────────────────────────────────────────
# One execution path for tools (review 2026-09-26: round 2 bypassed policy/audit)

def test_run_tool_enforces_policy_and_audits(tmp_path):
    executed = []
    stub = types.SimpleNamespace(
        _execute_tool=lambda fn, args: executed.append(fn) or {'success': True},
        _audit=ToolAuditLog(str(tmp_path / 'a.jsonl')),
        get_logger=lambda: types.SimpleNamespace(info=lambda *a: None))
    res, ok = LLMNode._run_tool(stub, 'items_control', {'name': 'Lamp'}, 'telegram',
                                frozenset({'items_control'}), 'R2')
    assert ok is False and res['success'] is False and executed == []
    res, ok = LLMNode._run_tool(stub, 'get_weather', {}, 'voice', frozenset(), 'R2')
    assert ok is True and executed == ['get_weather']
    lines = [json.loads(x) for x in (tmp_path / 'a.jsonl').read_text().splitlines()]
    assert [(e['tool'], e['ok']) for e in lines] == [('items_control', False), ('get_weather', True)]


def test_tools_execute_only_through_run_tool():
    """Structural guard: any new call site must go through _run_tool (policy + audit)."""
    import ast
    import inspect
    import inmoov_cognition.llm_node as m
    tree = ast.parse(inspect.getsource(m))
    offenders = []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name != '_run_tool':
            for n in ast.walk(fn):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr == '_execute_tool'):
                    offenders.append(f'{fn.name}:{n.lineno}')
    assert offenders == []
