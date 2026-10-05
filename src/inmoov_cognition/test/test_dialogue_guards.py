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
        _split_at_name=lambda text: text,
        _lock=threading.Lock(), _introducing=False, _processing=False,
        _pending=PendingUtterance(), _direction_hint_pub=_Pub(), _tts_cancel_pub=_Pub(),
        _speech_addressed_pub=_Pub(), _introduce_on_address=False, _social_ctx_ts=0.0,
        _GAZE_CTX_STALE_SEC=LLMNode._GAZE_CTX_STALE_SEC,
        _addressing_reason=lambda text: 'gaze',
        get_logger=lambda: types.SimpleNamespace(
            info=lambda *a, **k: None, debug=lambda *a, **k: None, warn=lambda *a, **k: None),
        _queried=queried,
    )
    s._query_llm = lambda text, **kw: (queried.append(text), s._query_kw.append(kw))
    s._query_kw = []
    s._speaker_tag = lambda other: '[Говорит Герман]: ' if other else None
    s._lips_ev = None                       # what _await_lips returns (no lip data)
    s._await_lips = lambda t, sv_rejected=False: s._lips_ev
    s._other_speaker_check = lambda *a: LLMNode._other_speaker_check(s, *a)
    s._sv_confirm_pub = _Pub()
    s._gaze_lips_check = lambda text, t: LLMNode._gaze_lips_check(s, text, t)
    s._accept_command = lambda *a, **k: LLMNode._accept_command(s, *a, **k)
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


def test_unknown_face_addressing_starts_introduction(monkeypatch):
    """An unknown face waits silently; when it addresses the robot the utterance
    goes to identity_manager (/speech_addressed → introduction), not to the LLM."""
    _run_threads_now(monkeypatch)
    s = _stub()
    s._introduce_on_address = True
    s._social_ctx_ts = time.monotonic()
    s._addressing_reason = lambda text: 'name'
    LLMNode.command_callback(s, String(data='лёня, привет, ты кто'))
    assert s._queried == []
    assert [m.data for m in s._speech_addressed_pub.msgs] == ['лёня, привет, ты кто']

    s._addressing_reason = lambda text: 'gaze'     # gaze alone doesn't introduce a stranger
    LLMNode.command_callback(s, String(data='не мерить'))
    assert len(s._speech_addressed_pub.msgs) == 1 and s._queried == []
    s._lips_ev = {'who': 'primary'}                # …unless their lips moved with the speech
    LLMNode.command_callback(s, String(data='а ты кто такой'))
    assert len(s._speech_addressed_pub.msgs) == 2 and s._queried == []

    s._social_ctx_ts = time.monotonic() - 10.0     # stale context — normal dialogue
    LLMNode.command_callback(s, String(data='который час'))
    assert s._queried == ['который час']


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
    g = _gate_ns(looking, ctx_age, since_wake, now)
    g._has_robot_name = lambda text: LLMNode._has_robot_name(g, text)
    g._split_at_name = lambda text: LLMNode._split_at_name(g, text)
    return g


def _gate_ns(looking, ctx_age, since_wake, now):
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


# ─────────────────────────────────────────────────────────────────────────────
# identity_manager social policy: once-a-day greeting, introduce only when addressed
# ─────────────────────────────────────────────────────────────────────────────

def _identity_stub(tmp_path):
    from inmoov_cognition.identity_manager_node import IdentityManagerNode, State
    s = types.SimpleNamespace(
        _lock=threading.Lock(), _greet_log={}, _greet_day_start_hour=4,
        _greet_log_path=str(tmp_path / 'greet.json'),
        _state=State.INTERACTING, _current_person={}, _intro_declined=False,
        get_logger=lambda: types.SimpleNamespace(
            info=lambda *a, **k: None, warn=lambda *a, **k: None),
    )
    for m in ('_greet_day', '_load_greet_log', '_greeted_today', '_mark_greeted_today',
              '_speech_addressed_cb'):
        setattr(s, m, getattr(IdentityManagerNode, m).__get__(s))
    s.intros = []
    s._start_introduction = lambda: s.intros.append(1)
    return s, State


def test_greeting_once_a_day_survives_restart(tmp_path):
    s, _ = _identity_stub(tmp_path)
    assert not s._greeted_today(5)
    s._mark_greeted_today(5)
    assert s._greeted_today(5)
    assert not s._greeted_today(6)
    s2, _ = _identity_stub(tmp_path)               # node restarted
    s2._greet_log = s2._load_greet_log()
    assert s2._greeted_today(5)
    s2._greet_log['5'] = '2000-01-01'              # yesterday's greeting doesn't count
    assert not s2._greeted_today(5)


def test_unknown_face_introduced_only_when_addressed(tmp_path):
    s, State = _identity_stub(tmp_path)
    s._speech_addressed_cb(String(data='привет'))
    assert s.intros == [1]
    s._intro_declined = True                       # gave no name once — don't nag
    s._speech_addressed_cb(String(data='привет'))
    assert s.intros == [1]
    s._intro_declined = False
    s._current_person = {'person_id': 5, 'name': 'Артур'}   # known — LLM handles it
    s._speech_addressed_cb(String(data='привет'))
    assert s.intros == [1]


# ─────────────────────────────────────────────────────────────────────────────
# Speaker fusion (identity_manager, shadow mode)
# ─────────────────────────────────────────────────────────────────────────────

def _mouth(tracks, sv_rejected=False):
    return {'segment_id': 7, 'side': 'left', 't_start': 10.0, 't_end': 12.0,
            'sv_rejected': sv_rejected,
            'tracks': [{'track_id': tid, 'verdict': v, 'excess': 0.05} for tid, v in tracks]}


_LOOKING = [(10.0 + 0.2 * k, 1, 0.1) for k in range(10)]   # track 1 frontal all phrase


def _fuse(mouth, gaze=_LOOKING):
    from inmoov_cognition.identity_manager_node import fuse_speaker_evidence
    return fuse_speaker_evidence(mouth, 1, gaze, 0.30, 0.50)


def test_fusion_primary_speaking_and_looking_passes():
    ev = _fuse(_mouth([(1, 'speaking')]))
    assert ev['who'] == 'primary' and ev['gaze'] is True
    assert ev['gaze_gate'] is True and ev['fused_gate'] is True


def test_fusion_looking_but_silent_lips_blocks():
    # Live 2026-10-03 "Не мерить.": face looking at the robot, someone else talking
    ev = _fuse(_mouth([(1, 'silent')]))
    assert ev['who'] == 'offscreen'
    assert ev['gaze_gate'] is True and ev['fused_gate'] is False


def test_fusion_other_face_speaking():
    ev = _fuse(_mouth([(1, 'unknown'), (2, 'speaking')]))
    assert ev['who'] == 'other_face' and ev['others_speaking'] == [2]
    assert ev['fused_gate'] is False


def test_fusion_gaze_outside_phrase_ignored():
    stale = [(5.0, 1, 0.1), (5.2, 1, 0.1)]   # looked before the phrase, not during
    ev = _fuse(_mouth([(1, 'speaking')]), gaze=stale)
    assert ev['gaze'] is None and ev['gaze_gate'] is False and ev['fused_gate'] is False


def test_fusion_sv_rejected_has_no_gate():
    ev = _fuse(_mouth([(1, 'silent')], sv_rejected=True))
    assert ev['gaze_gate'] is None and ev['fused_gate'] is None


def test_fusion_unknown_lips_do_not_veto():
    # Live seg#3/#18: real questions with lips 'unknown' — gaze decides
    ev = _fuse(_mouth([(1, 'unknown')]))
    assert ev['who'] == 'unknown' and ev['fused_gate'] is True


def test_fusion_briefly_seen_faces_ignored():
    m = _mouth([(1, 'unknown'), (2, 'speaking')])
    m['tracks'][1]['coverage'] = 0.13            # live seg#13: flickering tracks
    ev = _fuse(m)
    assert ev['others_speaking'] == [] and ev['fused_gate'] is True


# ─────────────────────────────────────────────────────────────────────────────
# Multi-person talk: another speaker's phrase (SV-rejected) → voice_command_other

def test_other_speaker_needs_robot_name(monkeypatch):
    _run_threads_now(monkeypatch)
    s = _stub()
    s._addressing_reason = lambda text: 'gaze'      # interlocutor looks — irrelevant here
    LLMNode.command_callback(s, String(data='вот так и делаем'), other_speaker=True)
    assert s._queried == []

    s._addressing_reason = lambda text: 'name'
    LLMNode.command_callback(s, String(data='Лёня, а ты что думаешь?'), other_speaker=True)
    assert s._queried == ['Лёня, а ты что думаешь?']
    assert s._query_kw[-1]['other_speaker'] is True
    assert s._query_kw[-1]['speaker_tag'] == '[Говорит Герман]: '


def test_other_speaker_never_starts_introduction(monkeypatch):
    _run_threads_now(monkeypatch)
    s = _stub()
    s._introduce_on_address = True
    s._social_ctx_ts = time.monotonic()
    s._addressing_reason = lambda text: 'name'
    LLMNode.command_callback(s, String(data='Лёня, привет'), other_speaker=True)
    assert s._speech_addressed_pub.msgs == [] and s._queried == ['Лёня, привет']


def test_other_speaker_queued_keeps_its_tag(monkeypatch):
    _run_threads_now(monkeypatch)
    s = _stub()
    s._processing = True
    s._addressing_reason = lambda text: 'name'
    LLMNode.command_callback(s, String(data='Лёня, привет'), other_speaker=True)
    s._processing = False
    LLMNode._dispatch_pending(s)
    assert s._query_kw[-1] == {'speaker_tag': '[Говорит Герман]: ', 'other_speaker': True}


def _tag_stub(speaker, history=(), face_name=''):
    return types.SimpleNamespace(_lock=threading.Lock(), _speaker=speaker,
                                 history=list(history),
                                 _person_context={'name': face_name} if face_name else None)


def test_speaker_tag():
    now = time.time()
    german = {'ts': now, 'confidence': 'high', 'name': 'Герман', 'other_speaker': True}
    t = LLMNode._speaker_tag
    assert t(_tag_stub(german), True) == '[Говорит Герман]: '
    assert t(_tag_stub({}), True) == '[Говорит другой человек, голос не опознан]: '
    # one-on-one: the interlocutor's turns stay untagged…
    assert t(_tag_stub({}, face_name='Артур'), False) is None
    # …until someone else spoke in this history; another phrase's voice-id is not used
    hist = [{'role': 'user', 'content': '[Говорит Герман]: Лёня, привет'}]
    assert t(_tag_stub(german, hist, 'Артур'), False) == '[Говорит Артур]: '


def test_other_speaker_block_and_tag_cleanup():
    from inmoov_cognition.llm_node import _build_speaker_block, _clean_llm_text
    block = _build_speaker_block({'other_speaker': True, 'confidence': 'high',
                                  'name': 'Герман', 'face_name': 'Артур'})
    assert 'Герман' in block and 'НЕ твой текущий собеседник' in block and 'Артур' in block
    assert _clean_llm_text('[Говорит Лёня]: Привет, Герман!') == 'Привет, Герман!'


def test_split_at_name_marks_glued_talk_as_background():
    g = _gate(False)
    glued = ('Очень поздно будет все. Чипсы, лучше один раз чипсы. '
             'Это на следующий. Леня, ты знаешь, сколько калорий в чипсах?')
    out = LLMNode._split_at_name(g, glued)
    assert out.startswith('[Перед обращением звучал разговор, возможно не тебе: «Очень поздно')
    assert out.endswith('\nЛеня, ты знаешь, сколько калорий в чипсах?')
    assert LLMNode._split_at_name(g, 'Лёня, который час? Я опаздываю.') == \
        'Лёня, который час? Я опаздываю.'


# ─────────────────────────────────────────────────────────────────────────────
# Gaze vetoed by lips (/speaker_evidence)

def test_gaze_vetoed_when_face_in_view_was_silent(monkeypatch):
    # Live 2026-10-04: "You change one year." — the face in view looked, lips still
    _run_threads_now(monkeypatch)
    s = _stub()
    for who in ('offscreen', 'other_face'):
        s._lips_ev = {'who': who, 'lips': 'silent', 'excess': 0.003}
        LLMNode.command_callback(s, String(data='You change one year.'))
    assert s._queried == []
    for who in ('primary', 'unknown', None):    # 'unknown' lips / no data don't veto
        s._processing = False
        s._lips_ev = {'who': who} if who else None
        LLMNode.command_callback(s, String(data='который час'))
    assert s._queried == ['который час'] * 3


def test_name_never_waits_for_lips(monkeypatch):
    _run_threads_now(monkeypatch)
    s = _stub()
    s._addressing_reason = lambda text: 'name'
    s._lips_ev = {'who': 'offscreen'}
    LLMNode.command_callback(s, String(data='Лёня, который час'))
    assert s._queried == ['Лёня, который час']


def test_await_lips_matches_the_phrase_once():
    s = types.SimpleNamespace(_lips_cond=threading.Condition(), _lips_used_seg=None,
                              _LIPS_WAIT_SEC=0.05, _LIPS_MATCH_SEC=LLMNode._LIPS_MATCH_SEC)
    import collections
    s._lips_evidence = collections.deque([{'segment_id': 6, 't_end': 90.0, 'who': 'primary'},
                                          {'segment_id': 7, 't_end': 99.0, 'who': 'offscreen'}])
    assert LLMNode._await_lips(s, 100.0)['segment_id'] == 7
    assert LLMNode._await_lips(s, 100.0) is None       # already used; seg 6 too old
    # arrives a moment after the text (the usual order)

    def late():
        time.sleep(0.01)
        LLMNode._speaker_evidence_cb(s, String(data=json.dumps(
            {'segment_id': 8, 't_end': 100.5, 'who': 'primary'})))
    s._LIPS_WAIT_SEC = 1.0
    threading.Thread(target=late).start()
    assert LLMNode._await_lips(s, 101.0)['segment_id'] == 8


def test_sv_rejected_but_lips_say_interlocutor(monkeypatch):
    # Live 2026-10-05: the SV anchor stayed Artur's, Nicole in view spoke — her
    # phrases came as "another person". Lips + gaze override SV, voice is learned.
    _run_threads_now(monkeypatch)
    s = _stub()
    s._addressing_reason = lambda text: None
    s._lips_ev = {'who': 'primary', 'lips': 'speaking', 'gaze': True,
                  'excess': 0.08, 'segment_id': 21}
    LLMNode.command_callback(s, String(data='зачем ходить в школу'), other_speaker=True)
    assert s._queried == ['зачем ходить в школу']
    assert s._query_kw[-1]['other_speaker'] is False
    assert json.loads(s._sv_confirm_pub.msgs[-1].data) == {'segment_id': 21}

    # lips 'unknown' or not looking → still another person (needs the name)
    for ev in ({'who': 'unknown', 'lips': 'unknown', 'gaze': True},
               {'who': 'primary', 'lips': 'speaking', 'gaze': False}):
        s._lips_ev = ev
        s._processing = False
        LLMNode.command_callback(s, String(data='налево нет'), other_speaker=True)
    assert s._queried == ['зачем ходить в школу'] and len(s._sv_confirm_pub.msgs) == 1


def test_await_lips_keeps_rejected_and_accepted_apart():
    import collections
    s = types.SimpleNamespace(_lips_cond=threading.Condition(), _lips_used_seg=None,
                              _LIPS_WAIT_SEC=0.01, _LIPS_MATCH_SEC=LLMNode._LIPS_MATCH_SEC,
                              _lips_evidence=collections.deque(
                                  [{'segment_id': 3, 't_end': 99.0, 'sv_rejected': True}]))
    assert LLMNode._await_lips(s, 100.0) is None
    assert LLMNode._await_lips(s, 100.0, sv_rejected=True)['segment_id'] == 3


# ─────────────────────────────────────────────────────────────────────────────
# identity_manager: a voice goes to the person's DB gallery only if their lips spoke

def test_voice_saved_to_db_only_when_lips_confirm(monkeypatch):
    from inmoov_cognition.identity_manager_node import IdentityManagerNode as IM
    saved, logs = [], []
    monkeypatch.setattr('inmoov_cognition.identity_manager_node.threading.Thread',
                        lambda target, args=(), daemon=None: types.SimpleNamespace(
                            start=lambda: saved.append(args)))
    now = time.time()
    s = types.SimpleNamespace(
        _lock=threading.Lock(), _current_person={'person_id': 10},
        _session_voice_save_count=0, _session_voice_last_save_ts=0.0,
        _SV_SESSION_MAX=IM._SV_SESSION_MAX, _VOICE_SAVE_WAIT_SEC=IM._VOICE_SAVE_WAIT_SEC,
        _VOICE_SAVE_MATCH_SEC=IM._VOICE_SAVE_MATCH_SEC, _add_voice_to_gallery=None,
        get_logger=lambda: types.SimpleNamespace(info=logs.append))
    s._expire_voice_saves = lambda t: IM._expire_voice_saves(s, t)
    s._pending_voice_saves = [{'person_id': 10, 'emb': [1], 'ts': now - 1.0, 'added': now - 0.5}]
    # Live 2026-10-04: Herman spoke, Nicole (in frame) silent → not saved
    IM._resolve_voice_saves(s, {'t_end': now - 0.9, 'who': 'offscreen', 'lips': 'silent'})
    assert saved == [] and s._pending_voice_saves == []
    s._pending_voice_saves = [{'person_id': 10, 'emb': [2], 'ts': now - 1.0, 'added': now - 0.5}]
    IM._resolve_voice_saves(s, {'t_end': now - 0.9, 'who': 'primary', 'lips': 'speaking'})
    assert saved == [(10, [2], now - 1.0)] and s._session_voice_save_count == 1
    # no lip verdict in time → dropped
    s._pending_voice_saves = [{'person_id': 10, 'emb': [3], 'ts': now - 9, 'added': now - 9}]
    IM._expire_voice_saves(s, now)
    assert s._pending_voice_saves == [] and len(saved) == 1
