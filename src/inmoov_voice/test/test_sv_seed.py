"""
test_sv_seed.py — the live SV seed can't lock the owner out (no hardware, no models).

Without a DB anchor the first full segment seeds the session's voice gallery. A seed
made of noise / the robot's own voice used to reject everything after it
(live 2026-09-26, similarity ~0.05). _sv_decide() runs on a stub with fake embeddings.

Run:
  cd ~/ros2_ws && source install/setup.bash
  python3 -m pytest src/inmoov_voice/test/test_sv_seed.py -v
"""

import types

import numpy as np

from inmoov_voice.voice_detector_node import VoiceDetectorNode

RATE = 16000


def _vec(seed):
    v = np.random.default_rng(seed).standard_normal(192).astype(np.float32)
    return v / np.linalg.norm(v)


OWNER = _vec(1)
JUNK = _vec(2)
TV = _vec(3)


def _near(v, seed):
    u = v + 0.3 * _vec(100 + seed)
    return u / np.linalg.norm(u)


def _stub():
    s = types.SimpleNamespace(
        rate=RATE, _sv_seg_sec=1.0, _sv_threshold=0.55, _SV_GALLERY_MAX=10,
        _sv_gallery=[], _sv_gallery_times=[], _sv_last_gallery_add=0.0,
        _SV_GALLERY_ADD_INTERVAL=30.0, _introducing=False, _person_present=None,
        _sv_seed_unconfirmed=False, _sv_seed_rejects=0, _SV_SEED_MAX_REJECTS=3,
        _sv_buf=[], audio_buffer=[], speech_chunks=0, silence_counter=0,
        get_logger=lambda: types.SimpleNamespace(info=lambda *a: None, warn=lambda *a: None,
                                                 debug=lambda *a: None),
        _publish_voice_emb=lambda *a: None, _next=None)
    for name in ('_sv_seed', '_sv_sim', '_sv_threshold_for', '_sv_add_to_gallery'):
        setattr(s, name, getattr(VoiceDetectorNode, name).__get__(s))
    s._sv_embed = lambda seg: s._next
    return s


def _segment(s, emb):
    """One full 1 s segment with the given embedding → accepted?"""
    s._next = emb
    s._sv_buf = [np.zeros(RATE, np.float32)]
    return VoiceDetectorNode._sv_decide(s, log_reject=True)


def test_junk_seed_is_replaced_by_the_owner():
    s = _stub()
    assert _segment(s, JUNK)                              # seeds (unconfirmed)
    results = [_segment(s, _near(OWNER, k)) for k in range(4)]
    assert results[:2] == [False, False]                  # rejected against the junk…
    assert results[2] is True                             # …3rd in a row re-seeds
    assert results[3] is True                             # owner now matches, seed confirmed
    assert not s._sv_seed_unconfirmed
    assert float(max(g @ OWNER for g in s._sv_gallery)) > 0.8


def test_confirmed_owner_seed_is_not_evicted_by_a_tv():
    s = _stub()
    assert _segment(s, OWNER)
    assert _segment(s, _near(OWNER, 1))                   # confirmed
    assert not any(_segment(s, _near(TV, k)) for k in range(6))
    assert float(max(g @ OWNER for g in s._sv_gallery)) > 0.99


def test_db_anchor_is_never_reseeded():
    s = _stub()
    s._sv_gallery, s._sv_gallery_times = [OWNER], [0.0]   # as loaded by _voice_anchor_cb
    assert not any(_segment(s, _near(TV, k)) for k in range(6))
    assert s._sv_gallery[0] is OWNER
