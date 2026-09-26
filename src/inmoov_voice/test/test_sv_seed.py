"""
test_sv_seed.py — phrase-level speaker verification (no hardware, no models).

_sv_check_phrase() runs on a stub with fake embeddings:
  - the owner's phrases pass, a TV / other person's phrase is dropped
  - introductions are accepted unchecked and the new voice becomes the reference
  - short phrases (a name, «да») aren't judged
  - an unconfirmed live seed made of noise is replaced; a confirmed one isn't

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


OWNER, JUNK, TV, GUEST = _vec(1), _vec(2), _vec(3), _vec(4)


def _near(v, seed, noise=0.5):
    u = v + noise * _vec(100 + seed)
    return u / np.linalg.norm(u)


def _stub():
    s = types.SimpleNamespace(
        rate=RATE, _sv_enabled=True, _sv_encoder=object(), _sv_threshold=0.35,
        _sv_min_speech_sec=1.2, _sv_debug_dir='', _SV_GALLERY_MAX=10,
        _sv_gallery=[], _sv_gallery_times=[], _sv_last_gallery_add=0.0,
        _SV_GALLERY_ADD_INTERVAL=0.0, _introducing=False, _sv_anchor_person_id=None,
        _sv_seed_unconfirmed=False, _sv_seed_rejects=0, _SV_SEED_MAX_REJECTS=2,
        get_logger=lambda: types.SimpleNamespace(info=lambda *a: None, warn=lambda *a: None,
                                                 debug=lambda *a: None),
        published=[], _next=None)
    s._publish_voice_emb = lambda emb, ts=None: s.published.append(emb)
    for name in ('_sv_seed', '_sv_sim', '_sv_add_to_gallery', '_sv_reset_reference',
                 '_sv_debug_save', '_sv_check_phrase'):
        setattr(s, name, getattr(VoiceDetectorNode, name).__get__(s))
    s._sv_embed = lambda seg: s._next
    return s


def _phrase(s, emb, sec=2.0):
    s._next = emb
    return s._sv_check_phrase(np.zeros(int(sec * RATE), np.float32))[0]


def test_owner_passes_tv_is_dropped():
    s = _stub()
    s._sv_gallery, s._sv_gallery_times = [OWNER], [0.0]            # DB anchor
    assert all(_phrase(s, _near(OWNER, k)) for k in range(5))
    assert not any(_phrase(s, _near(TV, k)) for k in range(5))


def test_introduction_accepts_the_new_persons_answer():
    s = _stub()
    s._sv_gallery, s._sv_gallery_times = [OWNER], [0.0]            # someone else before
    s._sv_reset_reference()                                        # what _introducing_cb does
    s._introducing = True
    assert _phrase(s, _near(GUEST, 1), sec=0.6)                    # «Меня зовут Ника»
    assert _phrase(s, _near(GUEST, 2))
    s._introducing = False
    assert _phrase(s, _near(GUEST, 3))                             # the guest is now the reference
    assert not _phrase(s, _near(TV, 1))


def test_short_phrases_are_not_judged():
    s = _stub()
    s._sv_gallery, s._sv_gallery_times = [OWNER], [0.0]
    assert _phrase(s, TV, sec=0.8)


def test_noise_seed_is_replaced_by_the_owner():
    s = _stub()
    assert _phrase(s, JUNK)                                        # seeds (unconfirmed)
    assert not _phrase(s, _near(OWNER, 1))                         # 1st rejection
    assert _phrase(s, _near(OWNER, 2))                             # 2nd → re-seed, accepted
    assert _phrase(s, _near(OWNER, 3))                             # matches → confirmed
    assert not s._sv_seed_unconfirmed


def test_confirmed_seed_is_not_evicted_by_a_tv():
    s = _stub()
    assert _phrase(s, OWNER)
    assert _phrase(s, _near(OWNER, 1))                             # confirmed
    assert not any(_phrase(s, _near(TV, k)) for k in range(6))
    assert _phrase(s, _near(OWNER, 9))
