"""
lip_activity.py
===============
Mouth-openness measurement and per-phrase lip activity / lip-audio sync, for
deciding who is speaking (active speaker detection). Pure numpy — used by
face_tracker_node, testable offline.

Landmarks: InsightFace 2d106det (buffalo_l). Mouth layout, checked on a gallery
photo 2026-10-01 (closed mouth: the inner pairs coincide):
  corners            52 (left), 61 (right)
  inner upper lip    66, 62, 70
  inner lower lip    54, 60, 57   (pairs 66-54, 62-60, 70-57 face each other)

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

from __future__ import annotations

import numpy as np

_CORNERS     = (52, 61)
_INNER_PAIRS = ((66, 54), (62, 60), (70, 57))


def mouth_openness(lm: np.ndarray) -> float | None:
    """Inner-lip gap / mouth width (≈0 closed, ~0.3-0.6 wide open).
    Width normalization makes it independent of face size/distance."""
    if lm is None or len(lm) < 106:
        return None
    width = float(np.linalg.norm(lm[_CORNERS[0]] - lm[_CORNERS[1]]))
    if width < 1e-3:
        return None
    gap = np.mean([np.linalg.norm(lm[a] - lm[b]) for a, b in _INNER_PAIRS])
    return float(gap / width)


def landmarks_bbox(lm: np.ndarray, margin: float = 0.15) -> list[float]:
    """Face box from the 106 landmarks (+margin) — the crop for the next frame,
    so the mouth is followed between the 1.5 Hz face detections."""
    x0, y0 = lm.min(axis=0)
    x1, y1 = lm.max(axis=0)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return [float(x0 - mx), float(y0 - my), float(x1 + mx), float(y1 + my)]


def _window(samples, t0: float, t1: float):
    ts = np.array([s[0] for s in samples if t0 <= s[0] <= t1], dtype=np.float64)
    vs = np.array([s[1] for s in samples if t0 <= s[0] <= t1], dtype=np.float64)
    return ts, vs


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else 0.0


# Verdict thresholds on `excess` (p90 openness during speech − median openness at
# rest around the phrase). Tuned on a 2026-10-03 recording (1280x960, face ~113 px):
# speaking 0.051 / 0.053 / 0.097, the same face silent during that audio ≤ 0.023.
# Only 3 positive phrases so far — revisit with more data (two people, off-screen speaker).
SPEAKING_EXCESS  = 0.035
SPEAKING_OPEN    = 0.20
SILENT_EXCESS    = 0.012
SILENT_OPEN      = 0.05
_OPEN_MARGIN     = 0.02   # "clearly open" = rest + this


def analyze_segment(samples, t_start: float, t_end: float, envelope: list[float],
                    env_hz: float, vad: list[int] | None = None,
                    speech_pad_sec: float = 0.3, rest_before_sec: float = 4.0,
                    rest_after_sec: float = 3.0, rest_gap_sec: float = 0.5,
                    max_lag_sec: float = 0.5, min_samples: int = 6) -> dict | None:
    """Did this face speak during this phrase.

    samples  — [(stamp, openness), ...] of this face (any time range)
    envelope — RMS loudness of the phrase, env_hz frames from t_start
    vad      — 1/0 speech flag per envelope frame (voice_detector); without it the
               whole phrase window counts as speech

    The mouth is compared with ITSELF at rest just before/after the speech: a
    speaker's mouth opens far wider while the sound is on. Syllable-level
    lip-audio correlation turned out useless at 14 fps (a silent face got the
    same 0.3-0.6 as the speaker), so `sync` is kept only as a diagnostic, on
    0.3 s-smoothed signals. The mouth also opens ~0.2-0.3 s before the sound —
    the speech window is padded and the rest windows keep a gap.

    Returns None if the face was not seen enough during the speech, else:
      n          — openness samples during speech
      coverage   — share of the speech window covered by samples (0..1)
      speech_sec — length of the speech window, s
      p90        — 90th percentile openness during speech
      rest       — median openness at rest around the phrase (None if unseen)
      excess     — p90 − rest: the main "this face is talking" signal
      open_frac  — share of speech samples with the mouth clearly open
      sync       — diagnostic: best correlation of smoothed openness vs loudness
      verdict    — 'speaking' | 'silent' | 'unknown'
    """
    env   = np.asarray(envelope, dtype=np.float64)
    env_t = t_start + (np.arange(len(env)) + 0.5) / env_hz
    s0, s1 = t_start, t_end
    if vad is not None and len(vad) == len(env) and any(vad):
        on = np.flatnonzero(np.asarray(vad) > 0)
        s0 = env_t[on[0]] - speech_pad_sec
        s1 = env_t[on[-1]] + speech_pad_sec

    ts, vs = _window(samples, s0, s1)
    if len(ts) < min_samples:
        return None

    _, rb = _window(samples, s0 - rest_before_sec, s0 - rest_gap_sec)
    _, ra = _window(samples, s1 + rest_gap_sec, s1 + rest_after_sec)
    rest_v = np.concatenate([rb, ra])
    rest = float(np.median(rest_v)) if len(rest_v) >= min_samples else None
    ref = rest if rest is not None else float(np.percentile(vs, 20))

    p90       = float(np.percentile(vs, 90))
    excess    = p90 - ref
    open_frac = float(np.mean(vs > ref + _OPEN_MARGIN))

    if excess >= SPEAKING_EXCESS and open_frac >= SPEAKING_OPEN:
        verdict = 'speaking'
    elif excess < SILENT_EXCESS and open_frac < SILENT_OPEN:
        verdict = 'silent'
    else:
        verdict = 'unknown'

    duration = max(s1 - s0, 1e-3)
    out = {
        'n':          int(len(ts)),
        'coverage':   round(float(min(1.0, (ts[-1] - ts[0]) / duration)), 2),
        'speech_sec': round(float(duration), 2),
        'p90':        round(p90, 3),
        'rest':       None if rest is None else round(rest, 3),
        'excess':     round(float(excess), 3),
        'open_frac':  round(open_frac, 2),
        'sync':       None,
        'verdict':    verdict,
    }

    # Diagnostic sync on 0.3 s-smoothed signals
    if len(env) >= 4:
        grid = np.arange(s0, s1, 1.0 / env_hz)
        if len(grid) >= 4:
            k   = max(1, int(round(0.3 * env_hz)))
            ker = np.ones(k) / k
            en  = np.convolve(np.interp(grid, env_t, np.log(env + 1e-4)), ker, mode='same')
            best = None
            for lag in np.arange(-max_lag_sec, max_lag_sec + 1e-9, 1.0 / env_hz):
                mo = np.convolve(np.interp(grid + lag, ts, vs), ker, mode='same')
                c = _corr(mo, en)
                best = c if best is None else max(best, c)
            out['sync'] = round(float(best), 2)
    return out
