#!/usr/bin/env python3
"""
identity_manager_node.py
========================
"Social Context Provider" — combines data from face_recognition
and emotion_recognition and publishes rich context for the Behavior Tree.

Does NOT send direct commands. Data only.

State machine:
  IDLE        — no face in frame
  RECOGNIZING — face present, waiting for identification
  INTERACTING — someone is in front of us (known, or an unknown face waiting
                silently), tracking emotion
  INTRODUCING — unknown person addressed the robot, collecting name via voice

Social policy (2026-10-03, "как у людей"):
  - a known person is greeted at most once per day (persisted in greet_log_path,
    survives IDLE and restarts); after that the robot only answers when addressed;
  - an unknown face is NOT introduced to proactively: the robot waits silently
    until that person addresses it (llm_node addressee gate → /speech_addressed),
    and only then asks the name.

Publishes:
  /social_context  (String JSON → behavior_manager Blackboard)
      Fields: person_present, person_id, name, is_known, emotion,
            should_greet, greet_text, introducing,
            introduce_pending, introduce_text, state
  /person_context  (String JSON → llm_node)
  /person_present  (Bool) — fast interrupt signal
  /introducing     (Bool → llm_node gate)
  /face_expression (String → face_expressions_node, facial mirroring)
  /vision/enable   (Bool, latched)
  /speaker_evidence (String JSON) — per phrase, who spoke (gaze + lips + SV)
      and whether gaze-only vs gaze+lips gates agree; see fuse_speaker_evidence

Subscribes:
  /face/identity  (String JSON)
  /face/emotion   (String JSON)
  /face/tracks    (String JSON)
  /voice_command  (String) — intercepted while in INTRODUCING state
  /speech_addressed (String) — llm_node: an unknown face addressed the robot → introduce
  /face/mouth_activity/{left,right} (String JSON) — face_tracker lip verdicts per phrase

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import collections
import datetime
import json
import os
import random
import re
import time
import threading

import numpy as np
import requests
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String, Bool
from inmoov_msgs.srv import MemoryQuery


def fuse_speaker_evidence(mouth: dict, primary_tid, gaze_samples,
                          yaw_threshold: float, frontal_fraction: float,
                          gaze_pad_sec: float = 0.5, min_coverage: float = 0.5,
                          strong_fraction: float = 0.8, strong_yaw: float = 0.15,
                          tail_sec: float = 1.5) -> dict:
    """Who spoke this phrase — gaze + lips + SV. llm_node vetoes a gaze-only
    address on who = offscreen / other_face (see its _gaze_lips_check).

    mouth        — /face/mouth_activity/{side} message (face_tracker, lip_activity)
    primary_tid  — identity_manager's primary track on that side
    gaze_samples — [(stamp, track_id, yaw_proxy[, pitch_proxy]), ...] from /face/tracks

    Gaze is taken over the phrase itself (± gaze_pad_sec), not "right now" as the
    llm_node gate does — by the time STT is done the head may have turned.

    who:
      'primary'    — the primary face's lips moved with the speech, no other face did
      'other_face' — another face in view spoke, the primary did not
      'offscreen'  — the primary face stayed silent and nobody in view spoke
      'unknown'    — not enough lip data / ambiguous
    gaze_gate  — today's llm_node rule (gaze alone), evaluated over the phrase
    fused_gate — proposed rule: gaze, VETOED by positive evidence that someone
                 else spoke (who = offscreen / other_face). 'unknown' lips don't
                 veto: requiring who='primary' rejected 2 of 8 real questions in
                 the live run 2026-10-03 (lips 'unknown', excess 0.016 / 0.043).
    Both gates are None for an SV-rejected phrase (it is gated by name only).
    gaze_strong — a confident look: frontal for >= strong_fraction of the phrase,
                 median yaw <= strong_yaw, and not looking away at its end
                 (the last tail_sec + pad — one finishes a question to the robot
                 looking at it). Live 2026-10-05 seg#27: a 7 s talk to someone else
                 passed the 0.5 gaze test at frac=0.61 yaw=0.20; real questions in
                 the log are frac 0.88-1.0, yaw 0.02-0.18. llm_node needs it to
                 override SV, and tells the LLM when the gaze was only weak.

    A face seen for less than min_coverage of the speech is ignored: with no rest
    window its reference is the speech itself, and briefly visible faces all came
    out 'speaking' (live seg#13: three tracks, coverage 0.08-0.17).
    """
    t0, t1 = mouth['t_start'], mouth['t_end']
    win  = [g for g in gaze_samples
            if g[1] == primary_tid and t0 - gaze_pad_sec <= g[0] <= t1 + gaze_pad_sec]
    yaws = [g[2] for g in win]
    pitches = [g[3] for g in win if len(g) > 3 and g[3] is not None]
    gaze_frac = (sum(1 for y in yaws if y < yaw_threshold) / len(yaws)) if len(yaws) >= 2 else None
    gaze = None if gaze_frac is None else gaze_frac >= frontal_fraction
    yaw_med = float(np.median(yaws)) if yaws else None
    tail = [g[2] for g in win if g[0] >= t1 - tail_sec]
    tail_frac = (sum(1 for y in tail if y < yaw_threshold) / len(tail)) if len(tail) >= 2 else None
    gaze_tail = None if tail_frac is None else tail_frac >= frontal_fraction
    gaze_strong = (gaze_frac is not None and gaze_frac >= strong_fraction
                   and yaw_med <= strong_yaw and gaze_tail is not False)

    by_tid  = {t['track_id']: t for t in mouth.get('tracks', [])
               if t.get('coverage', 1.0) >= min_coverage}
    primary = by_tid.get(primary_tid)
    lips    = primary['verdict'] if primary else None
    others  = sorted(tid for tid, t in by_tid.items()
                     if tid != primary_tid and t['verdict'] == 'speaking')

    if lips == 'speaking' and not others:
        who = 'primary'
    elif others and lips != 'speaking':
        who = 'other_face'
    elif lips == 'silent' and not others:
        who = 'offscreen'
    else:
        who = 'unknown'

    sv_rejected = bool(mouth.get('sv_rejected'))
    gaze_gate   = None if sv_rejected else bool(gaze)
    fused_gate  = None if sv_rejected else bool(gaze) and who not in ('offscreen', 'other_face')
    return {
        'segment_id':     mouth.get('segment_id'),
        't_end':          t1,
        'side':           mouth.get('side'),
        'primary_track':  primary_tid,
        'gaze_frac':      None if gaze_frac is None else round(gaze_frac, 2),
        'gaze_n':         len(yaws),
        # Diagnostics for tuning the gaze test (live 2026-10-03: "looking" at the
        # head's hardware limit while talking to someone else)
        'yaw_med':        None if yaw_med is None else round(yaw_med, 2),
        'pitch_med':      round(float(np.median(pitches)), 2) if pitches else None,
        'gaze':           gaze,
        'gaze_tail_frac': None if tail_frac is None else round(tail_frac, 2),
        'gaze_strong':    gaze_strong,
        'lips':           lips,
        'excess':         primary['excess'] if primary else None,
        'others_speaking': others,
        'sv_rejected':    sv_rejected,
        'who':            who,
        'gaze_gate':      gaze_gate,
        'fused_gate':     fused_gate,
    }


class State:
    IDLE        = 'idle'
    RECOGNIZING = 'recognizing'
    INTERACTING = 'interacting'
    INTRODUCING = 'introducing'


class IdentityManagerNode(LifecycleNode):
    _SV_SESSION_MAX = 3      # max number of voice recordings per INTERACTING session
    _SV_SESSION_GAP = 120.0  # minimum interval between recordings (sec)
    _GAZE_STALE_SEC = 1.5    # no primary-track kps for longer → gaze unknown (None)

    def __init__(self):
        super().__init__('identity_manager_node')

        # ── State ─────────────────────────────────────────────────────
        self._state           = State.IDLE
        self._primary_track   = None
        self._primary_embedding: list | None = None   # last track embedding
        self._enroll_embeddings: list[list] = []      # accumulated embeddings for enrollment
        self._enroll_max = 10                          # how many to collect before saving
        self._current_person  = {}
        self._last_face_time  = 0.0
        self._last_human_time = 0.0   # last signal from human_detection_node
        self._last_greet      = {}     # person_id → timestamp
        self._session_greeted = set()
        self._last_emotion    = None   # last recognized human emotion (for social_context)
        # Fusion: both sources must agree before changing facial expression
        self._face_emo_pend:  dict | None = None  # {'emotion', 'ts', 'confidence'}
        self._voice_emo_pend: dict | None = None  # {'emotion', 'ts', 'confidence'}
        self._FUSION_WINDOW = 15.0  # seconds within which both signals must match
        self._lock            = threading.Lock()

        self._introduce_last     = 0.0
        self._introduce_attempts = 0
        self._enrolled_track_id: int | None = None  # track locked BEFORE enrollment — ignore
        self._sleeping           = False   # sleep mode — PIR ignored
        self._face_hunt_since    = 0.0    # when we started waiting for a face while a body is present
        self._last_dialogue_ts   = 0.0    # last LLM response (dialogue active)
        self._voice_id_grace_ts  = 0.0    # voice identification from IDLE (watchdog grace)
        # Greet cooldown by name (independent of person_id — tracker may change id)
        self._greeted_names: dict[str, float] = {}   # name → timestamp
        self._greet_log: dict[str, str] = {}          # str(person_id) → day greeted (persisted)
        # The unknown face in this session already went through an introduction that
        # gave no name — don't ask again; the LLM just talks to them.
        self._intro_declined = False
        # After an OakD veto: block _tracks_cb until the next body signal.
        # This breaks the infinite "Hello/Goodbye" loop on face_detection false positives.
        self._waiting_for_body: bool = False
        # When the current IDLE → RECOGNIZING session started: the OakD veto counts
        # body absence from here, not from a body seen minutes before the session.
        self._session_start_ts: float = 0.0

        # Left-primary / right-fallback tracking: if the left camera is silent
        # longer than track_eye_fallback_sec — switch to the right one.
        self._left_track_last_msg: float = 0.0   # time of the last message from the left camera
        self._tracks_eye: str = 'left'            # currently active eye

        # Gaze: whether the interlocutor is looking the robot in the eyes.
        # Computed from InsightFace kps — nose/eye asymmetry (yaw-proxy).
        # A sliding window of the last 15 values is kept (~1.5s @ 10Hz detection).
        self._frontal_scores: collections.deque = collections.deque(maxlen=15)
        self._looking_at_robot: bool = True  # default True until data is available
        # When the primary track last gave kps. Older than _GAZE_STALE_SEC → the face
        # is gone: /social_context reports looking_at_robot=None instead of the last
        # verdict (a stale True kept the llm_node addressee gate open after leaving).
        self._gaze_ts: float = 0.0
        # (frame stamp, track_id, yaw_proxy) of the primary track — for the speaker
        # fusion, which looks at the gaze DURING a phrase (~20 s @ 5 Hz detection)
        self._gaze_hist: collections.deque = collections.deque(maxlen=100)

        # Voice fingerprint of the current session (from voice_detector via /voice_embedding)
        self._session_voice_emb: list | None = None
        # All voice embeddings of the session (for saving to the gallery during introduction)
        self._session_voice_gallery: list = []  # [{embedding, timestamp}]
        # Counter: how many voice recordings have already been saved to the DB in this INTERACTING session.
        # Up to 3 recordings per session are allowed, with at least a 2-minute interval.
        self._session_voice_save_count: int = 0
        self._session_voice_last_save_ts: float = 0.0
        # Voice embeddings waiting for the lip verdict before the DB (_resolve_voice_saves)
        self._pending_voice_saves: list = []

        # Verification of a claimed identity (in INTRODUCING mode)
        # Tiers: face_sim < FACE_VETO → different person; >= FACE_ACCEPT → accept;
        # in between — voice is needed.
        self._FACE_VETO         = 0.28   # clearly a different face — voice won't help
        self._FACE_ACCEPT       = 0.45   # face says YES — accept without voice
        self._VOICE_ACCEPT      = 0.52   # voice confirms in the face's uncertain zone
        # Three pending-answer state flags:
        #   _pending_name_confirm     — name, waiting for yes/no confirmation
        #   _pending_name_confirm_pid — person_id if the name is already in the DB (for accept_claim),
        #                               None if a new person (for enroll)
        #   _skip_db_check            — don't check the next name against the DB
        #                               (the "another Artur, come up with a nickname" case)
        self._pending_name_confirm:     str | None = None
        self._pending_name_confirm_pid: int | None = None
        self._skip_db_check:            bool       = False

        # Post-goodbye cooldown: after an explicit goodbye we ignore the person for N minutes
        # (or until wake word). Key — person's name, value — goodbye timestamp.
        self._post_goodbye_names: dict[str, float] = {}
        # Blocks face track processing for N seconds after goodbye — prevents
        # an IDLE→RECOGNIZING→post_goodbye→IDLE loop while face_detection keeps running.
        self._post_goodbye_track_block_until: float = 0.0

        # When the current person (by person_id from face_recognition) was last seen.
        # Switching to a different interlocutor is only possible after _dialogue_switch_timeout.
        self._current_person_last_seen: float = 0.0

        # ── Social context (published to /social_context) ───────────
        # One-shot flags: set before publishing, reset afterwards.
        self._should_greet       = False   # BT should greet
        self._greet_text         = ''      # greeting text
        self._introduce_pending  = False   # BT should speak the introduction phrase
        self._introduce_text     = ''      # text to speak

        self._watchdog_timer = None
        self._ctx_timer      = None

    # ── Wakeword / Sleep mode ───────────────────────────────────────────

    def _robot_sleep_cb(self, msg: Bool):
        """Receives the sleep command from behavior_manager. Vision is controlled by the BT."""
        if msg.data and not self._sleeping:
            # /sleep from Telegram — an instant transition, bypassing the normal watchdog/
            # goodbye path. If an introduction was in progress at that moment (State.INTRODUCING),
            # /introducing would otherwise stay True forever — llm_node/telegram_ask
            # would keep answering "busy" forever, even while asleep (bug found 2026-08-30).
            was_introducing = (self._state == State.INTRODUCING)
            self._sleeping = True
            self.get_logger().info('Sleep mode activated')
            with self._lock:
                self._state                    = State.IDLE
                self._primary_track            = None
                self._current_person           = {}
                self._last_emotion             = None
                self._face_emo_pend            = None
                self._voice_emo_pend           = None
                self._face_hunt_since          = 0.0
                self._frontal_scores.clear()
                self._looking_at_robot         = True
                self._pending_name_confirm     = None
                self._pending_name_confirm_pid = None
                self._skip_db_check            = False
            if was_introducing:
                self._set_introducing(False)
            self._pub_person_present(False)
        elif not msg.data and self._sleeping:
            self._sleeping = False
            self.get_logger().info('Waking up: restoring normal mode')

    def _wakeword_cb(self, msg: Bool):
        if not msg.data:
            return
        # Wake word: clear post-goodbye blocks — the person is initiating the dialogue themselves
        with self._lock:
            if self._post_goodbye_names:
                names = ', '.join(self._post_goodbye_names.keys())
                self._post_goodbye_names.clear()
                self.get_logger().info(f'Wake word: post-goodbye cooldown cleared ({names})')
            self._post_goodbye_track_block_until = 0.0
        if self._sleeping:
            self.get_logger().info('Wakeword: exiting sleep mode')
            self._sleeping = False
            wake_msg = Bool()
            wake_msg.data = False
            self._robot_sleep_pub.publish(wake_msg)
            face_msg = String()
            face_msg.data = 'neutral'
            self._face_expr_pub.publish(face_msg)
        # Enabling face_detection is the BT's job (via PIRScanBranch / wake-up)

    def _llm_response_seen_cb(self, _msg):
        """LLM responded → dialogue is active, reset the face-search timer."""
        with self._lock:
            self._last_dialogue_ts = time.time()
            self._face_hunt_since  = 0.0

    # ── Callbacks ───────────────────────────────────────────────────────────

    def _tracks_left_cb(self, msg: String):
        self._left_track_last_msg = time.time()
        if self._tracks_eye != 'left':
            self._tracks_eye = 'left'
            self.get_logger().info('Tracks: left camera restored — switching back from right')
        self._tracks_cb(msg)

    def _tracks_right_cb(self, msg: String):
        elapsed = time.time() - self._left_track_last_msg
        if self._left_track_last_msg > 0.0 and elapsed < self._track_eye_fallback_sec:
            return  # left is active — ignoring right
        if self._tracks_eye != 'right':
            self._tracks_eye = 'right'
            self.get_logger().warn(
                f'Tracks: left camera unavailable ({elapsed:.1f}s) — switching to right')
        self._tracks_cb(msg)

    def _tracks_cb(self, msg: String):
        if self._sleeping:
            return
        if self._waiting_for_body:
            return
        if time.time() < self._post_goodbye_track_block_until:
            return  # post-goodbye track block is active
        try:
            data   = json.loads(msg.data)
            tracks = data.get('tracks', [])
        except Exception:
            return

        with self._lock:
            if tracks:
                self._last_face_time  = time.time()
                self._face_hunt_since = 0.0   # face visible again — reset the hunt
                track_ids = {t['track_id'] for t in tracks}

                if self._primary_track not in track_ids:
                    # We tried selecting by embedding similarity here instead
                    # of bbox area (live bug 2026-08-31: hijacking someone else's/
                    # a false track) — reverted 2026-08-31: it did not fix the
                    # head-drift itself, just added complexity. Left as it was — by bbox area.
                    best = max(tracks, key=lambda t: (
                        (t['bbox'][2] - t['bbox'][0]) * (t['bbox'][3] - t['bbox'][1])))
                    new_track = best['track_id']

                    if self._state == State.IDLE:
                        # New face from IDLE — start recognizing
                        self._primary_track = new_track
                        self._state = State.RECOGNIZING
                        self._session_start_ts = time.time()
                        self.get_logger().info('Face detected — recognizing...')
                        self._pub_person_present(True)
                    else:
                        # INTRODUCING / INTERACTING / RECOGNIZING:
                        # Just update track_id. If the person changed — _identity_cb will detect it
                        # via person_id (or locked=True + unknown). This prevents unnecessary state
                        # transitions on an ordinary track_id change caused by head movement.
                        self.get_logger().debug(
                            f'Track reassigned: {self._primary_track}→{new_track} '
                            f'(state={self._state})')
                        self._primary_track = new_track

                # Save the embedding and compute the frontal score of the primary track
                for t in tracks:
                    if t['track_id'] == self._primary_track:
                        emb       = t.get('embedding', [])
                        det_score = t.get('det_score', 1.0)
                        if emb:
                            self._primary_embedding = emb
                            # In introduction mode, accumulate only quality embeddings
                            if self._state == State.INTRODUCING:
                                if (len(self._enroll_embeddings) < self._enroll_max
                                        and det_score >= self._min_enroll_det):
                                    self._enroll_embeddings.append(emb)
                        # Yaw-proxy from 5 InsightFace keypoints:
                        # kps[0]=left_eye, kps[1]=right_eye, kps[2]=nose_tip
                        # If the nose is offset from the mid-eye point by < 30% of inter-eye dist → frontal
                        kps = t.get('kps', [])
                        if len(kps) >= 3:
                            eye_mid_x   = (kps[0][0] + kps[1][0]) / 2.0
                            eye_dist    = abs(kps[1][0] - kps[0][0])
                            nose_offset = abs(kps[2][0] - eye_mid_x)
                            yaw_proxy   = nose_offset / max(eye_dist, 1.0)
                            now = time.monotonic()
                            if now - self._gaze_ts > self._GAZE_STALE_SEC:
                                # The face is back after a gap — don't let the old
                                # window vote for the new look
                                self._frontal_scores.clear()
                            self._gaze_ts = now
                            self._frontal_scores.append(yaw_proxy)
                            # Pitch-proxy (diagnostic only): where the nose sits between the
                            # eye line (0) and the mouth line (1) — shifts when the head
                            # tilts down/up; yaw alone misses looking at the table/phone
                            pitch_proxy = None
                            if len(kps) >= 5:
                                eye_y   = (kps[0][1] + kps[1][1]) / 2.0
                                mouth_y = (kps[3][1] + kps[4][1]) / 2.0
                                if mouth_y - eye_y > 1.0:
                                    pitch_proxy = (kps[2][1] - eye_y) / (mouth_y - eye_y)
                            self._gaze_hist.append((data.get('stamp') or time.time(),
                                                    self._primary_track, yaw_proxy, pitch_proxy))
                            if len(self._frontal_scores) >= 3:
                                frontal_fraction = sum(
                                    1 for s in self._frontal_scores
                                    if s < self._gaze_yaw_threshold
                                ) / len(self._frontal_scores)
                                self._looking_at_robot = (
                                    frontal_fraction >= self._gaze_frontal_fraction
                                )
                        break

    def _mouth_activity_cb(self, msg: String):
        """Speaker fusion: logs who spoke each phrase and whether the gaze-only and
        gaze+lips gates agree; publishes /speaker_evidence — llm_node vetoes a
        gaze-only address with it."""
        if self._sleeping:
            return
        try:
            mouth = json.loads(msg.data)
        except Exception:
            return
        with self._lock:
            if mouth.get('side') != self._tracks_eye or self._state == State.IDLE:
                return   # track ids of the other eye don't match _primary_track
            primary = self._primary_track
            hist    = list(self._gaze_hist)
        ev = fuse_speaker_evidence(mouth, primary, hist,
                                   self._gaze_yaw_threshold, self._gaze_frontal_fraction,
                                   strong_fraction=self._gaze_strong_fraction,
                                   strong_yaw=self._gaze_strong_yaw,
                                   tail_sec=self._gaze_tail_sec)
        self._speaker_evidence_pub.publish(String(data=json.dumps(ev)))
        self._resolve_voice_saves(ev)

        if ev['sv_rejected']:
            verdict = 'other speaker (SV) — name-only gate'
        elif ev['gaze_gate'] == ev['fused_gate']:
            verdict = f'agree: {"PASS" if ev["fused_gate"] else "block"}'
        else:
            verdict = (f'DISAGREE: gaze={"PASS" if ev["gaze_gate"] else "block"} '
                       f'fused={"PASS" if ev["fused_gate"] else "block"}')
        self.get_logger().info(
            f'Speaker seg#{ev["segment_id"]}: who={ev["who"]} '
            f'(track {primary} lips={ev["lips"]} excess={ev["excess"]}, '
            f'others={ev["others_speaking"] or "-"}) gaze={ev["gaze"]} '
            f'frac={ev["gaze_frac"]} n={ev["gaze_n"]} yaw={ev["yaw_med"]} '
            f'pitch={ev["pitch_med"]} tail={ev["gaze_tail_frac"]} '
            f'strong={ev["gaze_strong"]} → {verdict}')

    def _identity_cb(self, msg: String):
        if self._sleeping:
            return
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        with self._lock:
            if data.get('track_id') != self._primary_track:
                return
            state = self._state

        is_known  = data.get('is_known', False)
        locked    = data.get('locked', False)
        person_id = data.get('person_id')
        name      = data.get('name', 'Незнакомец')

        confidence  = data.get('confidence', 'unknown')
        candidate_id   = data.get('best_candidate_id')
        candidate_name = data.get('best_candidate_name') or name

        if state == State.RECOGNIZING:
            if is_known and confidence == 'high':
                self._on_known(person_id, name)
            elif not is_known and confidence == 'uncertain' and locked:
                # Likely a known person — silent INTERACTING without a greeting
                self._on_uncertain(candidate_id, candidate_name)
            elif not is_known and confidence == 'unknown' and locked:
                self._on_unknown()
            else:
                self.get_logger().debug(
                    f'Track {data.get("track_id")} — not yet recognized, waiting for lock...')

        elif state == State.INTRODUCING:
            # Cancel the introduction only if recognition of a known person came in
            if is_known:
                self.get_logger().info(
                    f'INTRODUCING: face recognized as {name} (id={person_id}) — cancelling introduction')
                now = time.time()
                with self._lock:
                    self._state               = State.INTERACTING
                    self._introduce_attempts  = 0
                    self._enroll_embeddings   = []
                    self._current_person      = {'person_id': person_id, 'name': name}
                    self._greeted_names[name] = now
                    self._last_emotion        = 'neutral'
                self._set_introducing(False)
                self._mark_greeted_today(person_id)
                threading.Thread(
                    target=self._update_seen_and_embedding,
                    args=(person_id,), daemon=True).start()
                threading.Thread(
                    target=self._fetch_and_publish_context,
                    args=(person_id,), daemon=True).start()
                self._should_greet = True
                self._greet_text   = f'Ой, {name}! Прости, я тебя не сразу узнал.'

        elif state == State.INTERACTING:
            now = time.time()
            with self._lock:
                current_id = self._current_person.get('person_id')
                current_name = self._current_person.get('name', '?')

            if is_known and confidence == 'high' and person_id == current_id:
                # Same person — update the last-seen timestamp
                with self._lock:
                    self._current_person_last_seen = now
            elif is_known and confidence == 'high' and person_id != current_id:
                # Different person — switch only after the current one has been absent for a long time
                with self._lock:
                    absent_sec = (now - self._current_person_last_seen
                                  if self._current_person_last_seen > 0 else float('inf'))
                if absent_sec < self._dialogue_switch_timeout:
                    self.get_logger().debug(
                        f'Ignoring {name} (id={person_id}) — '
                        f'dialogue with {current_name}, last seen {absent_sec:.0f}s ago')
                    return
                self.get_logger().info(
                    f'INTERACTING: person switch → {name} (id={person_id}) '
                    f'({current_name} not seen for {absent_sec:.0f}s)')
                with self._lock:
                    self._state             = State.RECOGNIZING
                    self._current_person    = {}
                    self._enroll_embeddings = []
                    self._last_emotion      = None
                self._on_known(person_id, name)
            elif not is_known and confidence == 'uncertain' and locked:
                # Face resembles a candidate but is below the threshold — don't change state
                pass
            elif not is_known and confidence == 'unknown' and locked:
                # Unknown — switch only after the current one has been absent for a long time
                with self._lock:
                    absent_sec = (now - self._current_person_last_seen
                                  if self._current_person_last_seen > 0 else float('inf'))
                    in_cooldown       = (now - self._introduce_last) < self._introduce_cooldown
                    is_enrolled_track = (data.get('track_id') == self._enrolled_track_id)
                if in_cooldown or is_enrolled_track:
                    return
                if absent_sec < self._dialogue_switch_timeout:
                    self.get_logger().debug(
                        f'Ignoring unknown — '
                        f'dialogue with {current_name}, last seen {absent_sec:.0f}s ago')
                    return
                self.get_logger().info(
                    f'INTERACTING: unknown after {absent_sec:.0f}s absence of '
                    f'{current_name} — switching to them')
                with self._lock:
                    self._state             = State.RECOGNIZING
                    self._current_person    = {}
                    self._enroll_embeddings = []
                    self._last_emotion      = None
                self._on_unknown()

    def _emotion_cb(self, msg: String):
        if self._sleeping:
            return
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        with self._lock:
            if data.get('track_id') != self._primary_track:
                return
            if self._state != State.INTERACTING:
                return

        emotion    = data.get('emotion', 'neutral')
        confidence = data.get('confidence', 0.0)

        if confidence < self._emotion_thresh:
            return

        # Update social_context (for the LLM) from the face without fusion
        if emotion != self._last_emotion:
            self._last_emotion = emotion

        # Fusion: store pending and check agreement with voice
        self._face_emo_pend = {'emotion': emotion, 'ts': time.time(), 'confidence': confidence}
        self._check_emotion_fusion()

    def _voice_emotion_cb(self, msg: String):
        if self._sleeping:
            return
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        with self._lock:
            if self._state != State.INTERACTING:
                return

        emotion    = data.get('emotion', 'neutral')
        confidence = data.get('confidence', 0.0)

        if confidence < 0.55:
            return

        self.get_logger().info(f'Voice emotion received: {emotion} (conf={confidence:.2f})')
        self._voice_emo_pend = {'emotion': emotion, 'ts': time.time(), 'confidence': confidence}
        self._check_emotion_fusion()

    def _check_emotion_fusion(self):
        """React to emotion only if face and voice agree within FUSION_WINDOW."""
        face  = self._face_emo_pend
        voice = self._voice_emo_pend
        if face is None or voice is None:
            return

        now = time.time()
        if (now - face['ts']) > self._FUSION_WINDOW:
            return
        if (now - voice['ts']) > self._FUSION_WINDOW:
            return

        if face['emotion'] != voice['emotion']:
            return

        emotion = face['emotion']
        self.get_logger().info(
            f'Fusion: face={face["emotion"]}({face["confidence"]:.2f}) '
            f'+ voice={voice["emotion"]}({voice["confidence"]:.2f}) → reacting'
        )
        # Reset so we don't react again to the same pair
        self._face_emo_pend  = None
        self._voice_emo_pend = None
        self._react_to_emotion(emotion)

    # ── Voice fingerprint ───────────────────────────────────────────────

    def _voice_embedding_cb(self, msg: String):
        """Receives a voice embedding from voice_detector.

        Five roles:
        1. IDLE → try to identify by voice, transition to INTERACTING if recognized
        2. INTERACTING with a known person → add_voice_to_gallery (once per session)
        3. INTRODUCING → accumulate in _session_voice_gallery, check against the DB
        4. RECOGNIZING → try to identify by voice (in parallel with face recognition)
        5. Always → update _session_voice_emb (latest) and _session_voice_gallery
        """
        try:
            data = json.loads(msg.data)
            emb  = data.get('embedding')
            if not emb:
                return
            ts = float(data.get('timestamp', time.time()))
        except Exception:
            return

        if data.get('other_speaker'):
            # SV-rejected phrase: not the interlocutor — keep it out of the session
            # gallery; only tell llm_node who it is (multi-person talk)
            if self._state != State.IDLE:
                threading.Thread(target=self._lookup_voice, args=(emb, True),
                                 daemon=True).start()
            return

        self._session_voice_emb = emb
        self._session_voice_gallery.append({'embedding': emb, 'timestamp': ts})

        with self._lock:
            state     = self._state
            person_id = self._current_person.get('person_id')

        if state == State.IDLE:
            # No face — try to identify by voice
            threading.Thread(
                target=self._try_voice_id_from_idle,
                args=(emb,), daemon=True).start()
            return

        # Whose voice is this: the SV anchor the phrase was accepted against. The face
        # in view is not proof — live 2026-10-05: the anchor stayed Artur's while the
        # face was Nicole, and Artur's phrases were being saved into Nicole's gallery
        # (only memory_node's centroid check stopped it).
        anchor_ok = data.get('anchor_person_id') == person_id
        if state == State.INTERACTING and person_id is not None and not anchor_ok:
            self.get_logger().info(
                f'Voice embedding not saved: SV anchor is person_id='
                f'{data.get("anchor_person_id")}, the face is {person_id}')
        elif state == State.INTERACTING and person_id is not None:
            # Save up to _SV_SESSION_MAX recordings per session with interval _SV_SESSION_GAP.
            # memory_node will apply the 7-day rule: if the gallery is full and fresh — it skips.
            now = time.time()
            gap_ok   = (now - self._session_voice_last_save_ts) >= self._SV_SESSION_GAP
            count_ok = self._session_voice_save_count < self._SV_SESSION_MAX
            if count_ok and gap_ok:
                # Saved only once the lips confirm the face in view spoke this phrase
                # (see _resolve_voice_saves). Live 2026-10-04: Herman's voice passed SV
                # against Nicole's anchor (0.41) while Nicole was in frame — and went
                # into her gallery; then Herman was recognized as Nicole by voice.
                with self._lock:
                    self._expire_voice_saves(now)
                    self._pending_voice_saves.append(
                        {'person_id': person_id, 'emb': emb, 'ts': ts, 'added': now})
                self.get_logger().info(
                    f'Voice embedding (pid={person_id}) — waiting for the lips to confirm '
                    f'before saving to DB')
            elif not count_ok:
                self.get_logger().debug(
                    f'Voice gallery: limit {self._SV_SESSION_MAX}/session reached, skipping')
            else:
                remaining = self._SV_SESSION_GAP - (now - self._session_voice_last_save_ts)
                self.get_logger().debug(
                    f'Voice gallery: too early (another {remaining:.0f}s until next recording)')

        if state == State.INTERACTING:
            # Only to tell llm_node who is speaking (/voice/speaker) — the person
            # being looked at is not necessarily the one talking.
            threading.Thread(target=self._lookup_voice, args=(emb,), daemon=True).start()

        if state == State.RECOGNIZING:
            threading.Thread(
                target=self._try_voice_identification,
                args=(emb,), daemon=True).start()

        elif state == State.INTRODUCING:
            threading.Thread(
                target=self._try_voice_id_in_introducing,
                args=(emb,), daemon=True).start()

    _VOICE_SAVE_WAIT_SEC  = 5.0   # no lip verdict by then → not saved
    _VOICE_SAVE_MATCH_SEC = 1.5   # phrase end (lips) vs embedding time

    def _expire_voice_saves(self, now: float):
        """Under self._lock."""
        keep = []
        for p in self._pending_voice_saves:
            if now - p['added'] > self._VOICE_SAVE_WAIT_SEC:
                self.get_logger().info(
                    f'Voice embedding (pid={p["person_id"]}) not saved: no lip data for the phrase')
            else:
                keep.append(p)
        self._pending_voice_saves = keep

    def _resolve_voice_saves(self, ev: dict):
        """A lip verdict arrived: save the pending voice embedding of that phrase to
        the DB only if the face in view spoke it (who='primary')."""
        if ev.get('sv_rejected') or ev.get('t_end') is None:
            return
        now = time.time()
        with self._lock:
            self._expire_voice_saves(now)
            match = [p for p in self._pending_voice_saves
                     if abs(ev['t_end'] - p['ts']) <= self._VOICE_SAVE_MATCH_SEC]
            if not match:
                return
            for p in match:
                self._pending_voice_saves.remove(p)
            p = match[-1]
            ok = (ev.get('who') == 'primary'
                  and p['person_id'] == self._current_person.get('person_id')
                  and self._session_voice_save_count < self._SV_SESSION_MAX)
            if ok:
                self._session_voice_save_count  += 1
                self._session_voice_last_save_ts = now
                n = self._session_voice_save_count
        if not ok:
            self.get_logger().info(
                f'Voice embedding (pid={p["person_id"]}) not saved: lips show '
                f'who={ev.get("who")} (lips={ev.get("lips")}) — maybe not their voice')
            return
        self.get_logger().info(
            f'Voice embedding confirmed by lips (pid={p["person_id"]}), '
            f'recording {n}/{self._SV_SESSION_MAX} — saving to DB')
        threading.Thread(target=self._add_voice_to_gallery,
                         args=(p['person_id'], p['emb'], p['ts']), daemon=True).start()

    def _lookup_voice(self, emb: list, other_speaker: bool = False):
        """lookup_by_voice for one utterance + publish who is speaking on
        /voice/speaker for llm_node. The robot may be looking at one person
        while another one, out of frame, talks to it — live bug 2026-09-30:
        "смотрю на Николь", Artur said "я справа", the LLM got confused."""
        result = self._call_memory({'op': 'lookup_by_voice', 'embedding': emb,
                                    'high_threshold': self._voice_high_threshold,
                                    'uncertain_threshold': self._voice_uncertain_threshold})
        now = time.time()
        with self._lock:
            face_name    = self._current_person.get('name', '')
            face_visible = (self._state != State.IDLE and
                            now - self._last_face_time < self._no_face_timeout)
        res = result or {}
        conf = res.get('confidence') or 'unknown'
        name = (res.get('name') or res.get('best_candidate_name')) if conf in ('high', 'uncertain') else ''
        self._speaker_pub.publish(String(data=json.dumps({
            'ts':           now,
            'confidence':   conf,
            'name':         name or '',
            'similarity':   round(float(res.get('similarity', 0.0) or 0.0), 3),
            'face_name':    face_name or '',
            'face_visible': face_visible,
            # Not the current interlocutor (SV-rejected) — see voice_detector
            'other_speaker': other_speaker,
        }, ensure_ascii=False)))
        return result

    def _try_voice_id_from_idle(self, emb: list):
        """Voice identification from IDLE (no face visible, wake word triggered).

        If the voice is recognized with high confidence — transition to INTERACTING as with
        normal face recognition. The 2Hz loop will publish person_present=True on the next tick.
        """
        result = self._lookup_voice(emb)
        if not result or result.get('confidence') != 'high':
            sim  = result.get('similarity', 0) if result else 0
            name = result.get('name', '?') if result else '?'
            conf = result.get('confidence', 'none') if result else 'no_result'
            self.get_logger().info(
                f'Voice identification from IDLE: not recognized '
                f'(best={name}, sim={sim:.3f}, confidence={conf}, '
                f'need sim >= {self._voice_high_threshold} to transition to INTERACTING)')
            return

        person_id = result['person_id']
        name      = result.get('name', '?')

        with self._lock:
            if self._state != State.IDLE:
                return  # state changed (face was recognized while we were querying)
            # Watchdog grace period: no face and no body is normal during voice
            # identification. A separate field, not _last_dialogue_ts (that one is checked
            # in _on_known → dialogue_recently, which would block the greeting).
            self._voice_id_grace_ts = time.time()

        self.get_logger().info(
            f'Voice identification from IDLE: {name} (id={person_id}, '
            f'sim={result.get("similarity", 0):.3f}) — transitioning to INTERACTING')
        self._on_known(person_id, name)

    def _try_voice_identification(self, emb: list):
        """Try to recognize the person by voice while face recognition is still running."""
        result = self._lookup_voice(emb)
        if not result or result.get('confidence') != 'high':
            sim  = result.get('similarity', 0) if result else 0
            name = result.get('name', '?') if result else '?'
            conf = result.get('confidence', 'none') if result else 'no_result'
            self.get_logger().info(
                f'Voice identification (RECOGNIZING): not recognized '
                f'(best={name}, sim={sim:.3f}, confidence={conf})')
            return

        person_id = result['person_id']
        name      = result.get('name', '?')

        with self._lock:
            # Race: check that we're still in RECOGNIZING (face recognition hasn't finished)
            if self._state != State.RECOGNIZING:
                return

        self.get_logger().info(
            f'Voice identification: {name} (id={person_id}, '
            f'sim={result.get("similarity", 0):.3f}) — ahead of face recognition')
        self._on_known(person_id, name)

    def _try_voice_id_in_introducing(self, emb: list):
        """In introduction mode: check the voice fingerprint against the gallery.

        high       → straight to INTERACTING (voice confidently recognized)
        uncertain  → ask "Вы случайно не {name}?" ("Aren't you {name} by any chance?") → pending confirm
        unknown    → ignore
        """
        result = self._lookup_voice(emb)
        if not result:
            return

        confidence = result.get('confidence', 'unknown')
        sim        = result.get('similarity', 0.0)

        if confidence == 'high':
            person_id = result['person_id']
            name      = result.get('name', '?')

            with self._lock:
                if self._state != State.INTRODUCING:
                    return
                now = time.time()
                self._state                    = State.INTERACTING
                self._introduce_attempts       = 0
                self._pending_name_confirm     = None
                self._pending_name_confirm_pid = None
                self._skip_db_check            = False
                self._enroll_embeddings        = []
                self._current_person           = {'person_id': person_id, 'name': name}
                self._greeted_names[name]      = now
                self._last_emotion             = 'neutral'
                self._current_person_last_seen = now

            self.get_logger().info(
                f'INTRODUCING: voice recognized — {name} (id={person_id}, sim={sim:.3f})')
            self._set_introducing(False)
            self._mark_greeted_today(person_id)
            self._should_greet = True
            self._greet_text   = f'Прости, {name}! Я тебя не узнал по лицу, но узнал по голосу.'
            threading.Thread(
                target=self._update_seen_and_embedding,
                args=(person_id,), daemon=True).start()
            threading.Thread(
                target=self._fetch_and_publish_context,
                args=(person_id,), daemon=True).start()

        elif confidence == 'uncertain':
            person_id   = result.get('best_candidate_id')
            cand_name   = result.get('best_candidate_name', '?')
            if person_id is None:
                return
            with self._lock:
                if self._state != State.INTRODUCING:
                    return
                if self._pending_name_confirm is not None:
                    return  # already waiting for an answer — don't interrupt
                self._pending_name_confirm     = cand_name
                self._pending_name_confirm_pid = person_id
            self.get_logger().info(
                f'INTRODUCING: voice resembles {cand_name} (sim={sim:.3f}) — asking')
            self._introduce_pending = True
            self._introduce_text    = f'Вы случайно не {cand_name}?'
        else:
            self.get_logger().debug(
                f'Voice identification (INTRODUCING): not recognized (sim={sim:.3f})')

    def _add_voice_to_gallery(self, person_id: int, emb: list, ts: float = None):
        """Adds the embedding to the voice gallery (memory_node applies the 7-day rule)."""
        result = self._call_memory({
            'op':        'add_voice_to_gallery',
            'person_id': person_id,
            'embedding': emb,
            'timestamp': ts if ts is not None else time.time(),
        })
        if result is None:
            self.get_logger().warn(f'Voice gallery: _call_memory returned None (timeout/error) for pid={person_id}')
        elif result.get('added'):
            self.get_logger().info(
                f'Voice gallery saved to DB: pid={person_id}, '
                f'count={result.get("count")}/10')
        else:
            self.get_logger().info(
                f'Voice gallery NOT saved: pid={person_id}, '
                f'reason={result.get("reason", "?")} (age={result.get("oldest_age_days", "?")}d)')

    def _publish_voice_anchor(self, person_id: int | None, name: str):
        """Loads the voice gallery from the DB and sends it to voice_detector.
        No voice in the DB (or no person_id) → an empty anchor: SV drops the previous
        person's gallery and learns this voice from scratch."""
        result = (self._call_memory({'op': 'get_voice_gallery', 'person_id': person_id})
                  if person_id is not None else None)
        if result and result.get('has_voice'):
            msg = String()
            msg.data = json.dumps({
                'person_id': person_id,
                'name':      name,
                'gallery':   result['gallery'],
            }, ensure_ascii=False)
            self._voice_anchor_pub.publish(msg)
            n = len(result['gallery'])
            self.get_logger().info(f'Voice gallery sent to SV: {name} ({n} entries)')
        else:
            self._voice_anchor_pub.publish(String(data=json.dumps(
                {'person_id': person_id, 'name': name, 'gallery': []}, ensure_ascii=False)))
            self.get_logger().info(f'No voice fingerprint for {name} — SV will learn from scratch')

    # ── Voice response in introduction mode ───────────────────────────────

    def _voice_cmd_cb(self, msg: String):
        """Intercept voice_command only in INTRODUCING mode."""
        with self._lock:
            if self._state != State.INTRODUCING:
                return
        text = msg.data.strip()

        # If we are waiting for a yes/no confirmation — handle it separately
        with self._lock:
            pending_confirm = self._pending_name_confirm
        if pending_confirm is not None:
            threading.Thread(
                target=self._handle_name_confirmation, args=(text,), daemon=True).start()
            return

        if not text:
            # STT did not recognize speech — re-ask like on a failed attempt
            with self._lock:
                self._introduce_attempts += 1
                attempts = self._introduce_attempts
            if attempts >= self._max_attempts:
                self.get_logger().warn('STT: no response — moving to INTERACTING')
                self._set_introducing(False)
                with self._lock:
                    self._state = State.INTERACTING
                    self._introduce_attempts = 0
                    self._intro_declined = True
            else:
                self.get_logger().info(
                    f'STT: silence (attempt {attempts}/{self._max_attempts}) — re-asking')
                self._introduce_pending = True
                self._introduce_text    = random.choice(self._RETRY_PHRASES)
            return
        self.get_logger().info(f'INTRODUCING: voice response received: "{text}"')
        threading.Thread(
            target=self._handle_introduce_response, args=(text,), daemon=True).start()

    def _handle_introduce_response(self, text: str):
        # While the LLM was thinking, the watchdog might have switched to IDLE
        with self._lock:
            if self._state != State.INTRODUCING:
                return
            skip_db = self._skip_db_check

        name = self._extract_name(text)
        if not name:
            with self._lock:
                if self._state != State.INTRODUCING:
                    return
                self._introduce_attempts += 1
                attempts = self._introduce_attempts
            if attempts >= self._max_attempts:
                self.get_logger().warn('Could not get a name — moving to INTERACTING')
                self._set_introducing(False)
                with self._lock:
                    self._state = State.INTERACTING
                    self._introduce_attempts = 0
                    self._intro_declined = True
            else:
                self.get_logger().info(
                    f'Name not found (attempt {attempts}/{self._max_attempts}) — re-asking')
                self._introduce_pending = True
                self._introduce_text    = random.choice(self._RETRY_PHRASES)
            return

        self.get_logger().info(f'Name recognized: "{name}"')

        # "Different person with the same name" mode — skip the DB check
        if skip_db:
            with self._lock:
                self._skip_db_check = False
            # Ask for confirmation before enrolling a new person
            with self._lock:
                self._pending_name_confirm     = name
                self._pending_name_confirm_pid = None
            self._introduce_pending = True
            self._introduce_text    = f'Вас зовут {name}? Я правильно понял?'
            return

        # Check: does this name exist in the DB?
        existing = self._call_memory({'op': 'lookup_by_name', 'name': name})
        with self._lock:
            if self._state != State.INTRODUCING:
                return

        if existing and existing.get('person_id') is not None:
            # The name is in the DB → run verification
            person_id = existing['person_id']
            db_name   = existing['name']
            self.get_logger().info(
                f'Name "{name}" found in DB as "{db_name}" (id={person_id}) — verifying')
            threading.Thread(
                target=self._verify_claimed_identity,
                args=(person_id, db_name), daemon=True).start()
        else:
            # New name → ask for confirmation before enrolling
            with self._lock:
                self._pending_name_confirm     = name
                self._pending_name_confirm_pid = None
            self._introduce_pending = True
            self._introduce_text    = f'Вас зовут {name}? Я правильно понял?'

    # Fast regex path: find the first capitalized word (Cyrillic/Latin)
    _NAME_RE = re.compile(r'\b([А-ЯЁ][а-яё]{1,20}|[A-Z][a-z]{1,20})\b')

    # Stopwords — not names even though they start with a capital letter
    _NAME_STOPWORDS = {
        'Меня', 'Зовут', 'Мне', 'Моё', 'Моя', 'Мой', 'Это',
        'Да', 'Нет', 'Привет', 'Здравствуй', 'Пожалуйста',
        'Пока', 'Всё', 'Ладно', 'Хорошо', 'Спасибо', 'Понятно',
        'Ничего', 'Прости', 'Извини', 'Стоп', 'Стой', 'Окей',
        'Можно', 'Нельзя', 'Конечно', 'Именно', 'Просто', 'Тогда',
        'Когда', 'Потом', 'Сейчас', 'Здесь', 'Туда', 'Сюда',
    }

    # Trigger words for the pattern — IGNORECASE only for them, not for the captured name
    # (?i:...) — the inline flag applies only inside the group
    _TRIGGER_RE = re.compile(r'(?i:зовут|зови|называй|меня)\s+([А-ЯЁ][а-яё]{1,20})')

    def _extract_name(self, text: str) -> str | None:
        """Extracts a name from a phrase.

        First tries a fast regex without hitting the network.
        If the phrase is complex (several words, no obvious name) — asks the LLM.
        """
        cleaned = text.strip().strip('.,!?"\'').strip()

        # Fast path via the pattern — priority, checked FIRST
        # "меня зовут Артур" / "зовут Артур" / "меня Артур"
        # IGNORECASE only for the trigger words, the name itself must start with a capital
        m = self._TRIGGER_RE.search(cleaned)
        if m:
            name = m.group(1)
            if name not in self._NAME_STOPWORDS:
                self.get_logger().info(f'Name from regex (pattern): "{name}"')
                return name

        # Fast path: the whole phrase is one or two words (e.g. "Артур" or "Я Артур")
        words = cleaned.split()
        if len(words) <= 2:
            names = [n for n in self._NAME_RE.findall(cleaned)
                     if n not in self._NAME_STOPWORDS]
            if names:
                self.get_logger().info(f'Name from regex (short phrase): "{names[0]}"')
                return names[0]

        # Fallback — LLM for complex cases (uses the same model that is already loaded).
        # A short non-streaming request — no need for the SSE delay for 10 tokens.
        payload = {
            'model': self._name_model,
            'messages': [
                {
                    'role':    'system',
                    'content': (
                        'Извлеки имя человека из фразы. '
                        'Ответь ТОЛЬКО именем в именительном падеже (например: Артур). '
                        'Если имя не упомянуто — ответь: UNKNOWN'
                    ),
                },
                {'role': 'user', 'content': text},
            ],
            'stream':      False,
            'temperature': 0.0,
            'max_tokens':  10,
            'chat_template_kwargs': {'enable_thinking': False},
        }
        headers = {'Authorization': f'Bearer {self._bearer_token}'} if self._bearer_token else {}
        for url in [u for u in (self._llm_url, self._llm_fallback_url) if u]:
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=(3.0, 25.0))
                r.raise_for_status()
                choices = r.json().get('choices') or []
                name = (choices[0].get('message', {}).get('content', '') if choices else '').strip()
                if name.upper() == 'UNKNOWN' or not name:
                    return None
                name = name.split()[0].strip('.,!?"\'')
                return name if name else None
            except requests.exceptions.ConnectionError:
                continue
            except Exception as e:
                self.get_logger().error(f'Name extraction error: {e}')
                return None
        return None

    # ── Verifying a claimed identity ────────────────────────────────────

    _YES_RE = re.compile(r'(?i:^да$|^верно$|^правильно$|^именно$|^точно$|^угу$|^ага$|\bда\b|\bверно\b|\bточно\b)')
    _NO_RE  = re.compile(r'(?i:^нет$|^неверно$|^неправильно$|\bнет\b|\bне\s+(?:так|верно|правильно)\b)')

    def _handle_name_confirmation(self, text: str):
        """Handles a yes/no answer to the name-confirmation question."""
        with self._lock:
            if self._state != State.INTRODUCING:
                self._pending_name_confirm     = None
                self._pending_name_confirm_pid = None
                return
            name = self._pending_name_confirm
            pid  = self._pending_name_confirm_pid

        if name is None:
            return

        is_yes = bool(self._YES_RE.search(text.strip()))
        is_no  = bool(self._NO_RE.search(text.strip()))

        self.get_logger().info(
            f'Name confirmation "{name}": text="{text}", yes={is_yes}, no={is_no}')

        if is_yes:
            with self._lock:
                self._pending_name_confirm     = None
                self._pending_name_confirm_pid = None
            if pid is not None:
                self._accept_identity_claim(pid, name)
            else:
                self._enroll_new_person(name)
        elif is_no:
            with self._lock:
                self._pending_name_confirm     = None
                self._pending_name_confirm_pid = None
                self._introduce_attempts      += 1
                attempts = self._introduce_attempts
            if attempts >= self._max_attempts:
                self.get_logger().warn('Confirmation rejected, attempt limit reached — INTERACTING')
                self._set_introducing(False)
                with self._lock:
                    self._state              = State.INTERACTING
                    self._introduce_attempts = 0
                    self._intro_declined     = True
            else:
                self._introduce_pending = True
                self._introduce_text    = 'Прошу прощения! Как вас зовут?'
        else:
            # Unclear answer — re-ask
            self._introduce_pending = True
            self._introduce_text    = f'Вас зовут {name}? Скажите "да" или "нет".'

    def _verify_claimed_identity(self, person_id: int, name: str):
        """Tiered verification: face is primary, voice is the tiebreaker.

        face_sim < FACE_VETO               → a different person (voice is not considered)
        face_sim in [FACE_VETO, FACE_ACCEPT) → uncertain: needs voice or a direct question
        face_sim >= FACE_ACCEPT            → accept (voice not needed)
        """
        with self._lock:
            if self._state != State.INTRODUCING:
                return
            face_emb  = self._primary_embedding
            voice_emb = self._session_voice_emb

        result = self._call_memory({
            'op':              'verify_person_claim',
            'person_id':       person_id,
            'face_embedding':  face_emb if face_emb else None,
            'voice_embedding': voice_emb if voice_emb else None,
        })
        if result is None:
            result = {}

        face_sim          = result.get('face_sim', 0.0)
        voice_sim         = result.get('voice_sim', 0.0)
        has_voice_gallery = result.get('has_voice_gallery', False)

        self.get_logger().info(
            f'verify_claim "{name}" (id={person_id}): '
            f'face={face_sim:.3f} [veto<{self._FACE_VETO} accept>={self._FACE_ACCEPT}], '
            f'voice={voice_sim:.3f} [accept>={self._VOICE_ACCEPT}] '
            f'has_voice={has_voice_gallery}')

        with self._lock:
            if self._state != State.INTRODUCING:
                return

        if face_sim >= self._FACE_ACCEPT:
            # The face is convincing — accept
            self._accept_identity_claim(person_id, name)

        elif face_sim >= self._FACE_VETO:
            # Uncertain zone — voice is needed
            if has_voice_gallery and voice_sim >= self._VOICE_ACCEPT:
                self._accept_identity_claim(person_id, name)
            elif has_voice_gallery and voice_sim < self._VOICE_ACCEPT:
                # Voice disproves it — a different person with the same name
                self._start_same_name_flow(name)
            else:
                # No voice in the DB — ask directly
                with self._lock:
                    self._pending_name_confirm     = name
                    self._pending_name_confirm_pid = person_id
                self._introduce_pending = True
                self._introduce_text    = (
                    f'Вы очень похожи на {name} в моей памяти, но я не уверен. '
                    f'Вы точно {name}?'
                )
        else:
            # The face is clearly different — voice does not matter
            self._start_same_name_flow(name)

    def _accept_identity_claim(self, person_id: int, name: str):
        """Transition to INTERACTING with an apology — identity confirmed."""
        with self._lock:
            if self._state != State.INTRODUCING:
                return
            now = time.time()
            self._state                    = State.INTERACTING
            self._introduce_attempts       = 0
            self._pending_name_confirm     = None
            self._pending_name_confirm_pid = None
            self._skip_db_check            = False
            self._enroll_embeddings        = []
            self._current_person           = {'person_id': person_id, 'name': name}
            self._greeted_names[name]      = now
            self._last_emotion             = 'neutral'
            self._current_person_last_seen = now

        self.get_logger().info(f'Verification succeeded: {name} (id={person_id})')
        self._set_introducing(False)
        self._mark_greeted_today(person_id)
        self._should_greet = True
        self._greet_text   = f'Прости, {name}! Я тебя не узнал. Больше постараюсь запомнить!'
        threading.Thread(
            target=self._update_seen_and_embedding, args=(person_id,), daemon=True).start()
        threading.Thread(
            target=self._fetch_and_publish_context, args=(person_id,), daemon=True).start()
        threading.Thread(
            target=self._publish_voice_anchor, args=(person_id, name), daemon=True).start()

    def _start_same_name_flow(self, existing_name: str):
        """Starts the introduction flow for a new person who has the same name."""
        with self._lock:
            if self._state != State.INTRODUCING:
                return
            self._pending_name_confirm     = None
            self._pending_name_confirm_pid = None
            self._skip_db_check            = True   # the next answer goes straight to enrollment

        self.get_logger().info(
            f'A different person named "{existing_name}" — starting a new introduction')
        self._introduce_pending = True
        self._introduce_text    = (
            f'Интересно! У меня уже есть знакомый по имени {existing_name}, '
            f'но вы на него не похожи. Как вас называть, чтобы не перепутать?'
        )

    # ─────────────────────────────────────────────────────────────────────

    def _enroll_new_person(self, name: str):
        """Saves the new person to memory and starts the greeting."""
        import numpy as np
        with self._lock:
            collected  = list(self._enroll_embeddings)
            fallback   = self._primary_embedding

        if not collected and not fallback:
            self.get_logger().warn('No embedding for enrollment — moving to INTERACTING')
            self._set_introducing(False)
            with self._lock:
                self._state = State.INTERACTING
            return

        if collected:
            # Average all collected embeddings and normalize the result
            mat = np.array(collected, dtype=np.float32)
            avg = mat.mean(axis=0)
            norm = np.linalg.norm(avg)
            if norm > 0:
                avg /= norm
            embedding = avg.tolist()
            self.get_logger().info(
                f'Enrollment: averaged {len(collected)} embeddings for "{name}"')
        else:
            embedding = fallback
            self.get_logger().warn(
                f'Enrollment: only 1 embedding (collection incomplete) for "{name}"')

        result = self._call_memory({
            'op':        'save_person',
            'name':      name,
            'embedding': embedding,
        })

        if result and 'person_id' in result:
            person_id = result['person_id']
            self.get_logger().info(f'Registered: {name} (id={person_id})')

            with self._lock:
                self._state                    = State.INTERACTING
                self._current_person           = {'person_id': person_id, 'name': name}
                self._session_greeted.add(person_id)
                self._introduce_attempts       = 0
                self._last_emotion             = 'neutral'
                self._introduce_last           = time.time()
                self._enrolled_track_id        = self._primary_track
                self._current_person_last_seen = time.time()
                voice_gallery = list(self._session_voice_gallery)

            self._set_introducing(False)
            self._mark_greeted_today(person_id)

            # Save the new person's voice gallery
            if voice_gallery:
                def _save_gallery(pid, entries):
                    for entry in entries:
                        self._call_memory({
                            'op':        'add_voice_to_gallery',
                            'person_id': pid,
                            'embedding': entry['embedding'],
                            'timestamp': entry['timestamp'],
                        })
                    self.get_logger().info(
                        f'Voice gallery saved for {name} ({len(entries)} entries)')
                threading.Thread(
                    target=_save_gallery, args=(person_id, voice_gallery), daemon=True).start()

            # Publish the context for the LLM
            threading.Thread(
                target=self._fetch_and_publish_context,
                args=(person_id,), daemon=True).start()

            # The BT greets the newly met person via social_context
            self._should_greet = True
            self._greet_text   = (
                f'Очень приятно познакомиться, {name}! '
                f'Расскажи немного о себе.'
            )
        else:
            self.get_logger().error('Failed to save the person to memory')
            self._set_introducing(False)
            with self._lock:
                self._state = State.INTERACTING

    def _set_introducing(self, active: bool):
        """Publishes /introducing — a gate for llm_node."""
        msg = Bool()
        msg.data = active
        self._introducing_pub.publish(msg)

    # ── State logic ─────────────────────────────────────────────────────

    # ── Once-a-day greeting ──────────────────────────────────────────────

    def _greet_day(self) -> str:
        shifted = datetime.datetime.now() - datetime.timedelta(hours=self._greet_day_start_hour)
        return shifted.date().isoformat()

    def _load_greet_log(self) -> dict:
        try:
            with open(self._greet_log_path, encoding='utf-8') as f:
                data = json.load(f)
            return {str(k): str(v) for k, v in data.items()}
        except FileNotFoundError:
            return {}
        except Exception as e:
            self.get_logger().warn(f'Greet log {self._greet_log_path} unreadable ({e}) — starting empty')
            return {}

    def _greeted_today(self, person_id) -> bool:
        if not person_id:
            return False
        with self._lock:
            return self._greet_log.get(str(person_id)) == self._greet_day()

    def _mark_greeted_today(self, person_id):
        if not person_id:
            return
        with self._lock:
            self._greet_log[str(person_id)] = self._greet_day()
            data = dict(self._greet_log)
        try:
            tmp = self._greet_log_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            os.replace(tmp, self._greet_log_path)
        except Exception as e:
            self.get_logger().warn(f'Greet log not saved: {e}')

    def _on_known(self, person_id: int, name: str):
        """Transition to INTERACTING for a known person.

        Called from RECOGNIZING (face recognition) or IDLE (voice identification).
        """
        now = time.time()
        greeted_today = self._greeted_today(person_id)   # takes the lock itself
        with self._lock:
            # Cooldown by name — independent of person_id (the tracker may change the id)
            already_greeted = ((now - self._greeted_names.get(name, 0)) < self._greet_cooldown
                               or greeted_today)
            goodbye_ts = self._post_goodbye_names.get(name, 0)

        # Post-farewell cooldown: the person said goodbye themselves — don't greet until the wake word
        if (now - goodbye_ts) < self._post_goodbye_ignore_sec:
            with self._lock:
                self._state         = State.IDLE
                self._primary_track = None
                self._current_person = {}
                # Extend the track block so it doesn't loop
                self._post_goodbye_track_block_until = now + self._post_goodbye_track_block_sec
            self._pub_person_present(False)
            return

        # If the dialogue has recently been carried only by voice (without face recognition),
        # and face recognition has now found the person — don't greet again.
        with self._lock:
            dialogue_recently = (self._last_dialogue_ts > 0.0 and
                                 (now - self._last_dialogue_ts) < self._greet_cooldown)

        if not already_greeted and not dialogue_recently:
            self._greet_known(person_id, name, now)
        else:
            # Silent transition — already greeted, or there was an active voice dialogue
            with self._lock:
                self._state                    = State.INTERACTING
                self._current_person           = {'person_id': person_id, 'name': name}
                self._greeted_names[name]      = now   # refresh the cooldown
                self._last_emotion             = 'neutral'
                self._current_person_last_seen = now
            reason = 'already greeted today' if already_greeted else 'a voice dialogue was active'
            self.get_logger().info(
                f'{name} recognized (id={person_id}), {reason} — straight to INTERACTING without a greeting')
            threading.Thread(
                target=self._publish_voice_anchor,
                args=(person_id, name), daemon=True).start()
            threading.Thread(
                target=self._fetch_and_publish_context,
                args=(person_id,), daemon=True).start()

    def _on_uncertain(self, person_id: int | None, name: str):
        """The face resembles a known person, but sim is below the confident threshold.
        Transition to INTERACTING without a greeting and without an introduction.
        The gallery will accumulate more photos and recognize more confidently next time.
        """
        now = time.time()
        with self._lock:
            self._state                    = State.INTERACTING
            self._current_person           = {'person_id': person_id, 'name': name}
            self._last_emotion             = 'neutral'
            self._current_person_last_seen = now
        self.get_logger().info(
            f'Unconfident recognition: probably {name} (id={person_id}) — '
            f'silent INTERACTING, not starting an introduction')
        # The voice anchor too — otherwise SV keeps the PREVIOUS person's voice and
        # rejects this one (live 2026-10-05: Nicole recognized "probably", every
        # phrase of hers judged against Artur's anchor). A wrong guess is caught by
        # the lips (llm_node accepts a lips-confirmed SV-rejected phrase).
        threading.Thread(
            target=self._publish_voice_anchor,
            args=(person_id, name), daemon=True).start()
        if person_id is not None:
            threading.Thread(
                target=self._fetch_and_publish_context,
                args=(person_id,), daemon=True).start()

    def _greet_known(self, person_id: int, name: str, now: float):
        with self._lock:
            self._state                    = State.INTERACTING
            self._current_person           = {'person_id': person_id, 'name': name}
            self._greeted_names[name]      = now
            self._last_emotion             = 'neutral'
            self._current_person_last_seen = now

        self._mark_greeted_today(person_id)
        self.get_logger().info(f'Greeting: {name} (id={person_id})')

        threading.Thread(
            target=self._update_seen_and_embedding,
            args=(person_id,), daemon=True).start()

        threading.Thread(
            target=self._publish_voice_anchor,
            args=(person_id, name), daemon=True).start()

        # Reminders are requested asynchronously — should_greet is set afterward
        threading.Thread(
            target=self._fetch_reminders_and_greet,
            args=(person_id, name), daemon=True).start()

        threading.Thread(
            target=self._fetch_and_publish_context,
            args=(person_id,), daemon=True).start()

    # Phrases to start an introduction — chosen at random for naturalness
    _INTRO_PHRASES = [
        'Привет! Я тебя раньше не видел. Как тебя зовут?',
        'Здравствуй! Мы ещё не знакомы. Как тебя зовут?',
        'О, новое лицо! Рад познакомиться. Как вас зовут?',
        'Привет! Я Лёня. А тебя как зовут?',
        'Добро пожаловать! Я тебя не знаю. Представься, пожалуйста.',
        'Привет! Не помню, чтобы мы встречались. Как тебя зовут?',
        'Здравствуй! Я не знаю твоего имени. Как тебя зовут?',
        'О, привет! Ты новый человек для меня. Как вас зовут?',
    ]

    # Phrases to re-ask — chosen at random
    _RETRY_PHRASES = [
        'Простите, я не расслышал. Как вас зовут?',
        'Извините, не разобрал. Повторите ваше имя, пожалуйста.',
        'Прошу прощения, не услышал. Как вас зовут?',
        'Не расслышал имя. Не могли бы повторить?',
    ]

    def _on_unknown(self):
        """An unknown face is locked: wait silently, don't introduce proactively.

        The robot keeps looking at the person (INTERACTING, person_id=None), and the
        introduction starts only when they address it — see _speech_addressed_cb.
        People don't walk up to everyone they see and ask their name either.
        """
        now = time.time()
        with self._lock:
            if self._state == State.INTRODUCING:
                return   # already asking
            # OakD gate: if OAK-D is active and hasn't seen a body for >10s — the face is a
            # false positive (the watchdog will move to IDLE on its own)
            if self._last_human_time > 0.0 and (now - self._last_human_time) > 10.0:
                self.get_logger().warn(
                    f'Unknown face, but OakD has not seen a body for '
                    f'{now - self._last_human_time:.0f}s — '
                    f'ignoring it (probably a face_detection false positive)')
                return
            self._state                    = State.INTERACTING
            self._current_person           = {}
            self._last_emotion             = 'neutral'
            self._current_person_last_seen = now
        self.get_logger().info(
            'Unknown person — waiting silently; introduction only if they address the robot')

    def _speech_addressed_cb(self, msg: String):
        """llm_node judged an utterance addressed to the robot while an unknown face
        is in front of it (social_context.introduce_on_address) — now introduce."""
        with self._lock:
            unknown_waiting = (self._state == State.INTERACTING
                               and not self._current_person.get('person_id')
                               and not self._intro_declined)
        if unknown_waiting:
            self.get_logger().info(f'Unknown person addressed the robot: "{msg.data[:60]}"')
            self._start_introduction()

    def _start_introduction(self):
        now = time.time()
        with self._lock:
            if self._state == State.INTRODUCING:
                return   # already asking
            self._state = State.INTRODUCING
            self._introduce_last     = now
            self._introduce_attempts = 0
            self._enroll_embeddings  = []   # start accumulating from scratch
            self._enrolled_track_id  = None

        phrase = random.choice(self._INTRO_PHRASES)
        self.get_logger().info('Unknown person — starting an introduction')
        self._set_introducing(True)
        # Don't publish a direct command — the BT will read introduce_pending from social_context
        self._introduce_pending = True
        self._introduce_text    = phrase

    def _react_to_emotion(self, emotion: str):
        mirror_map = {
            'happy':     'happy',
            'sad':       'sad',
            'angry':     'neutral',
            'surprised': 'surprised',
            'fear':      'neutral',
            'disgust':   'neutral',
            'neutral':   'neutral',
        }
        robot_emotion = mirror_map.get(emotion, 'neutral')
        # Facial mirroring: a reflex, published directly without the BT (low-level reflex)
        msg = String()
        msg.data = robot_emotion
        self._face_expr_pub.publish(msg)
        self.get_logger().info(f'Person emotion: {emotion} → facial expression: {robot_emotion}')

    def _human_detected_cb(self, msg: Bool):
        """Secondary presence signal from OAK-D body detection."""
        if self._sleeping:
            return
        if msg.data:
            with self._lock:
                self._last_human_time = time.time()
            if self._waiting_for_body:
                self._waiting_for_body = False
                self.get_logger().info('OakD: body detected → lifting the face_detection block')

    def _go_idle_cb(self, msg: Bool):
        """Forced transition to IDLE from the BT (say_goodbye LLM tool call)."""
        if not msg.data or self._sleeping:
            return
        with self._lock:
            if self._state == State.IDLE:
                return
            # Save the name and mode before clearing the state
            goodbye_name    = self._current_person.get('name', '')
            prev_state      = self._state
            now             = time.time()

            self.get_logger().info('go_idle: forced transition to IDLE')
            self._state                    = State.IDLE
            self._primary_track            = None
            self._primary_embedding        = None
            self._enroll_embeddings        = []
            self._current_person           = {}
            self._last_emotion             = None
            self._face_emo_pend            = None
            self._voice_emo_pend           = None
            self._introduce_attempts       = 0
            self._greeted_names            = {}
            self._intro_declined           = False
            self._enrolled_track_id        = None
            self._face_hunt_since          = 0.0
            self._last_dialogue_ts         = 0.0
            self._session_voice_emb        = None
            self._session_voice_gallery    = []
            self._session_voice_save_count   = 0
            self._pending_voice_saves        = []
            self._session_voice_last_save_ts = 0.0
            self._current_person_last_seen = 0.0
            self._pending_name_confirm     = None
            self._pending_name_confirm_pid = None
            self._skip_db_check            = False
            self._waiting_for_body         = True
            self._frontal_scores.clear()
            self._looking_at_robot         = True
            was_introducing                = (prev_state == State.INTRODUCING)

            # Post-farewell cooldown: the person said goodbye themselves — ignore them
            # for N minutes (or until the wake word), even if face_detection sees them.
            if goodbye_name:
                self._post_goodbye_names[goodbye_name] = now
                self.get_logger().info(
                    f'Post-farewell cooldown: {goodbye_name} — '
                    f'{self._post_goodbye_ignore_sec / 60:.0f} min (or until the wake word)')
            # Block tracks for N seconds — prevents a re-greeting loop
            self._post_goodbye_track_block_until = now + self._post_goodbye_track_block_sec

        self._pub_context({})
        self._pub_person_present(False)
        # Only publish /introducing False if we were actually in introduction mode —
        # otherwise voice_detector and llm_node get noisy "INTRODUCING finished" on an ordinary farewell.
        if was_introducing:
            self._set_introducing(False)

    def _watchdog(self):
        if self._sleeping:
            return
        now = time.time()
        with self._lock:
            if self._state == State.IDLE:
                return

            go_idle     = False
            idle_reason = ''

            # OakD veto: if OAK-D is active (we've received at least one signal) and hasn't
            # seen a body for longer than no_human_timeout — treat face_detection as a false
            # positive and go to IDLE. This is critical: the left camera can detect a
            # poster/reflection indefinitely, while the real person has been gone for a while.
            # Not applied while the face is recognized as the current person (a locked track
            # republishes every frame): a poster is not "Артур with sim=0.85", while OAK-D
            # routinely loses the body of someone standing close (<1 m) or turned sideways.
            # Also not applied during an active dialogue (the LLM answered < 60s ago).
            # Absence is counted from the later of the last body and the session start —
            # otherwise a body seen minutes ago killed a fresh session within one tick,
            # before recognition could even finish (live bug 2026-09-30).
            current_face_seen = (self._current_person_last_seen > 0.0 and
                                 (now - self._current_person_last_seen) < self._no_face_timeout)
            dialogue_active = (self._last_dialogue_ts > 0.0 and
                               (now - self._last_dialogue_ts) < 60.0)
            if (self._last_human_time > 0.0 and not current_face_seen
                    and not dialogue_active):
                body_absent_sec = now - max(self._last_human_time, self._session_start_ts)
                if body_absent_sec > self._no_human_timeout:
                    go_idle     = True
                    idle_reason = (
                        f'OakD veto: body not seen for {body_absent_sec:.0f}s '
                        f'(face_detection likely a false positive)')
                    self._waiting_for_body = True

            if not go_idle:
                # INTRODUCING needs more time: TTS + VAD delay + speech + STT
                face_timeout = (self._no_face_timeout * 4
                                if self._state == State.INTRODUCING
                                else self._no_face_timeout)

                face_lost = (now - self._last_face_time) > face_timeout
                if not face_lost:
                    return  # the face is visible — all good

                # Dialogue active: the LLM answered recently → give 60s for the user's reply
                dialogue_active = (self._last_dialogue_ts > 0.0 and
                                   (now - self._last_dialogue_ts) < 60.0)
                if dialogue_active:
                    self._face_hunt_since = 0.0
                    return

                # Voice identification from IDLE: no face/body — normal, the person is
                # nearby but out of frame. Give 60s until the first LLM reply (after which
                # _last_dialogue_ts updates and takes over).
                if (self._voice_id_grace_ts > 0.0 and
                        (now - self._voice_id_grace_ts) < 60.0):
                    self._face_hunt_since = 0.0
                    return

                # Face lost — check body detection as a fallback signal
                human_detected = (now - self._last_human_time) < self._no_human_timeout
                if human_detected:
                    if self._face_hunt_since == 0.0:
                        self._face_hunt_since = now
                    hunt_sec = now - self._face_hunt_since
                    if hunt_sec >= self._max_face_hunt:
                        idle_reason = (
                            f'Face not found for {hunt_sec:.0f}s while the body is present — '
                            f'resetting to IDLE, the BT will restart the search')
                        go_idle = True
                    else:
                        return
                else:
                    idle_reason = 'Person left (no face and no body)'
                    go_idle = True

            if go_idle:
                self.get_logger().info(f'{idle_reason} → IDLE')
                self._state                    = State.IDLE
                self._primary_track            = None
                self._primary_embedding        = None
                self._enroll_embeddings        = []
                self._current_person           = {}
                self._last_emotion             = None
                self._face_emo_pend            = None
                self._voice_emo_pend           = None
                self._introduce_attempts       = 0
                self._greeted_names            = {}
                self._intro_declined           = False
                self._enrolled_track_id        = None
                self._face_hunt_since          = 0.0
                self._last_dialogue_ts         = 0.0
                self._voice_id_grace_ts        = 0.0
                self._session_voice_emb        = None
                self._session_voice_gallery    = []
                self._session_voice_save_count   = 0
                self._pending_voice_saves        = []
                self._session_voice_last_save_ts = 0.0
                self._current_person_last_seen = 0.0
                self._pending_name_confirm     = None
                self._pending_name_confirm_pid = None
                self._skip_db_check            = False
                self._pub_context({})
                self._pub_person_present(False)
                self._set_introducing(False)

    def _update_seen_and_embedding(self, person_id: int):
        """update_seen + an EMA update of the embedding on every encounter."""
        import numpy as np
        self._call_memory({'op': 'update_seen', 'person_id': person_id})

        with self._lock:
            collected = list(self._enroll_embeddings)
            fallback  = self._primary_embedding

        embeddings = collected if collected else ([fallback] if fallback else [])
        if not embeddings:
            return

        # Average the available embeddings and update via EMA
        mat = np.array(embeddings, dtype=np.float32)
        avg = mat.mean(axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg /= norm
        self._call_memory({
            'op':        'update_embedding',
            'person_id': person_id,
            'embedding': avg.tolist(),
            'alpha':     0.2,   # soft update: 20% new, 80% old
        })

    # ── Helpers ────────────────────────────────────────────────────────

    def _publish_social_context(self):
        """Publishes the social context @ 2 Hz → the BehaviorManager Blackboard.

        Fully stops in sleep mode.
        should_greet and introduce_pending are one-shot signals: True is published
        once, then automatically reset.

        """
        if self._sleeping:
            return

        with self._lock:
            state           = self._state
            person          = dict(self._current_person)
            emotion         = self._last_emotion or 'neutral'
            intro           = (state == State.INTRODUCING)
            looking         = self._looking_at_robot
            intro_on_addr   = (state == State.INTERACTING
                               and not self._current_person.get('person_id')
                               and not self._intro_declined)
            has_face        = (len(self._frontal_scores) >= 3
                               and time.monotonic() - self._gaze_ts <= self._GAZE_STALE_SEC)

        person_present = state != State.IDLE
        ctx = {
            'person_present':    person_present,
            'person_id':         person.get('person_id'),
            'name':              person.get('name', ''),
            'is_known':          bool(person.get('person_id')),
            'emotion':           emotion,
            'state':             state,
            'introducing':       intro,
            # An unknown face waits silently: llm_node hands the next addressed
            # utterance to /speech_addressed (→ introduction) instead of the LLM.
            'introduce_on_address': intro_on_addr,
            # True if the interlocutor is looking the robot in the eye (yaw-proxy < 30% of
            # inter-eye dist, >50% of frames over the last ~1.5s). None if no face kps in the
            # last _GAZE_STALE_SEC (face gone / no data) or fewer than 3 frames yet.
            'looking_at_robot':  looking if has_face else None,
            # One-shot flags (reset after the first publish)
            'should_greet':      self._should_greet,
            'greet_text':        self._greet_text if self._should_greet else '',
            'introduce_pending': self._introduce_pending,
            'introduce_text':    self._introduce_text if self._introduce_pending else '',
        }
        # Reset the one-shot flags
        self._should_greet      = False
        self._introduce_pending = False

        msg = String()
        msg.data = json.dumps(ctx, ensure_ascii=False)
        self._social_ctx_pub.publish(msg)

        # Update /person_present Bool @ 2Hz — voice_detector uses this topic's
        # timestamp to determine whether a person is present during a dialogue.
        # Without this the grace period (120s) expires and the pipeline drops into wake-word
        # mode even while the person is actively interacting (state=INTERACTING).
        if person_present:
            self._pub_person_present(True)

    def _pub_context(self, ctx: dict):
        msg = String()
        msg.data = json.dumps(ctx, ensure_ascii=False)
        self._context_pub.publish(msg)

    def _pub_person_present(self, present: bool):
        msg = Bool()
        msg.data = present
        self._person_present_pub.publish(msg)

    def _fetch_reminders_and_greet(self, person_id: int, name: str):
        """Requests manual reminders and sets should_greet with the text.

        Env reminders (source='env:...') are no longer stored in the DB —
        they go straight to Telegram via openhab_bridge_node.
        Here we handle only manual reminders (delivered=0):
          - show them in the greeting
          - mark delivered=1 and delete right away (confirm_reminders)
        """
        import datetime as _dt

        now_dt     = _dt.datetime.now()
        today      = now_dt.date().isoformat()
        now_time   = now_dt.strftime('%H:%M')
        greet_text = f'Привет, {name}!'
        manual_ids: list[int] = []

        try:
            result = self._call_memory({
                'op':        'get_due_reminders',
                'person_id': person_id,
                'today':     today,
                'now_time':  now_time,
            })
            if result and isinstance(result.get('reminders'), list):
                for r in result['reminders']:
                    greet_text += ' ' + r['message']
                    manual_ids.append(r['id'])
        except Exception as e:
            self.get_logger().warn(f'_fetch_reminders_and_greet: {e}')

        with self._lock:
            if self._current_person.get('person_id') == person_id:
                self._should_greet = True
                self._greet_text   = greet_text

        # Mark delivered=1 and delete: said in person → safe to delete
        for rid in manual_ids:
            try:
                self._call_memory({'op': 'mark_reminder_delivered', 'reminder_id': rid})
            except Exception:
                pass
        if manual_ids:
            try:
                self._call_memory({'op': 'confirm_reminders', 'person_id': person_id})
            except Exception:
                pass
            self.get_logger().info(
                f'Reminders for {name}: shown and deleted {len(manual_ids)} manual ones')

    def _fetch_and_publish_context(self, person_id: int):
        result = self._call_memory({'op': 'get_context', 'person_id': person_id})
        if result and 'error' not in result:
            with self._lock:
                result['current_emotion'] = self._last_emotion
            self._pub_context(result)
            name = result.get('name', '?')
            self.get_logger().info(
                f'person_context → LLM: {name} (id={person_id}), '
                f'emotion={result.get("current_emotion")}'
            )
        else:
            self.get_logger().warn(
                f'Failed to get context for person_id={person_id}')

    def _call_memory(self, req: dict) -> dict | None:
        if not self._mem.wait_for_service(timeout_sec=2.0):
            return None
        request = MemoryQuery.Request()
        request.request_json = json.dumps(req)
        future = self._mem.call_async(request)
        done_event = threading.Event()
        future.add_done_callback(lambda _: done_event.set())
        if not done_event.wait(timeout=5.0):
            return None
        try:
            return json.loads(future.result().response_json)
        except Exception:
            return None

    # ── Lifecycle callbacks ────────────────────────────────────────────────

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores a repeated declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('no_face_timeout_sec',          15.0)
        self._dp('no_human_timeout_sec',         30.0)
        self._dp('max_face_hunt_sec',            90.0)
        self._dp('greet_cooldown_sec',          120.0)
        # Once-a-day greeting: person_id → day, persisted across restarts.
        # The day starts at greet_day_start_hour (a 01:00 return is still "today").
        self._dp('greet_log_path',   os.path.expanduser('~/inmoov_greet_log.json'))
        self._dp('greet_day_start_hour',          4)
        self._dp('emotion_react_thresh',          0.70)
        self._dp('introduce_cooldown_sec',       60.0)
        self._dp('max_introduce_attempts',        3)
        self._dp('min_enroll_det_score',         0.65)
        self._dp('dialogue_switch_timeout_sec', 30.0)
        self._dp('track_eye_fallback_sec',        2.0)
        self._dp('post_goodbye_ignore_sec',    1800.0)
        self._dp('post_goodbye_track_block_sec', 30.0)
        self._dp('llm_url', 'http://192.168.10.118:18020/v1/chat/completions')
        self._dp('voice_high_threshold',    0.62)
        self._dp('voice_uncertain_threshold', 0.50)
        self._dp('llm_fallback_url', '')   # optional backup endpoint; empty = none
        self._dp('bearer_token', '')
        self._dp('name_extract_model', 'qwen3.8-27b')
        self._dp('gaze_yaw_threshold',  0.30)
        self._dp('gaze_frontal_fraction', 0.50)
        self._dp('gaze_strong_fraction',  0.80)
        self._dp('gaze_strong_yaw',       0.15)
        self._dp('gaze_tail_sec',         1.5)

        self._no_face_timeout              = self.get_parameter('no_face_timeout_sec').value
        self._no_human_timeout             = self.get_parameter('no_human_timeout_sec').value
        self._max_face_hunt                = self.get_parameter('max_face_hunt_sec').value
        self._greet_cooldown               = self.get_parameter('greet_cooldown_sec').value
        self._emotion_thresh               = self.get_parameter('emotion_react_thresh').value
        self._introduce_cooldown           = self.get_parameter('introduce_cooldown_sec').value
        self._max_attempts                 = self.get_parameter('max_introduce_attempts').value
        self._min_enroll_det               = self.get_parameter('min_enroll_det_score').value
        self._dialogue_switch_timeout      = self.get_parameter('dialogue_switch_timeout_sec').value
        self._track_eye_fallback_sec       = self.get_parameter('track_eye_fallback_sec').value
        self._post_goodbye_ignore_sec      = self.get_parameter('post_goodbye_ignore_sec').value
        self._post_goodbye_track_block_sec = self.get_parameter('post_goodbye_track_block_sec').value
        self._llm_url                      = self.get_parameter('llm_url').value
        self._llm_fallback_url             = self.get_parameter('llm_fallback_url').value
        self._bearer_token                 = self.get_parameter('bearer_token').value
        self._name_model                   = self.get_parameter('name_extract_model').value
        self._voice_high_threshold         = self.get_parameter('voice_high_threshold').value
        self._voice_uncertain_threshold    = self.get_parameter('voice_uncertain_threshold').value
        self._gaze_yaw_threshold           = self.get_parameter('gaze_yaw_threshold').value
        self._gaze_frontal_fraction        = self.get_parameter('gaze_frontal_fraction').value
        self._gaze_strong_fraction         = self.get_parameter('gaze_strong_fraction').value
        self._gaze_strong_yaw              = self.get_parameter('gaze_strong_yaw').value
        self._gaze_tail_sec                = self.get_parameter('gaze_tail_sec').value
        self._greet_log_path               = self.get_parameter('greet_log_path').value
        self._greet_day_start_hour         = int(self.get_parameter('greet_day_start_hour').value)
        self._greet_log                    = self._load_greet_log()

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(String, '/face/identity',     self._identity_cb,        10)
        self.create_subscription(String, '/face/emotion',      self._emotion_cb,         10)
        self.create_subscription(String, '/voice/emotion',     self._voice_emotion_cb,   10)
        self.create_subscription(String, '/face/tracks/left',  self._tracks_left_cb,     10)
        self.create_subscription(String, '/face/tracks/right', self._tracks_right_cb,    10)
        self.create_subscription(Bool,   '/wake_detected',     self._wakeword_cb,        10)
        self.create_subscription(Bool,   '/human_detected',    self._human_detected_cb,  10)
        self.create_subscription(String, '/voice_command',     self._voice_cmd_cb,       10)
        self.create_subscription(String, '/speech_addressed',  self._speech_addressed_cb, 10)
        self.create_subscription(String, '/voice_embedding',   self._voice_embedding_cb, 10)
        self.create_subscription(Bool,   '/robot_sleep',       self._robot_sleep_cb,     latched_qos)
        self.create_subscription(Bool,   '/go_idle',           self._go_idle_cb,         10)
        self.create_subscription(String, '/llm_response',      self._llm_response_seen_cb, 10)
        self.create_subscription(String, '/face/mouth_activity/left',  self._mouth_activity_cb, 10)
        self.create_subscription(String, '/face/mouth_activity/right', self._mouth_activity_cb, 10)

        self._social_ctx_pub     = self.create_lifecycle_publisher(String, '/social_context',  10)
        self._context_pub        = self.create_lifecycle_publisher(String, '/person_context',  10)
        self._person_present_pub = self.create_lifecycle_publisher(Bool,   '/person_present',  10)
        self._face_expr_pub      = self.create_lifecycle_publisher(String, '/face_expression', 10)
        self._introducing_pub    = self.create_lifecycle_publisher(Bool,   '/introducing',     10)
        self._voice_anchor_pub   = self.create_lifecycle_publisher(String, '/voice_anchor',    10)
        self._speaker_pub        = self.create_lifecycle_publisher(String, '/voice/speaker',   10)
        self._robot_sleep_pub    = self.create_lifecycle_publisher(Bool,   '/robot_sleep',     latched_qos)
        self._speaker_evidence_pub = self.create_lifecycle_publisher(String, '/speaker_evidence', 10)

        self._mem = self.create_client(MemoryQuery, '/memory/query')
        self.get_logger().info('IdentityManager configured')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._social_ctx_pub.on_activate(state)
        self._context_pub.on_activate(state)
        self._person_present_pub.on_activate(state)
        self._face_expr_pub.on_activate(state)
        self._introducing_pub.on_activate(state)
        self._voice_anchor_pub.on_activate(state)
        self._robot_sleep_pub.on_activate(state)
        self._speaker_evidence_pub.on_activate(state)
        self._watchdog_timer = self.create_timer(0.5, self._watchdog)
        self._ctx_timer      = self.create_timer(0.5, self._publish_social_context)
        self.get_logger().info('IdentityManager ready')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        if self._watchdog_timer:
            self.destroy_timer(self._watchdog_timer)
            self._watchdog_timer = None
        if self._ctx_timer:
            self.destroy_timer(self._ctx_timer)
            self._ctx_timer = None
        self._social_ctx_pub.on_deactivate(state)
        self._context_pub.on_deactivate(state)
        self._person_present_pub.on_deactivate(state)
        self._face_expr_pub.on_deactivate(state)
        self._introducing_pub.on_deactivate(state)
        self._voice_anchor_pub.on_deactivate(state)
        self._robot_sleep_pub.on_deactivate(state)
        self._speaker_evidence_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS


def main():
    rclpy.init()
    node = IdentityManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
