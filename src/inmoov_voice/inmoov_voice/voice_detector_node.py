"""
voice_detector_node.py
======================
Voice activity detection (Silero VAD) + optional speaker verification
(ECAPA-TDNN) + phrase recording. Sits between the wake word and the STT.

Flow: wake_detected -> record speech from raw_audio until silence ->
audio_to_whisper (post-VAD segment) -> STT -> voice_command (used here only
to know that STT has answered and to re-arm listening).

Subscribes:
  wake_detected     (Bool)               — wake word fired, start recording
  tts_speaking      (Bool)               — robot is talking; mic input ignored
  raw_audio         (Float32MultiArray)  — 16kHz float32 chunks
  voice_command     (String)             — STT result (empty = silence)
  /person_present   (Bool)
  /introducing      (Bool)               — relaxed min speech length + gallery growth
  /go_idle          (Bool)               — explicit goodbye: stop and reset
  /robot_sleep      (Bool, latched)
  /voice_anchor     (String JSON)        — voice gallery from the DB
                                           {person_id, name, gallery: [{embedding, timestamp}]}

Publishes:
  audio_to_whisper  (Float32MultiArray)  — recorded phrase, dim label 'sample_rate';
                                           an extra dim 'other_speaker' marks a phrase SV
                                           rejected (not the current interlocutor)
  /voice_embedding  (String JSON)        — {embedding, timestamp[, other_speaker]} -> identity_manager
                                           (other_speaker: SV-rejected phrase, voice lookup only)
  /voice/segment    (String JSON)        — {id, t_start, t_end, env_hz, envelope, vad, sv_rejected}
                                           of every phrase sent to STT or rejected by SV
                                           -> face_tracker lip activity
  /robot_sleep      (Bool, latched)      — False on wake word while asleep
  /tts_cancel_queue (Bool)               — True on wake word during TTS

Parameters:
  sample_rate (16000), vad_threshold (0.4), silence_duration_sec (1.0),
  min_phrase_sec (0.3), min_speech_sec (1.0), min_speech_sec_introducing (0.4),
  max_phrase_sec (20.0), no_speech_timeout_sec (8.0), pipeline_timeout_sec (90.0),
  speaker_verification (True), sv_threshold (0.35), sv_min_speech_sec (1.2), sv_debug_dir ('')

Speaker verification is decided per PHRASE, not per 1 s segment: the whole
phrase is recorded as without SV, then one ECAPA embedding of all its speech is
compared with the current speaker's reference (the DB anchor, or the session's
own phrases). A phrase from someone else (another person, the TV) is dropped as
a whole. 1 s embeddings were too noisy on this mic (the owner's own segments
matched each other at 0.0-0.4) — they rejected the owner, cut phrases down to
fragments for STT and made introductions loop forever (live 2026-09-26).
  - introducing: accepted unchecked — the new person's answers become the reference
  - speech shorter than sv_min_speech_sec: accepted (too short to judge: «да», a name)
  - no reference yet: the phrase becomes an UNCONFIRMED seed; 2 rejections in a
    row while unconfirmed → re-seed (a seed of noise can't lock the owner out)

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import collections
import json
import os
import time

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray, Bool, MultiArrayDimension, String
import numpy as np
import torch


class VoiceDetectorNode(LifecycleNode):
    def __init__(self):
        super().__init__('voice_detector_node')
        # Placeholders — filled in on_configure
        self.publisher_      = None
        self._voice_emb_pub  = None
        self._sleep_pub      = None
        self._tts_cancel_pub = None
        self.vad_model       = None
        self._sv_encoder     = None

    # ── Lifecycle: Phase 2 ─────────────────────────────────────────────────

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores re-declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        # ── Parameters ────────────────────────────────────────────────────
        self._dp('sample_rate',           16000)
        self._dp('vad_threshold',         0.4)
        self._dp('silence_duration_sec',  1.0)
        self._dp('min_phrase_sec',        0.3)
        self._dp('min_speech_sec',        1.0)
        self._dp('min_speech_sec_introducing', 0.4)
        self._dp('max_phrase_sec',        20.0)
        self._dp('no_speech_timeout_sec', 8.0)
        self._dp('pipeline_timeout_sec',  90.0)
        self._dp('speaker_verification',  True)
        self._dp('sv_threshold',          0.35)   # phrase-level cosine similarity
        self._dp('sv_min_speech_sec',     1.2)    # shorter phrases are not judged
        self._dp('sv_debug_dir',          '')     # save every judged phrase as WAV (tuning)
        self._dp('sv_savedir',
                 os.path.expanduser('~/.cache/speechbrain/spkrec-ecapa-voxceleb'))

        self.rate              = self.get_parameter('sample_rate').value
        self.vad_threshold     = self.get_parameter('vad_threshold').value
        self.min_phrase_sec    = self.get_parameter('min_phrase_sec').value
        self.min_speech_sec    = self.get_parameter('min_speech_sec').value
        self.min_speech_sec_introducing = self.get_parameter('min_speech_sec_introducing').value
        self.max_phrase_sec    = self.get_parameter('max_phrase_sec').value
        self.no_speech_timeout = self.get_parameter('no_speech_timeout_sec').value
        self.pipeline_timeout  = self.get_parameter('pipeline_timeout_sec').value
        silence_duration_sec   = self.get_parameter('silence_duration_sec').value
        self._sv_enabled       = self.get_parameter('speaker_verification').value
        self._sv_threshold     = self.get_parameter('sv_threshold').value
        self._sv_min_speech_sec = self.get_parameter('sv_min_speech_sec').value
        self._sv_debug_dir     = os.path.expanduser(self.get_parameter('sv_debug_dir').value)

        # ── Subscriptions ─────────────────────────────────────────────────
        self.create_subscription(Bool,            'wake_detected',   self.wake_callback,          10)
        self.create_subscription(Bool,            'tts_speaking',    self._tts_speaking_callback, 10)
        self.create_subscription(Float32MultiArray, 'raw_audio',     self._audio_callback,        20)
        self.create_subscription(String,          'voice_command',   self._stt_done_callback,     10)
        self.create_subscription(Bool,            '/person_present', self._person_present_cb,     10)
        self.create_subscription(Bool,            '/introducing',    self._introducing_cb,         10)
        self.create_subscription(Bool,            '/go_idle',        self._go_idle_cb,            10)
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Bool,   '/robot_sleep',  self._robot_sleep_cb,  latched_qos)
        self.create_subscription(String, '/voice_anchor', self._voice_anchor_cb, 10)
        self.create_subscription(String, '/voice/sv_confirm', self._sv_confirm_cb, 10)

        # ── Lifecycle Publishers ───────────────────────────────────────────
        self.publisher_      = self.create_lifecycle_publisher(Float32MultiArray, 'audio_to_whisper', 10)
        self._voice_emb_pub  = self.create_lifecycle_publisher(String, '/voice_embedding', 10)
        self._segment_pub    = self.create_lifecycle_publisher(String, '/voice/segment', 10)
        self._segment_id     = 0
        self._sleep_pub      = self.create_lifecycle_publisher(Bool, '/robot_sleep', latched_qos)
        self._tts_cancel_pub = self.create_lifecycle_publisher(Bool, '/tts_cancel_queue', 10)

        # ── Silero VAD ────────────────────────────────────────────────────
        self.get_logger().info('Loading Silero VAD...')
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
        )
        self.vad_model.eval()
        self.get_logger().info('Silero VAD loaded')

        # ── Speaker Verification (ECAPA-TDNN) ─────────────────────────────
        self._sv_encoder = None
        if self._sv_enabled:
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
                self.get_logger().info('Loading ECAPA-TDNN (spkrec-ecapa-voxceleb)...')
                self._sv_encoder = EncoderClassifier.from_hparams(
                    source='speechbrain/spkrec-ecapa-voxceleb',
                    savedir=self.get_parameter('sv_savedir').value,
                    run_opts={'device': 'cpu'},
                )
                self.get_logger().info(
                    f'Speaker Verification enabled (ECAPA-TDNN, 192D, '
                    f'phrase-level, threshold={self._sv_threshold}, '
                    f'min speech {self._sv_min_speech_sec}s)')
            except Exception as e:
                self._sv_enabled = False
                self.get_logger().warn(f'ECAPA-TDNN not loaded: {e}')

        # ── Buffers and state ─────────────────────────────────────────────
        self._chunk_size        = None
        self._silence_threshold = None
        self._max_chunks        = None
        self._silence_secs      = silence_duration_sec
        self._pre_roll_buffer   = collections.deque()
        self._pre_roll_secs     = 1.5
        self._pre_roll_maxlen   = None
        self._onset_buf         = collections.deque()
        self._onset_secs        = 0.4
        self._onset_maxlen      = None
        self._speech_audio      = []   # VAD-positive chunks of the current phrase (SV input)
        self._sv_gallery        = []
        self._sv_gallery_times  = []
        self._sv_anchor_person_id = None  # whose anchor is currently in _sv_gallery (None = live session without a DB anchor)
        self._SV_GALLERY_MAX    = 10
        self._sv_last_gallery_add      = 0.0
        self._SV_GALLERY_ADD_INTERVAL  = 10.0   # phrase-level: learn from most phrases
        # Live seed guard: without a DB anchor the FIRST judged phrase becomes the
        # session's reference. A live seed stays "unconfirmed" until a later phrase
        # matches it; N rejections in a row while unconfirmed → the seed is dropped
        # and the current phrase re-seeds. DB anchors and confirmed seeds are never
        # dropped this way (a TV can't evict the owner).
        self._sv_seed_unconfirmed      = False
        self._sv_seed_rejects          = 0
        self._SV_SEED_MAX_REJECTS      = 2   # phrases
        # A phrase dropped as too short ("Эй, Лёня" — 0.6 s) is kept for a moment
        # and prepended to the next phrase if that one starts right after it
        # (live 2026-10-03: "Эй Лёня" dropped, then "Кими хороший" came without
        # the name and was ignored). (chunks, speech chunks, finish time)
        self._carry             = None
        # Embeddings of recent SV-rejected phrases by /voice/segment id (see _sv_confirm_cb)
        self._rejected_embs: dict = {}
        self._sv_last_reject_emb = None
        self._introducing       = False
        self.is_active          = False
        self.audio_buffer       = []
        self.silence_counter    = 0
        self.speech_chunks      = 0
        self.activation_time    = 0.0
        self.tts_speaking       = False
        self._stt_sent_time     = 0.0
        self._sleeping          = False
        self._person_present    = None
        self._person_present_time  = 0.0
        self._person_last_seen     = 0.0
        self._person_present_grace = 120.0
        # Live bug 2026-08-28: right after the wake word, BEFORE the first
        # successful phrase in this session, _is_person_present() is always
        # False (_person_last_seen hasn't been updated yet, neither from
        # /person_present nor from a successful send to STT) — if the very
        # first recording is dropped as too short (e.g. the VAD caught only the
        # tail of the wake word utterance), auto-reactivation doesn't fire and
        # the microphone "dies" until the next wake word, as if the person had
        # said nothing at all. We give a separate grace window AFTER the wake
        # word — we don't rely on _is_person_present() alone.
        self._last_wake_time          = 0.0
        self._POST_WAKE_LISTEN_GRACE_SEC = 15.0
        # Lifecycle ACTIVE flag: callbacks ignore wake/audio while INACTIVE
        # (subscriptions stay alive through deactivate).
        self._lc_active       = False
        # Pending delayed re-listen (one-shot ROS timer on the executor thread, so it
        # never races the audio callback). Cancelled on sleep/go_idle/wake/deactivate.
        self._relisten_timer  = None

        sv_status = 'on' if self._sv_enabled else 'off'
        self.get_logger().info(
            f'VAD node ready. Waiting for the wake word... '
            f'(vad_threshold={self.vad_threshold}, silence={silence_duration_sec}s, '
            f'max_phrase={self.max_phrase_sec}s, speaker_verification={sv_status})')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._lc_active = True
        self.publisher_.on_activate(state)
        self._voice_emb_pub.on_activate(state)
        self._segment_pub.on_activate(state)
        self._sleep_pub.on_activate(state)
        self._tts_cancel_pub.on_activate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self._lc_active = False
        self._cancel_relisten()
        self.is_active = False
        self.audio_buffer = []
        self._speech_audio.clear()
        self._pre_roll_buffer.clear()
        self.publisher_.on_deactivate(state)
        self._voice_emb_pub.on_deactivate(state)
        self._segment_pub.on_deactivate(state)
        self._sleep_pub.on_deactivate(state)
        self._tts_cancel_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

    # ── Delayed re-listen ─────────────────────────────────────────────────

    def _schedule_relisten(self, delay_sec: float):
        """(Re)schedules _activate_after_tts after delay_sec; replaces any pending one."""
        self._cancel_relisten()
        self._relisten_timer = self.create_timer(delay_sec, self._relisten_fire)

    def _cancel_relisten(self):
        if self._relisten_timer is not None:
            self.destroy_timer(self._relisten_timer)
            self._relisten_timer = None

    def _relisten_fire(self):
        self._cancel_relisten()   # one-shot
        if self._lc_active:
            self._activate_after_tts()

    # ── Pre-roll ──────────────────────────────────────────────────────────

    def _take_pre_roll_speech(self) -> list:
        """Speech chunks from the last ~1.5 s of idle audio; clears the pre-roll.

        While idle the audio callback only buffers chunks — VAD runs here, once,
        on activation (~50-80 ms) instead of on every chunk around the clock.
        """
        chunks = list(self._pre_roll_buffer)
        self._pre_roll_buffer.clear()
        if not chunks:
            return []
        self.vad_model.reset_states()
        with torch.no_grad():
            return [a for a in chunks
                    if self.vad_model(torch.from_numpy(a), self.rate).item() > self.vad_threshold]

    # ── Control callbacks ─────────────────────────────────────────────────

    def wake_callback(self, msg: Bool):
        if not msg.data or not self._lc_active:
            return
        self._cancel_relisten()

        if self._sleeping:
            wake_msg = Bool()
            wake_msg.data = False
            self._sleep_pub.publish(wake_msg)
            self.get_logger().info('Wake word while asleep → publishing /robot_sleep False')

        if self.tts_speaking:
            cancel_msg = Bool()
            cancel_msg.data = True
            self._tts_cancel_pub.publish(cancel_msg)
            self.get_logger().info('Wake word during TTS → cancelling the TTS queue')

        # Reset the gallery only if there is no active interlocutor.
        # In INTERACTING mode (a person is present) the gallery is kept between phrases.
        person_active = self._is_person_present()

        if self.is_active:
            # Keep the last _WAKE_KEEP_SEC, drop what came before. The wake word fires
            # ~0.3-0.8 s after "Лёня" (openWakeWord patience), so a full reset threw
            # away the wake word AND the first words of the request (live 2026-10-05:
            # STT got "Zavody", "Morredi de escogo."). The wake word stays in the
            # phrase — STT hears the whole thing and the name opens the gate.
            keep = max(1, int(self._WAKE_KEEP_SEC * self.rate / (self._chunk_size or 512)))
            kept = self.audio_buffer[-keep:]
            ids  = {id(c) for c in kept}
            self.get_logger().warn(
                f'Wake word during recording — keeping the last {self._WAKE_KEEP_SEC:.1f}s '
                f'(the wake word + what follows), dropping the rest')
            self.audio_buffer    = kept
            self._speech_audio   = [c for c in self._speech_audio if id(c) in ids]
            self.speech_chunks   = len(self._speech_audio)
            self.silence_counter = 0
            self.activation_time = time.time()
            self._last_wake_time = time.time()
            if not person_active:
                self._sv_reset_reference()
            return

        speech_chunks = self._take_pre_roll_speech()
        self._onset_buf.clear()
        self.get_logger().info('Woke up! Listening for a command...')
        self.is_active       = True
        self.audio_buffer    = speech_chunks
        self._speech_audio   = list(speech_chunks)
        self.silence_counter = 0
        self.speech_chunks   = 0
        self.activation_time = time.time()
        self._last_wake_time = time.time()
        self._speech_audio.clear()
        if not person_active:
            self._sv_reset_reference()

    def _stt_done_callback(self, msg: String):
        if self._stt_sent_time <= 0.0:
            return
        self._stt_sent_time = 0.0
        if not msg.data:
            if self._is_person_present() and not self._sleeping:
                self.get_logger().info('STT: silence — keep listening (person nearby)')
                self._schedule_relisten(0.5)
            else:
                self.get_logger().info('STT: silence — back to the wake word')
        else:
            # STT returned text — the LLM is now processing it and should start TTS.
            # If the LLM blocked the phrase (gaze gate, /introducing, busy) — no TTS will come.
            # Safety timer: activate after 6s if TTS never started.
            # If TTS did arrive earlier — _activate_after_tts checks tts_speaking and won't duplicate.
            if self._is_person_present() and not self._sleeping:
                self._schedule_relisten(6.0)

    def _go_idle_cb(self, msg: Bool):
        """Explicit goodbye from the BT — stop recording immediately, reset the grace period.

        Unlike person_present=False (grace 120s), go_idle means: the person
        said goodbye and no longer expects an answer. Return to the wake word immediately.
        """
        if not msg.data:
            return
        self._cancel_relisten()
        self._person_present   = False
        self._person_last_seen = 0.0   # reset the grace period — don't wait 120s
        self._stt_sent_time    = 0.0
        if self.is_active:
            self.audio_buffer    = []
            self.silence_counter = 0
            self.speech_chunks   = 0
            self.is_active       = False
            self._speech_audio.clear()
        self._sv_reset_reference()
        self._introducing = False
        self.get_logger().info('go_idle: stopping recording, back to the wake word')

    def _introducing_cb(self, msg: Bool):
        if msg.data and not self._introducing:
            # A new person: their answers must not be judged against whoever spoke
            # before — they become the reference (see _sv_check_phrase)
            self._sv_reset_reference()
            self.get_logger().info('SV: introduction started — reference reset for the new person')
        self._introducing = msg.data
        if not msg.data and self._stt_sent_time > 0.0:
            self.get_logger().info('INTRODUCING finished — resetting the pipeline timeout')
            self._stt_sent_time = 0.0

    def _robot_sleep_cb(self, msg: Bool):
        self._sleeping = msg.data
        if msg.data:
            self._cancel_relisten()
            if self.is_active:
                self.audio_buffer    = []
                self.silence_counter = 0
                self.is_active       = False
                self._speech_audio.clear()
            self._stt_sent_time = 0.0
            self._sv_reset_reference()
            self._introducing = False
            self.get_logger().info('Sleep mode: auto-activation disabled')
        else:
            self.get_logger().info('Wake up: auto-activation restored')

    def _person_present_cb(self, msg: Bool):
        now = time.time()
        appeared = msg.data and self._person_present is not True
        self._person_present      = msg.data
        self._person_present_time = now
        if msg.data:
            self._person_last_seen = now
        # A person showed up — listen without the wake word. Used to happen only
        # after the greeting's TTS; with greetings once a day (2026-10-03) a known
        # face got no TTS and the mic stayed on the wake word (live 2026-10-05).
        # Whether a phrase is meant for the robot is llm_node's gate (name / gaze + lips).
        if (appeared and self._lc_active and not self.is_active and not self.tts_speaking
                and not self._sleeping and self._stt_sent_time <= 0.0):
            self.get_logger().info('Person appeared — listening without the wake word')
            self._schedule_relisten(0.3)

    def _activate_after_tts(self):
        if not self.is_active and not self.tts_speaking and not self._sleeping:
            speech_chunks = self._take_pre_roll_speech()
            self._onset_buf.clear()
            self.is_active       = True
            self.audio_buffer    = []
            self.silence_counter = 0
            self.speech_chunks   = 0
            self.activation_time = time.time()
            self._speech_audio   = []
            # The gallery is NOT reset — it lives for the whole session. Reset only on go_idle / robot_sleep.

            if speech_chunks:
                # Kept unchecked here: SV judges the whole phrase at its end
                dur = len(speech_chunks) * (self._chunk_size or 512) / self.rate
                self.audio_buffer  = speech_chunks
                self._speech_audio = list(speech_chunks)
                self.speech_chunks = len(speech_chunks)
                self.get_logger().info(
                    f'Activation + pre-roll: captured {dur:.2f}s of speech before activation')

    def _is_person_present(self) -> bool:
        now = time.time()
        if self._person_present is None:
            return True
        if (now - self._person_last_seen) < self._person_present_grace:
            return True
        return False

    def _should_keep_listening(self) -> bool:
        """_is_person_present() OR we woke up on the wake word very recently.

        Needed separately from _is_person_present(): right after the wake word,
        before the first phrase of this session has been successfully sent to
        STT, presence is not yet confirmed (_person_last_seen hasn't been
        updated). If the very first recording is dropped as too short (the VAD
        caught only the tail of the wake word utterance) — without this check
        the microphone "died" for good, as if the person had not said a single
        word after the wake word."""
        if self._is_person_present():
            return True
        return (time.time() - self._last_wake_time) < self._POST_WAKE_LISTEN_GRACE_SEC

    def _tts_speaking_callback(self, msg: Bool):
        was_speaking  = self.tts_speaking
        self.tts_speaking = msg.data

        if msg.data:
            self._stt_sent_time = 0.0
            self._pre_roll_buffer.clear()
            if self.is_active:
                self.get_logger().info('TTS started speaking during recording — resetting the buffer')
                self.audio_buffer    = []
                self.silence_counter = 0
                self.speech_chunks   = 0
                self.is_active       = False
                self._speech_audio.clear()
                # The gallery is NOT reset — the next utterance is filtered against the same gallery
        else:
            if was_speaking and not self.is_active:
                if self._sleeping:
                    self.get_logger().info('TTS finished — sleep mode, auto-activation skipped')
                elif self._is_person_present():
                    self.get_logger().info('TTS finished — waiting for the user reply...')
                    self._schedule_relisten(1.2)
                else:
                    self.get_logger().info(
                        'TTS finished — nobody in frame, auto-activation disabled')

    def _sv_confirm_cb(self, msg: String):
        """llm_node: an SV-rejected phrase was spoken by the face in view (its lips
        moved with the speech, it looked at the robot) — the interlocutor after all.
        Its voice joins the live session gallery (not the DB directly), so the next
        phrases pass SV. The gallery is reset with the session anyway."""
        try:
            seg_id = json.loads(msg.data).get('segment_id')
        except Exception:
            return
        emb = self._rejected_embs.pop(seg_id, None)
        if emb is None or not self._sv_enabled:
            return
        self._sv_last_gallery_add = 0.0   # bypass the add interval: a correction
        if self._sv_add_to_gallery(emb):
            self.get_logger().info(
                f'SV: lips confirmed the interlocutor\'s voice (segment {seg_id}) — '
                f'added to the live gallery')

    def _voice_anchor_cb(self, msg: String):
        """Load the voice gallery from the DB (identity_manager → voice_detector).

        We receive the full JSON: {person_id, name, gallery: [{embedding, timestamp}]}

        If the current live gallery already belongs to THIS SAME person_id — we
        don't overwrite it (the session's live entries are more accurate than
        the old snapshot from the DB). But if the anchor is for a DIFFERENT
        person (the interlocutor changed while the live gallery was not reset —
        e.g. a quiet IDLE without /go_idle within the person_present grace
        window) — we replace it unconditionally, otherwise SV would compare the
        new voice against the old one and reject it as foreign (incident 2026-08-25).
        """
        if not self._sv_enabled or self._sv_encoder is None:
            return
        try:
            data = json.loads(msg.data)
            anchor_person_id = data.get('person_id')
        except Exception as e:
            self.get_logger().warn(f'SV: failed to parse the voice gallery: {e}')
            return
        if self._sv_gallery:
            if anchor_person_id is not None and anchor_person_id == self._sv_anchor_person_id:
                return  # the live gallery already belongs to the same person — don't overwrite
            self.get_logger().info(
                f'SV: anchor changed (was person_id={self._sv_anchor_person_id}, '
                f'now {anchor_person_id}) — replacing the live gallery ({len(self._sv_gallery)} entries)')
            self._sv_gallery.clear()
            self._sv_gallery_times.clear()
        self._sv_anchor_person_id = anchor_person_id
        self._sv_seed_unconfirmed = False   # a DB gallery is trusted, not a live guess
        try:
            gallery = data.get('gallery', [])
            if not gallery:
                # Fallback: legacy single-embedding format
                emb_list = data.get('embedding')
                if emb_list:
                    emb = np.array(emb_list, dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-8:
                        emb /= norm
                        self._sv_gallery = [emb]
                        self._sv_gallery_times = [0.0]
                        self.get_logger().info('SV: gallery loaded (1 legacy entry)')
                return
            new_gallery, new_times = [], []
            for entry in gallery:
                emb = np.array(entry['embedding'], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm < 1e-8:
                    continue
                emb /= norm
                new_gallery.append(emb)
                new_times.append(float(entry.get('timestamp', 0.0)))
            if new_gallery:
                self._sv_gallery = new_gallery
                self._sv_gallery_times = new_times
                self.get_logger().info(
                    f'SV: gallery loaded from the DB ({len(new_gallery)} entries)')
        except Exception as e:
            self.get_logger().warn(f'SV: failed to parse the voice gallery: {e}')

    # ── Speaker Verification ──────────────────────────────────────────────

    # ECAPA-TDNN needs at least ~0.5s of audio (otherwise conv padding > time_dim → RuntimeError)
    _SV_MIN_SAMPLES = 8000  # 0.5s @ 16kHz

    def _sv_embed(self, audio_seg: np.ndarray) -> 'np.ndarray | None':
        """L2-normalized 192-dim ECAPA-TDNN embedding.
        Returns None if the audio is shorter than _SV_MIN_SAMPLES.
        """
        if len(audio_seg) < self._SV_MIN_SAMPLES:
            return None
        # RMS normalization: ECAPA is sensitive to the input signal level.
        # Equalize to RMS=0.05 so embeddings are comparable across sessions.
        rms = float(np.sqrt(np.mean(audio_seg ** 2)))
        if rms > 1e-6:
            audio_seg = np.clip(audio_seg * (0.05 / rms), -1.0, 1.0)
        wav = torch.tensor(audio_seg).unsqueeze(0)
        wav_lens = torch.tensor([1.0])
        with torch.no_grad():
            emb = self._sv_encoder.encode_batch(wav, wav_lens)  # [1, 1, 192]
        emb = emb.squeeze().numpy().astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-8
        return emb

    def _sv_sim(self, emb: np.ndarray) -> float:
        """Maximum cosine similarity against the session's voice gallery.

        max instead of mean: the gallery contains entries from different sessions (anchor from
        the DB) — stale entries recorded under different acoustic conditions drag the mean
        down and hide a match with the current ones.
        """
        if not self._sv_gallery:
            return 0.0
        sims = [float(np.dot(g, emb)) for g in self._sv_gallery]
        self.get_logger().debug(
            f'SV sims [{len(sims)}]: {[f"{s:.2f}" for s in sorted(sims, reverse=True)]}')
        return float(np.max(sims))

    def _sv_add_to_gallery(self, emb: np.ndarray) -> bool:
        """Adds an embedding to the session's live gallery.

        Limits: at most once per _SV_GALLERY_ADD_INTERVAL seconds,
        at most _SV_GALLERY_MAX entries (the oldest is replaced by the new one).
        Publishes the embedding to /voice_embedding → identity_manager → DB.
        """
        now = time.time()
        if (now - self._sv_last_gallery_add) < self._SV_GALLERY_ADD_INTERVAL:
            return False
        if len(self._sv_gallery) >= self._SV_GALLERY_MAX:
            self._sv_gallery.pop(0)
            self._sv_gallery_times.pop(0)
        self._sv_gallery.append(emb)
        self._sv_gallery_times.append(now)
        self._sv_last_gallery_add = now
        n = len(self._sv_gallery)
        self.get_logger().info(f'SV: entry {n}/{self._SV_GALLERY_MAX} added to the gallery')
        self._publish_voice_emb(emb, now)
        return True

    def _publish_voice_emb(self, emb: np.ndarray, ts: float = None,
                           other_speaker: bool = False):
        emb_msg = String()
        data = {
            'embedding': emb.tolist(),
            'timestamp': ts if ts is not None else time.time(),
        }
        if other_speaker:
            data['other_speaker'] = True
        # Whose gallery this voice matched — identity_manager saves it to that
        # person's DB gallery only if it is the face in view (see its _voice_embedding_cb)
        data['anchor_person_id'] = self._sv_anchor_person_id
        emb_msg.data = json.dumps(data)
        self._voice_emb_pub.publish(emb_msg)

    def _sv_seed(self, emb: np.ndarray) -> None:
        """(Re)starts the live gallery from one segment, as an unconfirmed seed."""
        now = time.time()
        self._sv_gallery = [emb]
        self._sv_gallery_times = [now]
        self._sv_last_gallery_add = now
        self._sv_seed_unconfirmed = True
        self._sv_seed_rejects = 0

    def _sv_reset_reference(self) -> None:
        """Forget the current speaker (gallery, DB anchor, seed state)."""
        self._sv_gallery.clear()
        self._sv_gallery_times.clear()
        self._sv_anchor_person_id = None
        self._sv_last_gallery_add = 0.0
        self._sv_seed_unconfirmed = False
        self._sv_seed_rejects = 0

    def _sv_check_phrase(self, speech: np.ndarray) -> 'tuple[bool, str]':
        """Phrase-level speaker check of the phrase's speech. Returns (accept, log line)."""
        if not self._sv_enabled or self._sv_encoder is None:
            return True, ''
        sec = len(speech) / self.rate
        emb = self._sv_embed(speech)
        if emb is None:
            return True, ''

        if self._introducing:
            # A new person answers (their name…): accept, and learn their voice
            if self._sv_gallery:
                self._sv_add_to_gallery(emb)
            else:
                self._sv_seed(emb)
                self._sv_seed_unconfirmed = False   # introduction = who we talk to now
                self._publish_voice_emb(emb)
            self._sv_debug_save(speech, 'intro', None)
            return True, f'SV: introducing — phrase accepted unchecked ({sec:.1f}s speech), voice learned'

        if sec < self._sv_min_speech_sec:
            self._sv_debug_save(speech, 'short', None)
            return True, f'SV: {sec:.1f}s of speech — too short to judge, accepted'

        if not self._sv_gallery:
            self._sv_seed(emb)
            self._publish_voice_emb(emb)   # lets identity_manager try to recognise the voice
            self._sv_debug_save(speech, 'seed', None)
            return True, (f'SV: first phrase of the session ({sec:.1f}s) — the reference '
                          f'(unconfirmed until the next phrase matches it)')

        sim = self._sv_sim(emb)
        if sim >= self._sv_threshold:
            confirmed = self._sv_seed_unconfirmed
            self._sv_seed_unconfirmed = False
            self._sv_seed_rejects = 0
            self._sv_add_to_gallery(emb)
            self._sv_debug_save(speech, 'accept', sim)
            return True, (f'SV: phrase accepted (similarity={sim:.2f} >= {self._sv_threshold:.2f}'
                          f'{", reference confirmed" if confirmed else ""})')

        if self._sv_seed_unconfirmed:
            self._sv_seed_rejects += 1
            if self._sv_seed_rejects >= self._SV_SEED_MAX_REJECTS:
                self._sv_seed(emb)
                self._publish_voice_emb(emb)
                self._sv_debug_save(speech, 'reseed', sim)
                return True, (f'SV: the unconfirmed reference never matched '
                              f'({self._SV_SEED_MAX_REJECTS} phrases, last similarity={sim:.2f}) '
                              f'— re-seeded from this phrase, accepted')
        self._sv_debug_save(speech, 'reject', sim)
        # Not added to the gallery — only for identity_manager to tell llm_node
        # WHO this other person is (/voice/speaker, other_speaker=True). Kept: if the
        # lips show it was the interlocutor after all, /voice/sv_confirm adds it.
        self._publish_voice_emb(emb, other_speaker=True)
        self._sv_last_reject_emb = emb
        return False, (f'SV: phrase rejected — not the current speaker '
                       f'(similarity={sim:.2f} < {self._sv_threshold:.2f}, {sec:.1f}s speech)')

    def _sv_debug_save(self, speech: np.ndarray, verdict: str, sim) -> None:
        """sv_debug_dir set → keep every judged phrase as WAV for threshold tuning."""
        if not self._sv_debug_dir:
            return
        try:
            import wave
            os.makedirs(self._sv_debug_dir, exist_ok=True)
            # ms: the parts of one split recording are saved within the same second
            name = time.strftime('%Y%m%d_%H%M%S') + f'{int(time.time() * 1000) % 1000:03d}_{verdict}' + (
                f'_{sim:.2f}' if sim is not None else '') + '.wav'
            with wave.open(os.path.join(self._sv_debug_dir, name), 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.rate)
                w.writeframes((np.clip(speech, -1, 1) * 32767).astype(np.int16).tobytes())
        except OSError as e:
            self.get_logger().warn(f'SV debug save failed: {e}')

    # ── Main audio callback ───────────────────────────────────────────────

    def _audio_callback(self, msg: Float32MultiArray):
        if not self._lc_active:
            return
        audio_float32 = np.array(msg.data, dtype=np.float32)

        if self._chunk_size is None:
            self._chunk_size = len(audio_float32)
            chunks_per_sec          = self.rate / self._chunk_size
            self._silence_threshold = int(self._silence_secs * chunks_per_sec)
            self._max_chunks        = int(self.max_phrase_sec * chunks_per_sec)
            self._pre_roll_maxlen   = int(self._pre_roll_secs * chunks_per_sec)
            self._onset_maxlen      = max(1, int(self._onset_secs * chunks_per_sec))
            self.get_logger().info(
                f'chunk_size={self._chunk_size} '
                f'silence_threshold={self._silence_threshold} chunks'
            )

        if self.tts_speaking:
            self._pre_roll_buffer.clear()
            return

        if not self.is_active:
            # Idle: buffer only, VAD runs lazily on activation (_take_pre_roll_speech)
            self._pre_roll_buffer.append(audio_float32)
            if self._pre_roll_maxlen and len(self._pre_roll_buffer) > self._pre_roll_maxlen:
                self._pre_roll_buffer.popleft()

            if (self._stt_sent_time > 0.0
                    and (time.time() - self._stt_sent_time) > self.pipeline_timeout):
                self.get_logger().warn(
                    f'Pipeline timeout ({self.pipeline_timeout:.0f}s) — '
                    f'back to the wake word'
                )
                self._stt_sent_time = 0.0
            return

        with torch.no_grad():
            confidence = self.vad_model(
                torch.from_numpy(audio_float32), self.rate
            ).item()

        # Timeout waiting for the first word
        if not self.audio_buffer:
            if (time.time() - self.activation_time) > self.no_speech_timeout:
                if self._should_keep_listening() and not self._sleeping:
                    self.activation_time = time.time()
                else:
                    self.get_logger().info('Timed out waiting for speech — back to the wake word')
                    self.is_active       = False
                    self.activation_time = 0.0
                return

        if confidence > self.vad_threshold:
            self.silence_counter = 0
            if not self.audio_buffer and self._onset_buf:
                self.audio_buffer.extend(self._onset_buf)
                self._onset_buf.clear()
            self.audio_buffer.append(audio_float32)
            self._speech_audio.append(audio_float32)
            self.speech_chunks += 1

        else:
            if self.audio_buffer:
                self.audio_buffer.append(audio_float32)
                self.silence_counter += 1

                if self.silence_counter >= self._silence_threshold:
                    self._finish_recording(reason='silence')
            else:
                # Silence before the first speech — accumulate the onset buffer
                self._onset_buf.append(audio_float32)
                if self._onset_maxlen and len(self._onset_buf) > self._onset_maxlen:
                    self._onset_buf.popleft()

        # Protection against endless recording, by the number of accepted chunks
        if self.is_active and len(self.audio_buffer) >= self._max_chunks:
            self.get_logger().warn('Phrase length limit reached — forcing the end of recording')
            self._finish_recording(reason='timeout')

    _CARRY_SEC        = 3.0   # a dropped short phrase joins a phrase starting within this
    _WAKE_KEEP_SEC    = 2.0   # wake word during recording: keep this much of the tail

    def _take_carry(self) -> int:
        """Prepends a fresh dropped short phrase to the current one. Returns the
        number of samples prepended (0 if none)."""
        carry, self._carry = self._carry, None
        if carry is None:
            return 0
        chunks, speech, t_fin = carry
        cs = self._chunk_size or 512
        started = time.time() - len(self.audio_buffer) * cs / self.rate
        if started - t_fin > self._CARRY_SEC:
            return 0
        self.audio_buffer  = chunks + self.audio_buffer
        self._speech_audio = speech + self._speech_audio
        self.speech_chunks += len(speech)
        return sum(len(c) for c in chunks)

    def _finish_recording(self, reason: str = 'silence'):
        carried = self._take_carry()
        # During an introduction, short answers are expected (a name, "Ника", "да") —
        # the minimum speech length threshold is lowered so they aren't dropped.
        effective_min_speech_sec = (
            self.min_speech_sec_introducing if self._introducing else self.min_speech_sec
        )
        min_chunks        = int(self.min_phrase_sec * self.rate / self._chunk_size)
        min_speech_chunks = int(effective_min_speech_sec * self.rate / self._chunk_size)
        speech_sec        = self.speech_chunks * self._chunk_size / self.rate

        long_enough = (len(self.audio_buffer) > min_chunks
                       and self.speech_chunks >= min_speech_chunks)
        if carried and long_enough:
            self.get_logger().info(
                f'Short phrase just before ({carried / self.rate:.1f}s) joined to this one')
        sv_ok = True
        if long_enough and self._speech_audio:
            sv_ok, sv_log = self._sv_check_phrase(np.concatenate(self._speech_audio))
            # rclpy pins a severity per call site: info/warn must be separate lines
            if sv_log and sv_ok:
                self.get_logger().info(sv_log)
            elif sv_log:
                self.get_logger().warn(sv_log)

        if long_enough and not sv_ok:
            # Another person, not the current interlocutor. Still transcribed, marked
            # other_speaker: STT routes it to voice_command_other, where llm_node
            # answers it only if the robot is called by name (multi-person talk).
            # Does not touch the STT wait / presence state of the main dialogue.
            # The lip analysis needs it too: the face in view must come out
            # "silent" here (speaker detection).
            # (its voice-id was already published by _sv_check_phrase)
            full_audio = np.concatenate(self.audio_buffer)
            self.get_logger().info(
                f'Phrase from another speaker ({len(full_audio) / self.rate:.1f}s, '
                f'speech={speech_sec:.1f}s) — sending to STT as other_speaker')
            self._publish(full_audio, other_speaker=True)
            # The carried part is not contiguous in time — lips look at this phrase only
            self._publish_segment(full_audio[carried:], sv_rejected=True)
            if self._sv_last_reject_emb is not None:
                self._rejected_embs[self._segment_id] = self._sv_last_reject_emb
                self._sv_last_reject_emb = None
                while len(self._rejected_embs) > 5:
                    self._rejected_embs.pop(next(iter(self._rejected_embs)))
            if self._should_keep_listening() and not self._sleeping:
                self._schedule_relisten(0.5)
        elif long_enough:
            full_audio = np.concatenate(self.audio_buffer)
            duration   = len(full_audio) / self.rate

            self.get_logger().info(
                f'Phrase recorded ({duration:.1f}s, speech={speech_sec:.1f}s, reason={reason}, '
                f'∆wake={time.time()-self.activation_time:.1f}s). Sending to STT...'
            )
            self._stt_sent_time = time.time()
            # Real recognized speech = direct proof of presence, even if
            # /person_present (by face/body from identity_manager) never came
            # True in this session (a purely voice dialogue, no face caught).
            # Without this, _is_person_present() is always False (see its
            # implementation — it only checks the last True from the face),
            # and auto-activation after TTS is disabled for good — the robot
            # stopped hearing the user. Live bug 2026-08-28.
            self._person_last_seen = time.time()
            self._publish(full_audio)
            # The carried part is not contiguous in time — lips look at this phrase only
            self._publish_segment(full_audio[carried:])
        else:
            self.get_logger().info(
                f'Phrase dropped: speech {speech_sec:.1f}s < {effective_min_speech_sec:.1f}s'
                f'{" (introducing)" if self._introducing else ""}'
                f'{" (with the short phrase before it)" if carried else ""} — ignoring'
            )
            if self.speech_chunks > 0:
                # Without the trailing silence, so the join is tight
                keep = self.audio_buffer[:max(0, len(self.audio_buffer) - self.silence_counter)]
                self._carry = (keep, list(self._speech_audio), time.time())
            if self._should_keep_listening() and not self._sleeping:
                self._schedule_relisten(0.5)

        self.audio_buffer    = []
        self.silence_counter = 0
        self.speech_chunks   = 0
        self.is_active       = False
        self._onset_buf.clear()
        self._speech_audio.clear()
        # The gallery is NOT reset — it lives for the whole session until go_idle / robot_sleep

    _SEGMENT_ENV_HZ = 20   # loudness envelope rate for lip sync (50 ms frames)

    def _publish_segment(self, audio: np.ndarray, sv_rejected: bool = False):
        """Time window + loudness envelope of the phrase, for face_tracker to check
        whose lips moved in sync with it (who is speaking). The phrase ends now —
        its audio is the last `len(audio)` samples (chunk latency ~ one chunk)."""
        hop = self.rate // self._SEGMENT_ENV_HZ
        n   = len(audio) // hop
        if n == 0:
            return
        frames   = audio[:n * hop].reshape(n, hop)
        envelope = np.sqrt(np.mean(frames * frames, axis=1))
        vad      = self._segment_vad(audio, n, hop)
        t_end    = time.time()
        self._segment_id += 1
        msg = String()
        msg.data = json.dumps({
            'id':       self._segment_id,
            't_start':  t_end - n * hop / self.rate,
            't_end':    t_end,
            'env_hz':   self._SEGMENT_ENV_HZ,
            'envelope': [round(float(v), 5) for v in envelope],
            'vad':      vad,
            'sv_rejected': sv_rejected,
        })
        self._segment_pub.publish(msg)

    def _segment_vad(self, audio: np.ndarray, n: int, hop: int) -> list[int]:
        """Silero VAD per envelope frame (1 = speech): lip analysis must look only at
        the speech itself — the phrase also holds ~silence_duration_sec of trailing "silence"
        that may contain chewing/laughing (live record 2026-10-01: a bite of food
        in that tail looked like the strongest mouth movement of the phrase).
        Re-run over the finished phrase (a few ms), the same way as
        _take_pre_roll_speech; the live VAD state is reset before and after."""
        cs = self._chunk_size or 512
        self.vad_model.reset_states()
        with torch.no_grad():
            speech = [self.vad_model(torch.from_numpy(np.ascontiguousarray(
                          audio[i:i + cs])), self.rate).item() > self.vad_threshold
                      for i in range(0, len(audio) - cs + 1, cs)]
        self.vad_model.reset_states()
        if not speech:
            return [0] * n
        return [int(speech[min(len(speech) - 1, (k * hop + hop // 2) // cs)]) for k in range(n)]

    def _publish(self, audio: np.ndarray, other_speaker: bool = False):
        msg = Float32MultiArray()
        dim = MultiArrayDimension()
        dim.label  = 'sample_rate'
        dim.size   = len(audio)
        dim.stride = self.rate
        msg.layout.dim = [dim]
        if other_speaker:
            # Marker dim (size 0, no data): STT → voice_command_other
            other = MultiArrayDimension()
            other.label = 'other_speaker'
            msg.layout.dim.append(other)
        msg.data = audio.tolist()
        self.publisher_.publish(msg)


def main():
    rclpy.init()
    node = VoiceDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
