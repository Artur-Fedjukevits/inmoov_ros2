#!/usr/bin/env python3
"""
tts_node.py — ROS2 Action Server for speech synthesis and playback.

Action: /speak (inmoov_msgs/action/Speak)
  Goal:     text, voice, rate
  Feedback: status, progress, bytes_played
  Result:   success, message, audio_sec

Features:
  - Preemption: a new goal immediately interrupts the current playback
  - Fallback: if the primary TTS server is unavailable — a local one
  - Publishes tts_speaking (Bool) for voice_detector (backward compatibility)

Clients:
  - llm_node      — text replies to voice commands
  - inmoov_cognition — speech as part of behavior (gestures + speech)

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import math
import os
import threading
import time

import numpy as np
import requests
import sounddevice as sd

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

from inmoov_msgs.action import Speak


class TTSNode(LifecycleNode):
    # Incident 2026-08-07: when PipeWire/ALSA (Jabra) drops out, PortAudio's
    # stream.write() falls into an internal C-level busy-retry XRun loop
    # (AlsaRestart/PaAlsaStream_HandleXrun) with no sleep between attempts —
    # Python flags (my_abort) are not checked inside such a write(), so it
    # cannot be interrupted gracefully. Within seconds this hogs the CPU and
    # floods journald (386k messages/s were observed), which hangs the whole
    # machine. The watchdog below is the only reliable way out: if there is
    # no progress for longer than _STALL_TIMEOUT_SEC after playback has
    # started, the process is considered hopelessly stuck and is killed
    # entirely; respawn=True in the launch file brings up a clean process
    # after respawn_delay seconds.
    _STALL_TIMEOUT_SEC = 8.0

    def __init__(self):
        super().__init__('tts_node')

        # Threading state stays in __init__ — independent of the lifecycle
        self._execute_lock       = threading.Lock()
        self._abort_lock         = threading.Lock()
        self._abort_event        = threading.Event()
        self._goals_pending      = 0
        self._goals_pending_lock = threading.Lock()
        self._cancel_queued      = threading.Event()
        # Face emotion shown for the current speech (see _execute_speak) —
        # held on /face_expression_hold until the LAST pending goal finishes
        # (same pattern as tts_speaking below), protected by the same
        # _goals_pending_lock.
        self._active_face_emotion: str | None = None
        self._FACE_EMOTIONS = frozenset(('neutral', 'happy', 'sad', 'surprise'))

        # Stall-watchdog (see the comment at _STALL_TIMEOUT_SEC)
        self._playback_active      = threading.Event()
        self._last_write_ts        = 0.0
        # Incident 2026-08-15: the watchdog compared "now" with the moment of
        # the LAST SUCCESSFUL write() — which conflates two different things.
        # Between the HTTP chunks of /tts/stream there are legitimate pauses
        # (the server is still generating the next piece of audio) — during
        # which write() is not called at all, writes_in_progress=0. The
        # watchdog used to take such a pause for a "hung write()" and killed
        # the process after 8s, although ALSA/PortAudio had nothing to do with
        # it. Now the watchdog looks only at the time INSIDE the write() call
        # itself — i.e. the real symptom of the busy-retry XRun loop
        # (incident 2026-08-07), not at pauses between data arriving over the
        # network.
        self._write_in_progress_since = 0.0
        self._stall_watchdog_ready = False

        # Placeholders — filled in on_configure / on_activate
        self._session       = None
        self._speaking_pub  = None
        self._jaw_pub       = None
        self._face_expr_pub = None
        self._action_server = None
        self._active_url    = None
        self._output_device = None

    # ── Lifecycle: Phase 2 ─────────────────────────────────────────────────

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores re-declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('tts_server_url',       'http://192.168.10.118:8000')
        # Optional backup TTS server; empty = no fallback. (The local CosyVoice3 on the
        # NUC's ROCm iGPU was removed — too slow for live dialogue.)
        self._dp('tts_fallback_url',     '')
        self._dp('chunk_size',           4096)
        self._dp('connect_timeout_sec',  5.0)
        self._dp('timeout_sec',          30.0)
        self._dp('output_device_name',   '')
        self._dp('jaw_closed',           10)
        self._dp('jaw_open',             90)
        self._dp('jaw_rms_threshold',    300.0)
        self._dp('jaw_rms_max',          6000.0)
        self._dp('jaw_speed_deg_per_sec', 400.0)

        self.primary_url     = self.get_parameter('tts_server_url').value
        self.fallback_url    = self.get_parameter('tts_fallback_url').value
        self.chunk_size      = self.get_parameter('chunk_size').value
        self.connect_timeout = self.get_parameter('connect_timeout_sec').value
        self.timeout_sec     = self.get_parameter('timeout_sec').value
        self._active_url     = self.primary_url
        self._jaw_closed     = self.get_parameter('jaw_closed').value
        self._jaw_open       = self.get_parameter('jaw_open').value
        self._jaw_rms_threshold = self.get_parameter('jaw_rms_threshold').value
        self._jaw_rms_max       = self.get_parameter('jaw_rms_max').value
        self._jaw_speed_rad_s   = math.radians(
            self.get_parameter('jaw_speed_deg_per_sec').value)

        self._session = requests.Session()

        self.create_subscription(Bool,   '/tts_cancel_queue',  self._cancel_queue_cb, 10)
        self._speaking_pub = self.create_lifecycle_publisher(Bool, 'tts_speaking', 10)
        self._jaw_pub      = self.create_lifecycle_publisher(JointState, '/face_command', 10)
        # Held facial expression for the duration of speech (see _execute_speak) —
        # separate from the animated one-shot /face_expression (greet/farewell/BT).
        self._face_expr_pub = self.create_lifecycle_publisher(
            String, '/face_expression_hold', 10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._speaking_pub.on_activate(state)
        self._jaw_pub.on_activate(state)
        self._face_expr_pub.on_activate(state)

        # Incident 2026-08-15: on an emergency os._exit(1) from
        # _stall_watchdog_loop the process is killed in the middle of
        # speaking=True — the finally block with self._publish_speaking(False)
        # has no time to run. After the restart voice_detector_node stays
        # convinced forever that TTS is speaking and ignores all microphone
        # input (voice_detector_node._audio_callback:
        # `if self.tts_speaking: return`). We publish False right on activation
        # — on a normal start this is a no-op (the receivers are already
        # initialized as False), but it clears the stuck state after an
        # emergency restart.
        self._publish_speaking(False)

        # Find the output device (needs a live PipeWire)
        self._output_device = self._find_output_device(
            self.get_parameter('output_device_name').value)

        # Check the servers — WARN if unavailable, but not FAILURE (the server may come up later)
        self._check_servers()

        if not self._stall_watchdog_ready:
            self._stall_watchdog_ready = True
            threading.Thread(
                target=self._stall_watchdog_loop, daemon=True,
                name='tts_stall_watchdog').start()

        self._action_server = ActionServer(
            self,
            Speak,
            'speak',
            execute_callback=self._execute_speak,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info(f'TTS Action Server ready. Server: {self._active_url}')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        # Interrupt the current playback
        with self._abort_lock:
            self._abort_event.set()

        if self._action_server is not None:
            self._action_server.destroy()
            self._action_server = None

        self._speaking_pub.on_deactivate(state)
        self._jaw_pub.on_deactivate(state)
        self._face_expr_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        if self._session:
            self._session.close()
            self._session = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        with self._abort_lock:
            self._abort_event.set()
        if self._session:
            self._session.close()
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        if self._session:
            self._session.close()
        return TransitionCallbackReturn.SUCCESS

    # ── Stall-watchdog (see the comment at _STALL_TIMEOUT_SEC) ────────────

    def _stall_watchdog_loop(self):
        """Lives for the whole lifetime of the process (not tied to the lifecycle
        state: one can get stuck in the C loop during on_deactivate too).

        Looks ONLY at the time spent inside an active stream.write() call —
        see the comment at self._write_in_progress_since in __init__. Pauses
        between chunks (waiting for data over the network/WS) don't count."""
        while True:
            time.sleep(1.0)
            if not self._playback_active.is_set():
                continue
            started = self._write_in_progress_since
            if started <= 0.0:
                continue  # not inside write() right now — waiting for data, that's normal
            stalled = time.time() - started
            if stalled > self._STALL_TIMEOUT_SEC:
                self.get_logger().fatal(
                    f'TTS: stream.write() has made no progress for {stalled:.1f}s — '
                    f'looks like a stuck ALSA/PipeWire XRun loop inside '
                    f'PortAudio. Emergency process restart.'
                )
                import sys
                sys.stderr.flush()
                os._exit(1)  # SIGKILL-like exit: cut the C level off immediately

    def _write_chunk(self, stream, data: bytes):
        """stream.write() with a heartbeat for the stall watchdog.
        _write_in_progress_since marks the window of the REAL write() call —
        the watchdog reacts only while we are inside it."""
        self._write_in_progress_since = time.time()
        try:
            stream.write(data)
        finally:
            self._write_in_progress_since = 0.0
        self._last_write_ts = time.time()

    # ── Server checks ──────────────────────────────────────────────────────

    def _check_servers(self):
        if self._probe_server(self.primary_url):
            self._active_url = self.primary_url
            self.get_logger().info(f'TTS: primary server is reachable ({self.primary_url})')
        elif self.fallback_url and self._probe_server(self.fallback_url):
            self._active_url = self.fallback_url
            self.get_logger().warn(
                f'Primary TTS unavailable! Fallback: {self.fallback_url}')
        else:
            self.get_logger().error(
                'TTS server(s) unavailable!'
                + ('' if self.fallback_url else ' (no fallback configured)'))

    def _probe_server(self, url: str) -> bool:
        try:
            r = self._session.get(f'{url}/health', timeout=self.connect_timeout)
            info = r.json()
            self.get_logger().info(
                f'  {url}: GPU={info.get("gpu", "CPU")}, '
                f'VRAM={info.get("vram_used_mb", "?")}/'
                f'{info.get("vram_total_mb", "?")} MB, '
                f'SR={info.get("sample_rate", "?")}'
            )
            return True
        except Exception:
            return False

    # ── TTS queue flush (preemption on user interruption) ─────────────────

    def _cancel_queue_cb(self, msg: Bool):
        """Cancels the current goal and marks all pending ones for rejection."""
        if not msg.data:
            return
        with self._goals_pending_lock:
            if self._goals_pending == 0:
                return  # the action server is not playing — nothing to cancel
            self._cancel_queued.set()
        with self._abort_lock:
            self._abort_event.set()  # interrupts the current playback
        self.get_logger().info('TTS: queue flushed (user interruption)')

    # ── Action callbacks ───────────────────────────────────────────────────

    def _goal_callback(self, goal_request):
        """Accept all goals. Preemption happens inside execute."""
        text_preview = (goal_request.text[:40] + '...') if len(goal_request.text) > 40 else goal_request.text
        style_info = ''
        if goal_request.voice:
            style_info += f' emotion="{goal_request.voice}"'
        if goal_request.rate not in (0.0, 1.0):
            style_info += f' speed={goal_request.rate:.2f}'
        self.get_logger().info(f'New Speak goal: "{text_preview}"{style_info}')
        with self._goals_pending_lock:
            self._goals_pending += 1
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        """Accept cancellation requests from clients."""
        self.get_logger().info('TTS goal cancellation request')
        return CancelResponse.ACCEPT

    # ── Execute: main logic ────────────────────────────────────────────────

    def _execute_speak(self, goal_handle):
        """
        Runs in a separate thread (ReentrantCallbackGroup).

        Preemption: on start we create a new abort_event and signal the old one.
        The old execute loop checks its own event and exits.
        """
        # ── Wait for the previous goal to finish (queueing, not preemption) ──
        # Timeout: if the previous one is stuck — skip it after 35s
        if not self._execute_lock.acquire(timeout=35.0):
            self.get_logger().warn('TTS: previous goal is stuck — skipping')
            goal_handle.abort()
            result = Speak.Result()
            result.success = False
            result.message = 'Queue timeout'
            with self._goals_pending_lock:
                self._goals_pending -= 1
            return result

        my_abort = threading.Event()
        with self._abort_lock:
            self._abort_event = my_abort

        text    = goal_handle.request.text.strip()
        emotion = goal_handle.request.voice.strip()
        if not text:
            goal_handle.abort()
            result = Speak.Result()
            result.success = False
            result.message = 'Empty text'
            with self._goals_pending_lock:
                self._goals_pending -= 1
            self._execute_lock.release()
            return result

        # If the queue was flushed (the user interrupted) — reject this chunk
        if self._cancel_queued.is_set():
            goal_handle.abort()
            result = Speak.Result()
            result.success = False
            result.message = 'Queue flushed'
            with self._goals_pending_lock:
                self._goals_pending -= 1
                if self._goals_pending == 0:
                    self._cancel_queued.clear()
            self._execute_lock.release()
            return result

        feedback      = Speak.Feedback()
        bytes_played  = 0
        start_time    = time.time()

        def _fb(status: str, progress: float = -1.0):
            feedback.status      = status
            feedback.progress    = progress
            feedback.bytes_played = bytes_played
            goal_handle.publish_feedback(feedback)

        _fb('connecting')

        # ── URL selection with fallback ────────────────────────────────────
        urls = [self._active_url]
        other = self.fallback_url if self._active_url == self.primary_url else self.primary_url
        if other and other != self._active_url:
            urls.append(other)

        success   = False
        error_msg = ''

        self._publish_speaking(True)
        # Facial expression in sync with the voice for the whole phrase (see
        # set_voice_style in llm_node). Empty (greet/farewell) — leave the face alone.
        face_emotion = emotion.lower()
        if face_emotion in self._FACE_EMOTIONS:
            with self._goals_pending_lock:
                self._active_face_emotion = face_emotion
            self._publish_face_emotion(face_emotion)
        try:
            for url in urls:
                if my_abort.is_set() or goal_handle.is_cancel_requested:
                    break
                try:
                    success, bytes_played, error_msg = self._stream_and_play(
                        url, text, emotion, goal_handle, my_abort, _fb
                    )
                    if success or error_msg not in ('connection_error',):
                        if url != self._active_url:
                            self.get_logger().warn(f'Switched to the fallback TTS: {url}')
                            self._active_url = url
                        break
                except Exception as e:
                    error_msg = str(e)
                    self.get_logger().error(f'TTS error [{url}]: {e}')
        finally:
            revert_face = False
            with self._goals_pending_lock:
                self._goals_pending -= 1
                if self._goals_pending == 0:
                    self._cancel_queued.clear()
                    # Last pending goal — if a non-neutral emotion was shown
                    # for it, return the face to neutral. Works both on normal
                    # completion and on cancel/abort (both paths end up here).
                    if self._active_face_emotion not in (None, 'neutral'):
                        revert_face = True
                    self._active_face_emotion = None
            if revert_face:
                self._publish_face_emotion('neutral')
            self._publish_speaking(False)
            self._execute_lock.release()

        # ── Result ────────────────────────────────────────────────────────
        audio_sec = time.time() - start_time

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result = Speak.Result()
            result.success   = False
            result.message   = 'Canceled by client'
            result.audio_sec = audio_sec
            return result

        if success:
            self.get_logger().info(f'TTS played in {audio_sec:.1f}s')
            goal_handle.succeed()
        else:
            goal_handle.abort()

        result = Speak.Result()
        result.success   = success
        result.message   = '' if success else error_msg
        result.audio_sec = audio_sec
        return result

    def _stream_and_play(
        self, url: str, text: str, emotion: str,
        goal_handle, my_abort: threading.Event, fb
    ) -> tuple[bool, int, str]:
        """
        Streams audio from the TTS server and plays it.
        Returns (success, bytes_played, error_msg).

        emotion — name of the server's voice preset (OmniVoice/audio.cpp,
        server.json → voice_presets): "neutral"/"happy"/"sad"/"surprise".
        Voice cloning dominates over textual instruct instructions
        (a CosyVoice3 legacy), hence only pre-defined presets.
        Empty/unknown name = the server falls back to "neutral".
        """
        body: dict = {'text': text}
        if emotion:
            body['emotion'] = emotion

        # TTFA (Time To First Audio) — from sending the POST to the first
        # received audio byte: connect + server time until it starts emitting
        # the stream. This latency determines the perceived "responsiveness"
        # of the TTS — see the discussion of the CosyVoice3 → OmniVoice
        # migration (MIGRATION_NOTES.md).
        t_req_start = time.time()
        ttfa_logged = False

        try:
            with self._session.post(
                f'{url}/tts/stream',
                json=body,
                stream=True,
                timeout=(self.connect_timeout, self.timeout_sec),
            ) as resp:
                resp.raise_for_status()
                fb('synthesizing')

                sample_rate    = self._parse_sample_rate(resp.headers)
                content_length = int(resp.headers.get('Content-Length', -1))
                bytes_played   = 0

                with sd.RawOutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype='int16',
                    device=self._output_device,
                ) as stream:
                    fb('playing')
                    self._last_write_ts = time.time()
                    self._playback_active.set()
                    try:
                        for chunk in resp.iter_content(chunk_size=self.chunk_size):
                            # Preemption / cancellation check between chunks
                            if my_abort.is_set() or goal_handle.is_cancel_requested:
                                stream.abort()
                                resp.close()
                                self._publish_jaw(self._jaw_closed)
                                return False, bytes_played, 'aborted'

                            if chunk:
                                if not ttfa_logged:
                                    self.get_logger().info(
                                        f'TTFA: {time.time() - t_req_start:.2f}s '
                                        f'[{url}]')
                                    ttfa_logged = True
                                self._write_chunk(stream, chunk)
                                bytes_played += len(chunk)
                                progress = (bytes_played / content_length
                                            if content_length > 0 else -1.0)
                                fb('playing', progress)
                                self._publish_jaw(self._chunk_to_jaw(chunk))
                    finally:
                        self._playback_active.clear()

                self._publish_jaw(self._jaw_closed)

                if bytes_played == 0:
                    return False, 0, 'Empty stream from the TTS server'

                return True, bytes_played, ''

        except requests.exceptions.ConnectionError as e:
            self.get_logger().warn(f'TTS ConnectionError [{url}]: {e}')
            return False, 0, 'connection_error'
        except requests.exceptions.Timeout:
            return False, 0, f'Timeout ({self.timeout_sec}s)'
        except requests.exceptions.HTTPError as e:
            return False, 0, f'HTTP error: {e}'

    # ── Helper methods ─────────────────────────────────────────────────────

    def _chunk_to_jaw(self, chunk: bytes) -> int:
        """Computes the RMS of a PCM int16 chunk and maps it to a jaw position."""
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < self._jaw_rms_threshold:
            return self._jaw_closed
        t = min(1.0, (rms - self._jaw_rms_threshold) /
                     (self._jaw_rms_max - self._jaw_rms_threshold))
        return int(self._jaw_closed + t * (self._jaw_open - self._jaw_closed))

    def _publish_jaw(self, position: int):
        """Publishes the jaw position via /face_command (JointState).
        position — degrees [jaw_closed..jaw_open], converted to radians
        with (deg - 90) * π/180 (center_deg=90, as everywhere in the face protocol).
        velocity  — speed in rad/s; arduino_left_node converts it to a step
                    and sends CMD_SET_SPEEDS only when it changes.
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = ['jaw']
        msg.position = [(float(position) - 90.0) * math.pi / 180.0]
        msg.velocity = [self._jaw_speed_rad_s]
        self._jaw_pub.publish(msg)

    def _publish_speaking(self, speaking: bool):
        msg = Bool()
        msg.data = speaking
        self._speaking_pub.publish(msg)

    def _publish_face_emotion(self, name: str):
        """Held facial expression for the duration of speech, see _execute_speak.
        Separate from the animated /face_expression — face_expressions_node
        applies the pose statically, without a built-in auto-return."""
        msg = String()
        msg.data = name
        self._face_expr_pub.publish(msg)

    def _find_output_device(self, name: str):
        if not name:
            return None
        name_lower = name.lower()
        # Prefer PipeWire/pulse virtual device (no "(hw:" in name).
        # Direct hw: access conflicts with PipeWire exclusive ownership → silent playback.
        for i, d in enumerate(sd.query_devices()):
            if '(hw:' not in d['name'] and d['max_output_channels'] > 0 \
                    and name_lower in d['name'].lower():
                self.get_logger().info(f'Audio output: [{i}] {d["name"]}')
                return i
        # hw: device found but PipeWire manages it → fall back to system default
        for i, d in enumerate(sd.query_devices()):
            if '(hw:' in d['name'] and d['max_output_channels'] > 0 \
                    and name_lower in d['name'].lower():
                self.get_logger().warn(
                    f'Device "{d["name"]}" is only available via raw ALSA — '
                    f'using the system default (PipeWire)')
                return None
        self.get_logger().warn(f'Output device "{name}" not found, using the system default')
        return None

    @staticmethod
    def _parse_sample_rate(headers) -> int:
        content_type = headers.get('Content-Type', '')
        for part in content_type.split(';'):
            part = part.strip()
            if part.startswith('rate='):
                try:
                    return int(part[5:])
                except ValueError:
                    pass
        return int(headers.get('X-Sample-Rate', 24000))




def main():
    rclpy.init()
    node = TTSNode()
    # MultiThreadedExecutor is required for ReentrantCallbackGroup:
    # the execute callback of a new goal must start while the old one is still running
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
