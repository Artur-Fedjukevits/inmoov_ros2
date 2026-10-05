#!/usr/bin/env python3

"""
audio_source_node.py
====================
The only node that opens the microphone.
Publishes normalized float32 chunks to the 'raw_audio' topic.

Lifecycle:
  on_configure — declare params, create publisher
  on_activate  — open the audio stream; FAILURE if PipeWire is not ready (→ retry)
  on_deactivate — close the stream, stop timers

Parameters:
  sample_rate  (int)   — 16000
  chunk_size   (int)   — 512  (32 ms; optimal for Silero VAD and OWW)
  device_index (int)   — -1 = system default device
  device_name  (str)   — search by name substring
  watchdog_sec (float) — restart if no audio for N seconds (default 5.0)

/diagnostics ('microphone'): ERROR when no audio chunk for watchdog_sec,
WARN while the device returns pure zeros (hardware Mute), with chunk age.

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import os
import queue
import re
import subprocess
import threading
import time
from collections import deque

import rclpy
from diagnostic_msgs.msg import DiagnosticStatus
from diagnostic_updater import Updater
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import pyaudio
import numpy as np


class AudioSourceNode(LifecycleNode):
    # Incident 2026-08-07: on a PipeWire/ALSA failure the reader thread can hang
    # inside a blocking stream.read() in PortAudio's C-level busy-retry XRun loop
    # (same nature as tts_node's — see the comment there). _restart_stream()
    # below does NOT kill such a thread (see the comment in _restart_stream) — it
    # deliberately "abandons" it, hoping PipeWire recovers and read() itself
    # returns OSError. If that doesn't happen, the thread burns a full CPU core
    # forever and more threads pile up with every new restart. The only
    # protection is to detect frequent restarts in a row (a symptom of a stuck
    # reader that the watchdog can't cure) and kill the whole process:
    # respawn=True in the launch file will bring up a clean process after
    # respawn_delay seconds.
    _RESTART_STORM_COUNT      = 5     # restarts...
    _RESTART_STORM_WINDOW_SEC = 30.0  # ...within this window → emergency exit

    # Incident 2026-08-09: pipewire-pulse (the PulseAudio protocol server)
    # hung dead — confirmed externally via `pactl info`, which didn't answer
    # ANY client at all (not just this node). _restart_wireplumber() only
    # fixes WirePlumber (the graph policy manager) — that's a different
    # systemd unit and it cannot heal a hung pipewire-pulse/pipewire. Because
    # of this the node spun in an endless "restart WP → didn't help → backoff
    # pause → retry" loop and was NOT caught by the _RESTART_STORM detector,
    # since backoff stretches the attempts out longer than
    # _RESTART_STORM_WINDOW_SEC. So: if N consecutive successful WirePlumber
    # restarts fail to bring back source_ok — escalate and restart the whole
    # stack (pipewire, pipewire-pulse, wireplumber).
    _PW_ESCALATE_AFTER = 2

    def __init__(self):
        super().__init__('audio_source_node')
        self._restart_times = deque(maxlen=self._RESTART_STORM_COUNT)
        self._pa                  = None
        self._stream              = None
        self._last_ok             = 0.0
        self._diag                = None
        self._restart_count       = 0
        self._timers              = []
        self._pub                 = None
        self._audio_queue         = queue.Queue(maxsize=5)
        self._stream_running      = False
        self._read_thread         = None
        self._last_wp_restart     = 0.0
        self._wp_restart_fail_streak = 0
        self._last_full_pw_restart   = 0.0
        self._recovery_lock       = threading.Lock()
        self._restart_in_progress = False
        # Exponential backoff for persistent PipeWire failures
        self._retry_backoff_sec   = 5.0
        self._next_retry_time     = 0.0
        self._MAX_BACKOFF_SEC     = 120.0

    # ── Lifecycle callbacks ────────────────────────────────────────────────

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores re-declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('sample_rate',              16000)
        self._dp('chunk_size',               512)
        self._dp('device_index',             -1)
        self._dp('device_name',              '')
        self._dp('watchdog_sec',             5.0)
        self._dp('pa_source_check',          '')   # substring in pactl get-default-source; empty = don't check
        self._dp('jabra_card_name',          'Jabra_Speak2')  # substring to find the card in pactl list cards
        self._dp('jabra_profile',            'output:analog-stereo+input:mono-fallback')
        self._dp('zero_wp_check_count',      500)  # zero chunks (~15s) → check PipeWire
        self._dp('wp_restart_cooldown_sec',  90.0)
        self._dp('pw_full_restart_cooldown_sec', 180.0)

        self.rate                    = self.get_parameter('sample_rate').value
        self.chunk_size              = self.get_parameter('chunk_size').value
        self._device_index           = self.get_parameter('device_index').value
        self._device_name            = self.get_parameter('device_name').value
        self._watchdog_sec           = self.get_parameter('watchdog_sec').value
        self._pa_source_check        = self.get_parameter('pa_source_check').value
        self._jabra_card_name        = self.get_parameter('jabra_card_name').value
        self._jabra_profile          = self.get_parameter('jabra_profile').value
        self._zero_wp_check_count    = self.get_parameter('zero_wp_check_count').value
        self._wp_restart_cooldown    = self.get_parameter('wp_restart_cooldown_sec').value
        self._pw_full_restart_cooldown = self.get_parameter('pw_full_restart_cooldown_sec').value

        self._pub = self.create_lifecycle_publisher(Float32MultiArray, 'raw_audio', 20)
        if self._diag is None:
            self._diag = Updater(self, period=1.0)
            device = self._device_name or 'default'
            self._diag.setHardwareID(f'audio input "{device}"')
            self._diag.add('microphone', self._diagnose)
        return TransitionCallbackReturn.SUCCESS

    def _diagnose(self, stat):
        active = bool(getattr(self, '_stream_running', False))
        age = time.time() - self._last_ok if self._last_ok else float('inf')
        stat.add('stream_open', str(self._stream is not None))
        stat.add('last_chunk_age_sec', f'{age:.1f}')
        stat.add('zero_chunks_in_a_row', str(self._zero_streak))
        if not active:
            stat.summary(DiagnosticStatus.OK, 'inactive')
        elif age > self._watchdog_sec:
            stat.summary(DiagnosticStatus.ERROR, f'no audio for {age:.0f} s')
        elif self._zero_streak >= self._ZERO_WARN_AFTER:
            stat.summary(DiagnosticStatus.WARN, 'microphone returns zeros (Mute button?)')
        else:
            stat.summary(DiagnosticStatus.OK, 'streaming')
        return stat

    def on_activate(self, state):
        self._pub.on_activate(state)

        if not self._open_stream():
            self._pub.on_deactivate(state)
            return TransitionCallbackReturn.FAILURE

        self._stream_running = True
        self._read_thread = threading.Thread(
            target=self._stream_reader, daemon=True, name='audio_reader')
        self._read_thread.start()

        timer_period = (self.chunk_size / self.rate) * 0.9
        self._timers.append(self.create_timer(timer_period, self._drain_and_publish))
        self._timers.append(self.create_timer(self._watchdog_sec, self._watchdog))
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self._stream_running = False
        for t in self._timers:
            self.destroy_timer(t)
        self._timers.clear()
        # Signal the reader thread via self._stream = None BEFORE calling pa.terminate().
        # Without this, pa.terminate() while a read() is still blocked → SIGABRT.
        self._safe_close_stream()
        self._pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self._safe_close_stream()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self._safe_close_stream()
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        self._safe_close_stream()
        return TransitionCallbackReturn.SUCCESS

    # ── Opening the stream ───────────────────────────────────────────────

    def _check_pa_source(self) -> bool:
        """Checks that the PulseAudio default source contains self._pa_source_check.
        Returns False if the device is not found — the stream is not opened."""
        if not self._pa_source_check:
            return True
        try:
            out = subprocess.check_output(
                ['pactl', 'get-default-source'], timeout=3, text=True
            ).strip()
            if self._pa_source_check.lower() not in out.lower():
                self.get_logger().error(
                    f'pa_source_check: default source "{out}" does not contain '
                    f'"{self._pa_source_check}" — Jabra is not connected, not opening the stream'
                )
                return False
            return True
        except Exception as e:
            self.get_logger().error(f'pa_source_check: pactl failed: {e}')
            return False

    def _open_stream(self) -> bool:
        self._close_stream()

        if not self._check_pa_source():
            return False

        try:
            self._pa = pyaudio.PyAudio()

            device_index = self._device_index
            if self._device_name:
                device_index = self._find_device_by_name(self._device_name)

            open_kwargs = dict(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )
            if device_index >= 0:
                open_kwargs['input_device_index'] = device_index
                dev_name = self._pa.get_device_info_by_index(device_index).get('name', '?')
                self.get_logger().info(f'Device: [{device_index}] {dev_name}')
            else:
                self.get_logger().info('Device: system default')

            self._stream = self._pa.open(**open_kwargs)
            self._last_ok = time.time()
            self.get_logger().info(
                f'AudioSource started: {self.rate} Hz, chunk={self.chunk_size} '
                f'({1000 * self.chunk_size / self.rate:.0f} ms)'
            )
            return True

        except Exception as e:
            self.get_logger().error(f'Failed to open audio stream: {e}')
            self._close_stream()
            return False

    def _safe_close_stream(self):
        """Closes the stream safely: nulls out self._stream BEFORE terminate(),
        so the reader thread (possibly blocked in read()) doesn't trigger a SIGABRT."""
        stream, pa = self._stream, self._pa
        self._stream = None  # reader thread sees None and exits
        self._pa = None
        try:
            if stream is not None:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
        except Exception:
            pass
        try:
            if pa is not None:
                pa.terminate()
        except Exception:
            pass

    def _close_stream(self):
        try:
            if self._stream is not None:
                if self._stream.is_active():
                    self._stream.stop_stream()
                self._stream.close()
        except Exception:
            pass
        finally:
            self._stream = None

        try:
            if self._pa is not None:
                self._pa.terminate()
        except Exception:
            pass
        finally:
            self._pa = None

    def _find_device_by_name(self, name: str) -> int:
        name_lower = name.lower()
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if name_lower in info['name'].lower():
                return i
        self.get_logger().warn(f'Device "{name}" not found, using system default')
        return -1

    # ── Reading in a background thread (doesn't block the executor) ──────

    def _stream_reader(self):
        """Background thread: reads audio from PyAudio into a queue.
        The blocking stream.read() is isolated from the ROS2 executor — the watchdog can react."""
        while self._stream_running:
            if self._stream is None:
                time.sleep(0.05)
                continue
            try:
                data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                try:
                    self._audio_queue.put_nowait(data)
                except queue.Full:
                    pass
            except OSError as e:
                self.get_logger().warn(f'Audio read error: {e}')
                break  # Watchdog will detect the pause and restart the stream

    # Counter of consecutive all-zero chunks — for detecting a hardware Mute
    _zero_streak: int = 0
    _ZERO_WARN_AFTER: int = 100   # ~3s at chunk=512, rate=16000

    def _drain_and_publish(self):
        """Timer callback: drains chunks from the queue and publishes them (non-blocking)."""
        try:
            data = self._audio_queue.get_nowait()
        except queue.Empty:
            return

        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # Detect hardware Mute (Jabra button): the device returns strictly
        # zero data. We don't stop publishing, but we warn.
        if audio.max() == 0.0 and audio.min() == 0.0:
            self._zero_streak += 1
            if self._zero_streak == self._ZERO_WARN_AFTER:
                self.get_logger().warn(
                    'AudioSource: microphone returns zeros — '
                    'check the Mute button on the Jabra (red indicator)')
            elif self._zero_streak > self._ZERO_WARN_AFTER and self._zero_streak % 300 == 0:
                self.get_logger().warn('AudioSource: microphone still muted (Mute button)')
            if self._zero_streak == self._zero_wp_check_count:
                threading.Thread(
                    target=self._check_and_maybe_restart_wireplumber,
                    daemon=True, name='wp_check').start()
        else:
            if self._zero_streak >= self._ZERO_WARN_AFTER:
                self.get_logger().info('AudioSource: microphone unmuted — signal is back')
            self._zero_streak = 0

        msg = Float32MultiArray()
        dim = MultiArrayDimension()
        dim.label  = 'sample_rate'
        dim.size   = len(audio)
        dim.stride = self.rate
        msg.layout.dim = [dim]
        msg.data = audio.tolist()

        self._pub.publish(msg)
        self._last_ok = time.time()

    # ── Watchdog — runtime recovery (stream thread died while running) ────

    def _watchdog(self):
        if self._stream is None:
            now = time.time()
            if now < self._next_retry_time:
                return  # still in backoff
            self.get_logger().warn(
                f'Watchdog: stream not open — restarting (backoff={self._retry_backoff_sec:.0f}s)')
            self._next_retry_time = now + self._retry_backoff_sec
            self._retry_backoff_sec = min(self._retry_backoff_sec * 2, self._MAX_BACKOFF_SEC)
            self._restart_stream()
            return

        elapsed = time.time() - self._last_ok
        if elapsed > self._watchdog_sec:
            self.get_logger().warn(
                f'Watchdog: no audio for {elapsed:.1f}s — restarting the stream'
            )
            self._restart_stream()

    def _restart_stream(self):
        if self._restart_in_progress:
            return
        self._restart_in_progress = True
        try:
            self._restart_count += 1

            now = time.time()
            self._restart_times.append(now)
            if len(self._restart_times) >= self._RESTART_STORM_COUNT and \
                    now - self._restart_times[0] < self._RESTART_STORM_WINDOW_SEC:
                self.get_logger().fatal(
                    f'AudioSource: {len(self._restart_times)} restarts in '
                    f'{now - self._restart_times[0]:.1f}s — looks like a stuck '
                    f'reader thread in a busy-retry loop (cannot be cured in-process). '
                    f'Emergency process restart.'
                )
                import sys
                sys.stderr.flush()
                os._exit(1)

            self.get_logger().info(f'Restarting audio stream (attempt #{self._restart_count})...')
            # Signal the reader thread to stop and IMMEDIATELY forget the old stream/pa.
            # We do NOT call stop_stream()/close()/terminate() — if the thread is blocked
            # in read(), touching the PA context from another thread will trigger SIGABRT.
            # The old (daemon) thread will die on its own once PipeWire recovers and
            # returns OSError.
            self._stream_running = False
            self._stream = None   # Reader sees None and exits on its next loop
            self._pa     = None   # GC will clean up without an explicit terminate()
            self._read_thread = None
            # Reset the queue
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break
            if self._open_stream():
                self._stream_running = True
                self._read_thread = threading.Thread(
                    target=self._stream_reader, daemon=True, name='audio_reader')
                self._read_thread.start()
                self._retry_backoff_sec = 5.0  # reset backoff on success
                self._next_retry_time   = 0.0
                self.get_logger().info('Audio stream restored')
            else:
                self.get_logger().error('Restart failed — checking PipeWire...')
                threading.Thread(
                    target=self._check_and_maybe_restart_wireplumber,
                    daemon=True, name='wp_check').start()
        finally:
            self._restart_in_progress = False

    # ── PipeWire / WirePlumber recovery ──────────────────────────────────────

    def _pipewire_jabra_state(self):
        """Checks the state of the Jabra device in PipeWire.
        Returns (has_card, card_id, profile_ok, source_ok)."""
        if not self._jabra_card_name:
            return True, '', True, True
        has_card, card_id, profile_ok, source_ok = False, '', False, False
        try:
            raw = subprocess.check_output(['pactl', 'list', 'cards'], timeout=5, text=True)
        except Exception as e:
            self.get_logger().warn(f'pactl list cards: {e}')
            return False, '', False, False

        for sec in re.split(r'\n(?=Card #)', raw):
            if self._jabra_card_name.lower() not in sec.lower():
                continue
            has_card = True
            m = re.search(r'Name:\s*(\S+)', sec)
            if m:
                card_id = m.group(1)
            m = re.search(r'Active Profile:\s*(\S+)', sec)
            if m:
                profile_ok = self._jabra_profile.lower() == m.group(1).lower()
            break

        if has_card:
            try:
                src = subprocess.check_output(
                    ['pactl', 'get-default-source'], timeout=3, text=True).strip()
                check = self._pa_source_check or self._jabra_card_name
                source_ok = check.lower() in src.lower()
            except Exception as e:
                self.get_logger().warn(f'pactl get-default-source: {e}')

        return has_card, card_id, profile_ok, source_ok

    def _check_and_maybe_restart_wireplumber(self):
        """Checks PipeWire and restarts WirePlumber if the profile/source are broken.
        Runs in a daemon thread."""
        if not self._jabra_card_name:
            return
        has_card, card_id, profile_ok, source_ok = self._pipewire_jabra_state()
        if has_card and profile_ok and source_ok:
            self._wp_restart_fail_streak = 0
            self.get_logger().info(
                'PipeWire is fine — the reason for the zeros: the Mute button on the Jabra')
            return
        reasons = []
        if not has_card:
            reasons.append('card not found')
        elif not profile_ok:
            reasons.append(f'wrong profile (expected {self._jabra_profile})')
        if not source_ok:
            reasons.append('source not active')
        self.get_logger().error(
            f'PipeWire problem ({", ".join(reasons)}) — restarting WirePlumber')
        self._restart_wireplumber()

    def _restart_wireplumber(self):
        """Restarts WirePlumber, restores the profile and PCM.
        Called from a daemon thread, protected by a cooldown."""
        with self._recovery_lock:
            now = time.time()
            if now - self._last_wp_restart < self._wp_restart_cooldown:
                remaining = self._wp_restart_cooldown - (now - self._last_wp_restart)
                self.get_logger().info(
                    f'WP cooldown: next restart in {remaining:.0f}s')
                return
            self._last_wp_restart = now

        self.get_logger().warn('Restarting WirePlumber...')
        try:
            subprocess.run(
                ['systemctl', '--user', 'restart', 'wireplumber'],
                timeout=20, check=True)
        except Exception as e:
            self.get_logger().error(f'Failed to restart WirePlumber: {e}')
            return

        # Wait for the Jabra source to appear (up to 10s)
        card_id = ''
        source_ok = False
        for _ in range(20):
            time.sleep(0.5)
            _, card_id, _, source_ok = self._pipewire_jabra_state()
            if source_ok:
                break
        else:
            self.get_logger().warn('Jabra source did not appear within 10s after the WP restart')

        if source_ok:
            self._wp_restart_fail_streak = 0
        else:
            self._wp_restart_fail_streak += 1
            if self._wp_restart_fail_streak >= self._PW_ESCALATE_AFTER:
                self._restart_pipewire_full()
                self._wp_restart_fail_streak = 0
                return

        # Fix the profile if needed
        _, card_id, profile_ok, _ = self._pipewire_jabra_state()
        if card_id and not profile_ok:
            try:
                subprocess.run(
                    ['pactl', 'set-card-profile', card_id, self._jabra_profile],
                    timeout=5, check=True)
                self.get_logger().info(f'Profile restored: {self._jabra_profile}')
            except Exception as e:
                self.get_logger().warn(f'Failed to set the profile: {e}')

        self._restore_pcm_and_stream()

    def _restore_pcm_and_stream(self):
        """PCM=100% (looking up the card by name to avoid hard-coding an index) +
        restart the audio stream. The common recovery tail after restarting
        WirePlumber or the whole PipeWire stack — without this the node thinks
        PipeWire is fine, but capture actually remains dead/muted."""
        try:
            aplay = subprocess.run(
                ['aplay', '-l'], capture_output=True, text=True, timeout=5)
            for line in aplay.stdout.splitlines():
                if 'jabra' in line.lower():
                    m = re.search(r'card (\d+):', line)
                    if m:
                        subprocess.run(
                            ['amixer', '-c', m.group(1), 'set', 'PCM', '100%'],
                            timeout=5)
                        self.get_logger().info(f'PCM card {m.group(1)} → 100%')
                        break
        except Exception as e:
            self.get_logger().warn(f'amixer PCM: {e}')

        self.get_logger().info('WirePlumber restored — restarting the audio stream')
        self._restart_stream()

    def _restart_pipewire_full(self):
        """Escalation: N consecutive WirePlumber restarts failed to bring back
        source_ok — likely it's not WirePlumber (the policy manager) that's
        hung, but pipewire-pulse itself (the PulseAudio protocol server) or
        pipewire (the graph core). Restarting just WP is then useless. Restart
        the whole user audio stack. Has its own cooldown protection —
        unrelated to _wp_restart_cooldown, since this is a much heavier
        operation (kills audio for the entire session, not just Jabra)."""
        now = time.time()
        if now - self._last_full_pw_restart < self._pw_full_restart_cooldown:
            remaining = self._pw_full_restart_cooldown - (now - self._last_full_pw_restart)
            self.get_logger().info(
                f'PW full-restart cooldown: next attempt in {remaining:.0f}s')
            return
        self._last_full_pw_restart = now

        self.get_logger().fatal(
            f'{self._PW_ESCALATE_AFTER} consecutive WirePlumber restart(s) did not help — '
            f'looks like pipewire-pulse/pipewire is hung, not WirePlumber itself. '
            f'Restarting the whole audio stack (pipewire, pipewire-pulse, wireplumber)...')
        try:
            subprocess.run(
                ['systemctl', '--user', 'restart',
                 'pipewire.service', 'pipewire-pulse.service', 'wireplumber.service'],
                timeout=30, check=True)
            self.get_logger().info('Audio stack restarted')
        except Exception as e:
            self.get_logger().error(f'Failed to restart the audio stack: {e}')

        self._restore_pcm_and_stream()


def main():
    rclpy.init()
    node = AudioSourceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
