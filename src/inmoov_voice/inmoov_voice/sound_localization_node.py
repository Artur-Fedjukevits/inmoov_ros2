#!/usr/bin/env python3

"""
sound_localization_node.py
===========================
Coarse direction-to-sound-source estimation from a pair of MAX9814
microphones on a CM6206 sound card (USB 0d8c:0102, ALSA name
"ICUSBAUDIO7D") — via majority voting on the SIGN of band-passed
GCC-PHAT (TDOA), NOT via inter-channel level difference (ILD) and NOT
via a physical angle model.

History:
1) The original broadband GCC-PHAT worked cleanly on an open table
   (±90°, hiss noise), but once mounted in the head's ears it produced
   physically impossible delays: the open skull skeleton (servos,
   wiring, neck) lets sound leak directly between the capsules, and
   this internal leak dominates over the direct path through the air
   outside.
2) Pivoted to ILD (level difference) — worked at 5-6cm, but at real
   conversational distance (1-3m, ~3x2.5m room) degraded to almost
   random sign: wall reflections swamp the weak level difference from
   direction (verified statistically 2026-08-22 — several consecutive
   measurements gave the wrong sign at 1m).
3) 2026-08-22: a band-pass filter (2-6kHz — where the cavity leak is
   weaker than at bass frequencies) before GCC-PHAT did NOT remove the
   excess delay (median values are still several times over the
   physical limit for the inter-ear baseline), BUT its SIGN correlates
   reliably with the real direction at every distance tested (0.2-2m,
   both hiss and live speech). Majority voting on the sign over a
   sliding window of blocks was confirmed blind: 6/6 correct guesses in
   a live test (different sides, distances, 45°, center).

Microphones are read DIRECTLY via raw ALSA, bypassing PipeWire (the
card is excluded from WirePlumber by a device.disabled rule in
~/.config/wireplumber/main.lua.d/52-cm6206-disable.lua — without this,
PipeWire holds a D-Bus reservation on the device and raw access is
impossible).

IMPORTANT — the card's gain does NOT survive a reboot/reconnect (ALSA's
alsa-restore fires before this USB card initializes): after every host
reboot check
`amixer -c ICUSBAUDIO7D contents | grep -A3 "Mic Capture Volume"` —
it should read **60% (4157/6928, +0.23dB)**, not the maximum
(6928/6928). To restore: `amixer -c ICUSBAUDIO7D sset Mic 60% cap`. If
the udev rule 99-cm6206-gain.rules is in place (see memory) this should
happen automatically, but it's still worth checking.
(Tried asymmetric per-channel gain to compensate for the different
acoustic coupling of the capsules to outside sound — didn't work: too
sensitive to the exact value, and any imbalance made one channel lose
sensitivity to its own direction entirely. Symmetric gain is simpler
and more predictable; the exact value (60%) matters less for the TDOA
algorithm than it did for ILD — it's still a dB figure, but symmetry
matters more.)

Algorithm per block (~85ms @ 48kHz):
  1. Block RMS (both channels) → energy gate (rms_gate_dbfs), as before.
  2. Band-pass filter 2-6kHz (bandpass_low/high, Butterworth,
     sosfiltfilt — zero phase delay, important for TDOA accuracy) on
     both channels.
  3. GCC-PHAT: Hann window → FFT → cross-spectrum XL·conj(XR) → PHAT
     normalization (divide by magnitude, keep only phase) → IFFT →
     peak within a wide search window (±5ms, NOT limited to the
     physical bound — the peak itself will still land outside the
     physical range, but its SIGN is informative).
  4. The sign of the raw delay (µs) is appended to a sliding window
     (vote_window_sec, default 3s) — the delays themselves are NOT
     averaged (the magnitude is physically meaningless and erratic),
     only a majority vote on the sign is taken.
  5. angle_deg = (right_votes − left_votes) / total_votes · 90°,
     confidence = |same ratio|. NOT a physical model.

IMPORTANT — calibration before use on the head:
The sign depends on which physical capsule is wired into which ALSA
channel (0=right/FL, 1=left/FR — yes, "flipped" relative to intuition,
see memory). Verify by hand (speak/hiss from a known side, watch the
sign of angle_deg) and set swap_channels:=false if needed (the default
true was chosen 2026-08-22, the same mapping as the ILD version — if
the card/wiring hasn't been touched since that test, no need to
change it).

Practical takeaway for topic consumers: rely on `angle_deg`/`confidence`
(already aggregated over the window), NOT on `tdoa_us` (that's the
window median, purely for debugging, physically unrealistic). Low
confidence (<0.5) means the window hasn't filled yet or the sign is
flip-flopping within the window — worth waiting for a bit more speech
before acting on a turn decision.

Known limitations:
  - Front/back ambiguity (common to any microphone pair) — cannot tell
    a source in front from one behind at the same angle. Resolve via
    vision (OAK-D/face_detection) — coarse bearing from sound, exact
    side and front/back from the camera.
  - Speech is noticeably noisier than hiss beyond 1m (vowels/periodicity
    are worse for PHAT than broadband noise) — the voting window
    smooths this out, but short utterances (<1-2s) may not accumulate a
    confident result in time.
  - Votes are time-stamped: votes older than vote_window_sec expire, and a
    silence longer than silence_reset_sec clears the window, so a new
    speaker after a pause starts from a clean vote (the old direction does
    not "stick"). The price: right after a pause the vote rests on the few
    blocks of the new utterance only.
  - No on_set_parameters_callback — `ros2 param set` at runtime has no
    effect, only a restart with -p.

LifecycleNode (integrated into inmoov_bringup, tier 1 — Hardware
Drivers, alongside audio_source_node/oak_node/face_capture_node, which
also have direct hardware access). Parameters are read in on_configure,
the stream is opened and the worker thread started in on_activate.

Topic:
  /sound_direction  (inmoov_msgs/SoundDirection)

Standalone test run (no launch file, manual lifecycle transitions):
  ros2 run inmoov_voice sound_localization_node
  ros2 lifecycle set /sound_localization_node configure
  ros2 lifecycle set /sound_localization_node activate
  ros2 topic echo /sound_direction

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import math
import os
import queue
import statistics
import threading
import time
from collections import deque

# Limit BLAS/numpy internal multithreading BEFORE importing numpy/scipy —
# on this NUC, heavy nodes run in parallel (face_detection ~60%+ CPU per
# eye); extra BLAS threads only add core contention and undermine the
# very reason a separate queue exists here (see below).
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn  # noqa: E402
from std_msgs.msg import Header  # noqa: E402
import sounddevice as sd  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

from inmoov_msgs.msg import SoundDirection  # noqa: E402


class SoundLocalizationNode(LifecycleNode):

    def __init__(self):
        super().__init__('sound_localization_node')

        self._stream = None
        self._last_block_time = 0.0
        self._error_streak = 0
        self._timers = []
        self._queue = None
        self._stop_event = None
        self._worker = None

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores re-declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_configure(self, state):
        self._dp('device_name', 'ICUSBAUDIO7D')
        self._dp('sample_rate', 48000)
        self._dp('block_size', 4096)       # ~85 ms @ 48 kHz
        self._dp('bandpass_low_hz', 2000.0)
        self._dp('bandpass_high_hz', 6000.0)
        self._dp('mic_distance_m', 0.145)   # for reference / window sizing only, NOT used in angle_deg
        self._dp('search_window_sec', 0.005)  # ±5 ms — deliberately wider than the physical limit so the peak itself isn't cut off
        self._dp('vote_window_sec', 3.0)    # sliding sign-vote window; larger = more reliable but slower to react
        self._dp('rms_gate_dbfs', -24.0)    # tuned for 60% gain and a real distance of 1-3 m (see the note in the module docstring)
        self._dp('publish_silence', False)
        self._dp('silence_reset_sec', 2.0)  # silence longer than this clears the vote window
        self._dp('swap_channels', True)     # raw ch0 = physically right, ch1 = physically left — see the module docstring
        self._dp('watchdog_sec', 3.0)

        self._device_name       = self.get_parameter('device_name').value
        self.rate                = self.get_parameter('sample_rate').value
        self.block_size          = self.get_parameter('block_size').value
        self._bandpass_low       = self.get_parameter('bandpass_low_hz').value
        self._bandpass_high      = self.get_parameter('bandpass_high_hz').value
        self._mic_distance_m     = self.get_parameter('mic_distance_m').value
        self._search_window_sec  = self.get_parameter('search_window_sec').value
        self._vote_window_sec    = self.get_parameter('vote_window_sec').value
        self._rms_gate_dbfs      = self.get_parameter('rms_gate_dbfs').value
        self._publish_silence    = self.get_parameter('publish_silence').value
        self._silence_reset_sec  = self.get_parameter('silence_reset_sec').value
        self._swap_channels      = self.get_parameter('swap_channels').value
        self._watchdog_sec       = self.get_parameter('watchdog_sec').value

        self._vote_window_blocks = max(1, int(self._vote_window_sec * self.rate / self.block_size))
        self._vote_window = deque(maxlen=self._vote_window_blocks)   # (monotonic t, tdoa_us)
        self._last_voiced_t = 0.0

        self._sos = butter(4, [self._bandpass_low, self._bandpass_high],
                           btype='band', fs=self.rate, output='sos')

        self._n_fft = 1
        while self._n_fft < 2 * self.block_size:
            self._n_fft *= 2
        self._hann = np.hanning(self.block_size)

        self._pub = self.create_lifecycle_publisher(SoundDirection, 'sound_direction', 10)

        self.get_logger().info(
            f'SoundLocalization configured (TDOA sign-vote): rate={self.rate} block={self.block_size} '
            f'({1000 * self.block_size / self.rate:.0f}ms) '
            f'band={self._bandpass_low:.0f}-{self._bandpass_high:.0f}Hz '
            f'vote_window={self._vote_window_sec}s ({self._vote_window_blocks} blocks) '
            f'rms_gate={self._rms_gate_dbfs}dBFS')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._pub.on_activate(state)

        self._vote_window.clear()
        self._queue = queue.Queue(maxsize=8)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        if not self._open_stream():
            self._stop_worker()
            self._pub.on_deactivate(state)
            return TransitionCallbackReturn.FAILURE

        self._timers = [self.create_timer(self._watchdog_sec, self._watchdog)]
        self.get_logger().info('SoundLocalization active')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        for t in self._timers:
            self.destroy_timer(t)
        self._timers = []
        self._close_stream()
        self._stop_worker()
        self._pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self._close_stream()
        self._stop_worker()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self._close_stream()
        self._stop_worker()
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        self._close_stream()
        self._stop_worker()
        return TransitionCallbackReturn.SUCCESS

    def _stop_worker(self):
        if self._stop_event is not None:
            self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        self._worker = None

    # ── Opening the device ─────────────────────────────────────────────────

    def _find_device_index(self):
        for i, d in enumerate(sd.query_devices()):
            if self._device_name.lower() in d['name'].lower() and d['max_input_channels'] >= 2:
                return i
        return None

    def _open_stream(self) -> bool:
        self._close_stream()

        idx = self._find_device_index()
        if idx is None:
            self.get_logger().error(
                f'Device "{self._device_name}" with 2 input channels not found. '
                f'Check: is the card excluded from WirePlumber? (wpctl status must not '
                f'list it); is it still at hw:CARD={self._device_name}?')
            return False

        try:
            self.get_logger().info(f'Opening [{idx}] {sd.query_devices()[idx]["name"]}')
            self._stream = sd.InputStream(
                device=idx,
                channels=2,
                samplerate=self.rate,
                blocksize=self.block_size,
                dtype='float32',
                latency=0.2,  # seconds, as an EXPLICIT number — the string 'high' maps to only ~35 ms with this driver (too little!) and didn't help; 0.2 s is empirically clean with no overflow (2026-08-22: a 0.1 s test still hit overflow, 0.15 s+ was clean)
                callback=self._on_audio_block,
            )
            self._stream.start()
            self._last_block_time = self.get_clock().now().nanoseconds / 1e9
            return True
        except Exception as e:
            self.get_logger().error(f'Failed to open the audio stream: {e}')
            self._stream = None
            return False

    def _close_stream(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _watchdog(self):
        now = self.get_clock().now().nanoseconds / 1e9
        if self._stream is None or (now - self._last_block_time) > self._watchdog_sec:
            self.get_logger().warn('Watchdog: no audio — reopening the stream')
            self._open_stream()

    # ── Audio stream callback (PortAudio realtime thread, NOT the ROS executor) ──
    # Must be as fast as possible — only copy the block into the queue; all
    # heavy processing (band-pass filter, FFT/PHAT) lives in _worker_loop().

    def _on_audio_block(self, indata, frames, time_info, status):
        self._last_block_time = self.get_clock().now().nanoseconds / 1e9
        if status:
            self.get_logger().warn(f'Audio status: {status}')
        try:
            self._queue.put_nowait(indata.copy())
        except queue.Full:
            # Processing can't keep up — drop the oldest block and enqueue the
            # new one so the queue doesn't accumulate latency indefinitely.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(indata.copy())
            except (queue.Empty, queue.Full):
                pass

    # ── Worker thread: heavy processing outside the realtime callback ───────

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                block = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                left, right = block[:, 0], block[:, 1]
                if self._swap_channels:
                    left, right = right, left
                self._process_block(left, right)
                self._error_streak = 0
            except Exception as e:
                self._error_streak += 1
                if self._error_streak <= 3 or self._error_streak % 100 == 0:
                    self.get_logger().error(f'Block processing error: {e}')

    def _process_block(self, left: np.ndarray, right: np.ndarray):
        rms_l = float(np.sqrt(np.mean(left.astype(np.float64) ** 2)))
        rms_r = float(np.sqrt(np.mean(right.astype(np.float64) ** 2)))
        rms_avg = math.sqrt((rms_l ** 2 + rms_r ** 2) / 2.0)
        rms_dbfs = 20.0 * math.log10(max(rms_avg, 1e-9))
        voiced = rms_dbfs >= self._rms_gate_dbfs

        if not voiced and not self._publish_silence:
            return

        if voiced:
            now = time.monotonic()
            # A pause longer than silence_reset_sec starts a new vote (new
            # speaker / new utterance); older votes also expire by age.
            if now - self._last_voiced_t > self._silence_reset_sec:
                self._vote_window.clear()
            self._last_voiced_t = now
            while self._vote_window and now - self._vote_window[0][0] > self._vote_window_sec:
                self._vote_window.popleft()

            tdoa_us = self._gcc_phat_tdoa_us(left, right)
            self._vote_window.append((now, tdoa_us))

            delays = [d for _, d in self._vote_window]
            total = len(delays)
            pos = sum(1 for x in delays if x > 0)
            neg = sum(1 for x in delays if x < 0)
            score = (pos - neg) / total if total > 0 else 0.0
            angle_deg = score * 90.0
            confidence = abs(score)
            tdoa_us_median = statistics.median(delays)
        else:
            tdoa_us_median, angle_deg, confidence = 0.0, 0.0, 0.0

        msg = SoundDirection()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'head'
        msg.angle_deg = float(angle_deg)
        msg.tdoa_us = float(tdoa_us_median)
        msg.confidence = float(confidence)
        msg.rms_dbfs = float(rms_dbfs)
        msg.voiced = bool(voiced)
        self._pub.publish(msg)

    def _gcc_phat_tdoa_us(self, left: np.ndarray, right: np.ndarray) -> float:
        """Band-passed GCC-PHAT; returns the signed delay in µs (raw, physically
        unrealistic in magnitude because of sound leaking through the skull
        cavity — use only the SIGN, see the module docstring)."""
        fl = sosfiltfilt(self._sos, left.astype(np.float64))
        fr = sosfiltfilt(self._sos, right.astype(np.float64))

        fl_w = fl * self._hann
        fr_w = fr * self._hann

        XL = np.fft.rfft(fl_w, n=self._n_fft)
        XR = np.fft.rfft(fr_w, n=self._n_fft)
        R = XL * np.conj(XR)
        R_phat = R / (np.abs(R) + 1e-12)
        r = np.fft.fftshift(np.fft.irfft(R_phat, n=self._n_fft))
        center = self._n_fft // 2

        wide = max(1, int(self._search_window_sec * self.rate))
        lo, hi = center - wide, center + wide
        lag = int(np.argmax(r[lo:hi])) + lo - center
        return lag / self.rate * 1e6


def main():
    rclpy.init()
    node = SoundLocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
