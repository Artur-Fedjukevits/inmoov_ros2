#!/usr/bin/env python3
"""
parakeet_stt_node.py
=====================
Primary STT on NVIDIA Parakeet-TDT-0.6B-v3 (ONNX, int8, CPU).
Replaced whisper.cpp (removed from the system 2026-08-27) — 2-4x faster in
live use (RTF 0.07-0.23 vs. 0.4-0.6 for whisper large-v3-turbo/Vulkan iGPU).
The model (onnx-asr) loads directly into
the process — no separate HTTP server needed.

Subscribes:
  audio_to_whisper  (Float32MultiArray) — post-VAD audio segment, 16kHz float32

Publishes:
  voice_command  (String) — the same topic llm_node listens to.
  voice_command_other (String) — phrases marked 'other_speaker' by voice_detector
                                 (not the current interlocutor); empty results dropped

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import collections
import threading
import time

import numpy as np
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from std_msgs.msg import Float32MultiArray, String


class ParakeetSTTNode(LifecycleNode):
    def __init__(self):
        super().__init__('parakeet_stt_node')
        self.subscription  = None
        self.text_pub      = None
        self._model        = None
        self._transcribing = False
        self._lock         = threading.Lock()

    # ── Lifecycle: Phase 2 ─────────────────────────────────────────────────

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores re-declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('model_name',    'nemo-parakeet-tdt-0.6b-v3')
        # '' = full-precision model. int8 returned empty text for quiet / far-field
        # Russian speech that the full model transcribes correctly (live 2026-09-26),
        # for only +10-40 ms per phrase.
        self._dp('quantization',  '')
        self._dp('language',      'ru')
        self._dp('pnc',           True)   # punctuation & capitalization
        self._dp('min_audio_sec', 0.8)
        self._dp('output_topic',  'voice_command')
        # Phrases voice_detector marked 'other_speaker' (SV: not the current
        # interlocutor) — llm_node answers them only when called by name
        self._dp('other_output_topic', 'voice_command_other')

        self.model_name    = self.get_parameter('model_name').value
        self.quantization  = self.get_parameter('quantization').value
        self.language      = self.get_parameter('language').value
        self.pnc           = self.get_parameter('pnc').value
        self.min_audio_sec = self.get_parameter('min_audio_sec').value
        output_topic       = self.get_parameter('output_topic').value
        other_topic        = self.get_parameter('other_output_topic').value

        self.get_logger().info(
            f'Loading Parakeet ({self.model_name}, quant={self.quantization})...')
        t0 = time.perf_counter()
        import onnx_asr
        self._model = onnx_asr.load_model(self.model_name,
                                          quantization=self.quantization or None)
        self.get_logger().info(
            f'Parakeet model ready in {time.perf_counter() - t0:.1f}s')

        self.subscription = self.create_subscription(
            Float32MultiArray, 'audio_to_whisper', self.audio_callback, 10)
        self.text_pub = self.create_lifecycle_publisher(String, output_topic, 10)
        self.other_pub = self.create_lifecycle_publisher(String, other_topic, 10)
        self._transcribing = False
        self._queue = collections.deque(maxlen=4)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.text_pub.on_activate(state)
        self.other_pub.on_activate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.text_pub.on_deactivate(state)
        self.other_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self._model = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

    # ── Receiving audio ───────────────────────────────────────────────────────
    def audio_callback(self, msg: Float32MultiArray):
        with self._lock:
            if self._transcribing:
                # A queue instead of a drop: another speaker's phrase may still be
                # transcribing when the interlocutor's phrase arrives — dropping it
                # would leave voice_detector waiting for STT.
                if len(self._queue) == self._queue.maxlen:
                    self.get_logger().warn('Parakeet: queue full — oldest waiting phrase dropped.')
                self._queue.append(msg)
                self.get_logger().info(f'Parakeet: busy — phrase queued ({len(self._queue)} waiting).')
                return
            self._transcribing = True

        thread = threading.Thread(
            target=self._transcribe, args=(msg,), daemon=True)
        thread.start()

    # ── Transcription ──────────────────────────────────────────────────────────
    def _transcribe(self, msg: Float32MultiArray):
        other = False
        try:
            audio = np.array(msg.data, dtype=np.float32)
            other = any(d.label == 'other_speaker' for d in msg.layout.dim)

            sample_rate = 16000
            if msg.layout.dim and msg.layout.dim[0].label == 'sample_rate':
                sample_rate = msg.layout.dim[0].stride

            duration = len(audio) / sample_rate

            if duration < self.min_audio_sec:
                self.get_logger().warn(
                    f'Parakeet: audio too short ({duration:.2f}s < '
                    f'{self.min_audio_sec}s) — skipping.')
                self._publish('', other)
                return

            t0 = time.perf_counter()
            text = self._model.recognize(
                audio, sample_rate=sample_rate,
                language=self.language, pnc=self.pnc,
            )
            elapsed = time.perf_counter() - t0

            text = (text or '').strip()
            if not text:
                self.get_logger().warn(
                    f'Parakeet: speech not recognized (audio {duration:.1f}s).')
                self._publish('', other)
                return

            self.get_logger().info(
                f'Parakeet recognized in {elapsed:.2f}s (audio {duration:.1f}s, '
                f'RTF={elapsed / duration:.2f}){" [other speaker]" if other else ""}: "{text}"')
            self._publish(text, other)

        except Exception as e:
            self.get_logger().error(f'Parakeet transcription error: {e}')
            self._publish('', other)
        finally:
            with self._lock:
                nxt = self._queue.popleft() if self._queue else None
                self._transcribing = nxt is not None
        if nxt is not None:
            self._transcribe(nxt)

    def _publish(self, text: str, other: bool = False):
        if other:
            # Nobody waits for an empty result here (voice_detector's STT-done
            # handshake is only for the main interlocutor)
            if text:
                self.other_pub.publish(String(data=text))
            return
        msg = String()
        msg.data = text
        self.text_pub.publish(msg)


def main():
    rclpy.init()
    node = ParakeetSTTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
