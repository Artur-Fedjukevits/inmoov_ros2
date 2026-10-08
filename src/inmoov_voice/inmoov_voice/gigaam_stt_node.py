#!/usr/bin/env python3
"""
gigaam_stt_node.py
==================
STT: GigaAM-v3 e2e-RNNT (Russian-only, punctuation + "ё", ONNX int8, CPU) as
the primary model, Parakeet-TDT-0.6B-v3 as the fallback for English speech.

Why (bench ~/gigaam_ml_test/bench, 28 test clips + 164 live phrases 2026-10-07):
  - GigaAM-v3 is more accurate on Russian and ~2x faster than Parakeet;
    e2e-rnnt drops far fewer words than plain rnnt on far-field speech and
    keeps punctuation;
  - Parakeet turns Russian speech into Italian/English/Ukrainian word salad or
    "Субтитры сделал DimaTorzok" surprisingly often, but is the only model
    with English (Norwegian comes out as Swedish-ish text).

Fallback to Parakeet when the GigaAM text is empty / too short for the audio
(fewer than fallback_min_cps chars per second) or mostly Latin letters — on
English speech e2e-rnnt outputs Latin gibberish rather than nothing.
Both models often miss the unusual "Лёня" — the robot also answers to
"Леонид" (llm_node._ROBOT_NAMES) and to the wake word "Эй, Лёня".

compare_parakeet=True runs Parakeet on every phrase too and logs both
transcripts (+ JSONL + WAV) for a side-by-side comparison.

Subscribes:
  audio_to_whisper  (Float32MultiArray) — post-VAD audio segment, 16kHz float32

Publishes:
  <output_topic>        (String) — main interlocutor (empty string = not recognized)
  <other_output_topic>  (String) — phrases marked 'other_speaker'; '' disables

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import collections
import datetime
import json
import os
import re
import threading
import time
import wave

import numpy as np
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from std_msgs.msg import Float32MultiArray, String

def _latin_share(text: str) -> float:
    letters = re.findall(r'[a-zа-яёі]', text.lower())
    return sum('a' <= c <= 'z' for c in letters) / len(letters) if letters else 0.0


class GigaAMSTTNode(LifecycleNode):
    def __init__(self):
        super().__init__('gigaam_stt_node')
        self.subscription  = None
        self.text_pub      = None
        self.other_pub     = None
        self._model        = None
        self._fallback     = None
        self._transcribing = False
        self._lock         = threading.Lock()

    # ── Lifecycle: Phase 2 ─────────────────────────────────────────────────

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores re-declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('model_name',          'gigaam-v3-e2e-rnnt')
        # int8 = fp32 quality on the bench, 3x less RAM
        self._dp('quantization',        'int8')
        self._dp('fallback_model_name', 'nemo-parakeet-tdt-0.6b-v3')
        self._dp('fallback_quantization', '')   # see parakeet_stt_node: int8 loses quiet speech
        # Same as parakeet_stt_node; Parakeet-v3 still outputs English with 'ru'
        self._dp('fallback_language',   'ru')
        # Fallback when GigaAM gives fewer chars per second of audio: Russian
        # speech is >= 6 c/s on the bench, English through GigaAM 0-0.7 c/s
        self._dp('fallback_min_cps',    2.0)
        # ...or when more than this share of the letters is Latin: e2e-rnnt
        # writes English as Latin gibberish; 0 of 95 live Russian phrases > 0.5
        self._dp('fallback_max_latin',  0.5)
        self._dp('min_audio_sec',       0.8)
        self._dp('output_topic',        'voice_command')
        # Phrases voice_detector marked 'other_speaker' — llm_node answers them
        # only when called by name
        self._dp('other_output_topic',  'voice_command_other')
        # Comparison: also run Parakeet on every phrase and log both
        self._dp('compare_parakeet',    False)
        self._dp('compare_log',         '')   # e.g. ~/inmoov_sv_debug/stt_shadow/stt_shadow.jsonl
        self._dp('save_audio_dir',      '')   # WAV per phrase for the comparison

        self.model_name     = self.get_parameter('model_name').value
        self.quantization   = self.get_parameter('quantization').value
        fb_name             = self.get_parameter('fallback_model_name').value
        fb_quant            = self.get_parameter('fallback_quantization').value
        self.fb_language    = self.get_parameter('fallback_language').value
        self.min_cps        = self.get_parameter('fallback_min_cps').value
        self.max_latin      = self.get_parameter('fallback_max_latin').value
        self.min_audio_sec  = self.get_parameter('min_audio_sec').value
        output_topic        = self.get_parameter('output_topic').value
        other_topic         = self.get_parameter('other_output_topic').value
        self.compare        = self.get_parameter('compare_parakeet').value
        log_path            = self.get_parameter('compare_log').value
        audio_dir           = self.get_parameter('save_audio_dir').value
        self.compare_log    = os.path.expanduser(log_path) if log_path else ''
        self.audio_dir      = os.path.expanduser(audio_dir) if audio_dir else ''
        for path in (os.path.dirname(self.compare_log), self.audio_dir):
            if path:
                os.makedirs(path, exist_ok=True)

        import onnx_asr
        t0 = time.perf_counter()
        self._model = onnx_asr.load_model(self.model_name,
                                          quantization=self.quantization or None)
        t1 = time.perf_counter()
        self._fallback = onnx_asr.load_model(fb_name, quantization=fb_quant or None)
        self.get_logger().info(
            f'GigaAM ({self.model_name}, {self.quantization or "fp32"}) ready in '
            f'{t1 - t0:.1f}s, fallback ({fb_name}) in {time.perf_counter() - t1:.1f}s'
            f'{" — shadow comparison ON" if self.compare else ""}')

        self.subscription = self.create_subscription(
            Float32MultiArray, 'audio_to_whisper', self.audio_callback, 10)
        self.text_pub = self.create_lifecycle_publisher(String, output_topic, 10)
        self.other_pub = (self.create_lifecycle_publisher(String, other_topic, 10)
                          if other_topic else None)
        self._transcribing = False
        self._queue = collections.deque(maxlen=4)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.text_pub.on_activate(state)
        if self.other_pub:
            self.other_pub.on_activate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.text_pub.on_deactivate(state)
        if self.other_pub:
            self.other_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self._model = None
        self._fallback = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

    # ── Receiving audio ───────────────────────────────────────────────────────
    def audio_callback(self, msg: Float32MultiArray):
        with self._lock:
            if self._transcribing:
                if len(self._queue) == self._queue.maxlen:
                    self.get_logger().warn('GigaAM: queue full — oldest waiting phrase dropped.')
                self._queue.append(msg)
                return
            self._transcribing = True

        thread = threading.Thread(
            target=self._transcribe, args=(msg,), daemon=True)
        thread.start()

    # ── Transcription ──────────────────────────────────────────────────────────
    def _run_fallback(self, audio, sample_rate):
        t0 = time.perf_counter()
        text = self._fallback.recognize(audio, sample_rate=sample_rate,
                                        language=self.fb_language, pnc=True)
        return (text or '').strip(), time.perf_counter() - t0

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
                self._publish('', other)
                return

            t_start = time.perf_counter()
            giga = (self._model.recognize(audio, sample_rate=sample_rate) or '').strip()
            t_giga = time.perf_counter() - t_start

            use_fallback = (len(giga) / duration < self.min_cps
                            or _latin_share(giga) > self.max_latin)
            para, t_para = None, 0.0
            if use_fallback or self.compare:
                para, t_para = self._run_fallback(audio, sample_rate)
            text, engine = (para, 'parakeet') if use_fallback else (giga, 'gigaam')
            elapsed = time.perf_counter() - t_start

            tag = ' [other speaker]' if other else ''
            if self.compare:
                self.get_logger().info(
                    f'STT audio {duration:.1f}s{tag} in {elapsed:.2f}s → {engine.upper()}\n'
                    f'    gigaam   {t_giga:.2f}s: "{giga}"\n'
                    f'    parakeet {t_para:.2f}s: "{para}"')
            else:
                self.get_logger().info(
                    f'{engine} recognized in {elapsed:.2f}s (audio {duration:.1f}s)'
                    f'{tag}: "{text}"')
            self._record(audio, sample_rate, duration, other, giga, t_giga,
                         para, t_para, engine, text)
            self._publish(text, other)

        except Exception as e:
            self.get_logger().error(f'GigaAM transcription error: {e}')
            self._publish('', other)
        finally:
            with self._lock:
                nxt = self._queue.popleft() if self._queue else None
                self._transcribing = nxt is not None
        if nxt is not None:
            self._transcribe(nxt)

    def _record(self, audio, sample_rate, duration, other, giga, t_giga,
                para, t_para, engine, text):
        if not (self.compare_log or self.audio_dir):
            return
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        wav_name = ''
        try:
            if self.audio_dir:
                wav_name = f'{stamp}.wav'
                pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                with wave.open(os.path.join(self.audio_dir, wav_name), 'wb') as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(sample_rate)
                    w.writeframes(pcm.tobytes())
            if self.compare_log:
                with open(self.compare_log, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(dict(
                        t=stamp, wav=wav_name, dur=round(duration, 2), other=other,
                        engine=engine, text=text, model=self.model_name,
                        gigaam=giga, t_gigaam=round(t_giga, 3),
                        parakeet=para, t_parakeet=round(t_para, 3)),
                        ensure_ascii=False) + '\n')
        except OSError as e:
            self.get_logger().warn(f'GigaAM: comparison record failed: {e}')

    def _publish(self, text: str, other: bool = False):
        if other:
            if text and self.other_pub:
                self.other_pub.publish(String(data=text))
            return
        self.text_pub.publish(String(data=text))


def main():
    rclpy.init()
    node = GigaAMSTTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
