#!/usr/bin/env python3
"""
voice_emotion_node.py
=====================
Detects emotion from voice using SpeechBrain wav2vec2-IEMOCAP.

Subscribes:
  audio_to_whisper  (Float32MultiArray) — post-VAD audio segment, 16kHz float32
  /robot_sleep      (Bool, latched)

Publishes:
  /voice/emotion  (String JSON)
  {
    "emotion":    "sad",
    "confidence": 0.87,
    "all":        {"neutral": 0.12, "angry": 0.01, "happy": 0.00, "sad": 0.87}
  }

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import json
import os
import threading

import numpy as np
import torch

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32MultiArray, String

_SB_TO_EMOTION = {
    'neu': 'neutral',
    'ang': 'angry',
    'hap': 'happy',
    'sad': 'sad',
}

_MIN_SAMPLES = 16000  # 1.0 s @ 16kHz — wav2vec2 needs headroom for its context window


class VoiceEmotionNode(LifecycleNode):
    def __init__(self):
        super().__init__('voice_emotion_node')
        self._pub      = None
        self._model    = None
        self._labels   = None
        self._sleeping = False
        self._busy     = False

    # ── Lifecycle: Phase 2 ─────────────────────────────────────────────────

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores re-declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('min_confidence', 0.55)
        self._dp('savedir', os.path.expanduser('~/.cache/speechbrain/voice_emotion'))

        self._min_conf = self.get_parameter('min_confidence').value
        savedir        = self.get_parameter('savedir').value

        self.get_logger().info('Loading SpeechBrain emotion model (wav2vec2-IEMOCAP)...')
        from speechbrain.inference.classifiers import EncoderClassifier
        self._model = EncoderClassifier.from_hparams(
            'speechbrain/emotion-recognition-wav2vec2-IEMOCAP',
            savedir=savedir,
        )
        self._labels = self._model.hparams.label_encoder.ind2lab
        self.get_logger().info(
            f'Voice-emotion model ready. Classes: {list(self._labels.values())}')

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Bool, '/robot_sleep', self._sleep_cb, latched_qos)
        self.create_subscription(Float32MultiArray, 'audio_to_whisper', self._audio_cb, 10)
        self._pub = self.create_lifecycle_publisher(String, '/voice/emotion', 10)
        self.get_logger().info(f'VoiceEmotion configured (min_confidence={self._min_conf})')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._pub.on_activate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self._pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

    def _sleep_cb(self, msg: Bool):
        self._sleeping = msg.data

    def _audio_cb(self, msg: Float32MultiArray):
        if self._sleeping or self._busy:
            return
        if any(d.label == 'other_speaker' for d in msg.layout.dim):
            return   # not the interlocutor — their emotion must not drive the face

        audio = np.array(msg.data, dtype=np.float32)
        if len(audio) < _MIN_SAMPLES:
            self.get_logger().debug(
                f'VoiceEmotion: audio {len(audio)/16000:.2f}s < 1.0s — skipping'
            )
            return

        self._busy = True
        threading.Thread(target=self._analyze, args=(audio,), daemon=True).start()

    def _analyze(self, audio: np.ndarray):
        try:
            with torch.no_grad():
                sig    = torch.from_numpy(audio).unsqueeze(0)            # [1, N]
                feats  = self._model.mods.wav2vec2(sig)                  # [1, T, 768]
                pooled = self._model.mods.avg_pool(feats).squeeze(1)     # [1, 768]
                logits = self._model.mods.output_mlp(pooled)             # [1, 4]
                probs  = self._model.hparams.softmax(logits)[0]          # [4]

            best_idx = int(probs.argmax())
            best_sb  = self._labels[best_idx]
            emotion  = _SB_TO_EMOTION.get(best_sb, 'neutral')
            conf     = float(probs[best_idx])

            all_scores = {
                _SB_TO_EMOTION.get(self._labels[i], 'neutral'): round(float(probs[i]), 3)
                for i in self._labels
            }

            dur = len(audio) / 16000.0
            self.get_logger().info(
                f'Voice emotion: {emotion} (conf={conf:.2f}, dur={dur:.1f}s)'
            )

            msg = String()
            msg.data = json.dumps({
                'emotion':    emotion,
                'confidence': round(conf, 3),
                'all':        all_scores,
            })
            self._pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f'VoiceEmotion error: {e}')
        finally:
            self._busy = False


def main():
    rclpy.init()
    node = VoiceEmotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
