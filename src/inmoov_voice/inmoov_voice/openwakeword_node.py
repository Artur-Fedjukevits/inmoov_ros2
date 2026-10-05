#!/usr/bin/env python3

"""
openwakeword_node.py
=====================
Wake word detection ("Эй Лёня") on a custom openWakeWord (ONNX) model.

Subscribes:
  raw_audio  (Float32MultiArray) — 16kHz float32 audio chunks from audio_source_node

Publishes:
  wake_detected (Bool)    — True on activation (debounced)
  wake_score    (Float32) — raw model score for every 80 ms frame, for tuning/diagnostics

Parameters:
  model_path    (str)   — path to the custom .onnx wake word model
  threshold     (float) — activation score threshold (default 0.9)
  patience      (int)   — consecutive 80 ms frames >= threshold required to activate (default 2)
  debounce_sec  (float) — minimum interval between two activations (default 1.5)
  save_dir      (str)   — if set, the last save_sec of audio before every activation is
                          written there as WAV (collects false activations for retraining)
  save_sec      (float) — how much audio before an activation to save (default 3.0)

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import collections
import os
import time
import wave

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from std_msgs.msg import Bool, Float32, Float32MultiArray
import numpy as np
from openwakeword.model import Model

FRAME_SAMPLES = 1280   # openWakeWord computes one score per 80 ms at 16 kHz


class WakeWordNode(LifecycleNode):
    def __init__(self):
        super().__init__('wakeword_node')
        self.model            = None
        self.model_key        = None
        self.threshold        = 0.9
        self.patience         = 2
        self.debounce_sec     = 1.5
        self._frames_above    = 0     # consecutive frames with score >= threshold
        self._buf             = np.zeros(0, dtype=np.int16)
        self.save_dir         = ''
        self._history         = collections.deque()   # recent frames, for save_dir
        self.last_activation  = 0.0
        self.activation_count = 0
        self.wake_pub         = None
        self.score_pub        = None
        self._active          = False   # set in on_activate — inference only when active

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores re-declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp(
            'model_path',
            os.path.expanduser('~/openWakeWord/my_custom_model/ey_lyonya.onnx')
        )
        self._dp('threshold',    0.9)
        self._dp('patience',     2)
        self._dp('debounce_sec', 1.5)
        self._dp('save_dir',     '')
        self._dp('save_sec',     3.0)

        model_path        = self.get_parameter('model_path').value
        self.threshold    = self.get_parameter('threshold').value
        self.patience     = max(1, int(self.get_parameter('patience').value))
        self.debounce_sec = self.get_parameter('debounce_sec').value
        self.save_dir     = os.path.expanduser(self.get_parameter('save_dir').value)
        n_hist = int(np.ceil(self.get_parameter('save_sec').value * 16000 / FRAME_SAMPLES))
        self._history = collections.deque(maxlen=max(1, n_hist))
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        self.wake_pub  = self.create_lifecycle_publisher(Bool,    'wake_detected', 10)
        self.score_pub = self.create_lifecycle_publisher(Float32, 'wake_score',    10)

        self.get_logger().info(f'Loading model: {model_path}')
        self.model = Model(
            wakeword_models=[model_path],
            inference_framework='onnx',
        )
        self.model_key = None
        self.create_subscription(Float32MultiArray, 'raw_audio', self._audio_callback, 20)
        self.get_logger().info(
            f"Wake word ready (threshold={self.threshold}, patience={self.patience}, "
            f"debounce={self.debounce_sec}s, save_dir={self.save_dir or 'off'})")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.wake_pub.on_activate(state)
        self.score_pub.on_activate(state)
        self._active = True
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self._active = False
        self._buf = np.zeros(0, dtype=np.int16)
        self._frames_above = 0
        self.wake_pub.on_deactivate(state)
        self.score_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

    def _audio_callback(self, msg: Float32MultiArray):
        # The subscription exists from on_configure — skip inference until activated
        if not self._active:
            return
        # float32 [-1, 1] → int16 (openWakeWord expects int16)
        audio = (np.array(msg.data, dtype=np.float32) * 32768.0).astype(np.int16)

        # Feed the model exactly one 80 ms frame per predict() call, so that
        # `patience` counts real model frames. With smaller chunks (e.g. 512)
        # predict() just repeats the previous score.
        self._buf = np.concatenate((self._buf, audio))
        while len(self._buf) >= FRAME_SAMPLES:
            frame, self._buf = self._buf[:FRAME_SAMPLES], self._buf[FRAME_SAMPLES:]
            self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray):
        self._history.append(frame)
        prediction = self.model.predict(frame)

        # Determine the model key on the first call
        if self.model_key is None and prediction:
            self.model_key = list(prediction.keys())[0]

        if self.model_key is None:
            return

        score = float(prediction[self.model_key])

        score_msg = Float32()
        score_msg.data = score
        self.score_pub.publish(score_msg)

        self._frames_above = self._frames_above + 1 if score >= self.threshold else 0

        now = time.time()

        if self._frames_above >= self.patience and (now - self.last_activation) >= self.debounce_sec:
            self.last_activation  = now
            self.activation_count += 1
            self._frames_above    = 0
            wake_msg = Bool()
            wake_msg.data = True
            self.wake_pub.publish(wake_msg)
            self.get_logger().info(
                f"Wake word detected! Score={score:.3f} "
                f"(activation #{self.activation_count})"
            )
            if self.save_dir:
                self._save_activation(score)

    def _save_activation(self, score: float):
        """Write the audio that triggered the activation, e.g. 20261003_142048_0.990.wav."""
        name = time.strftime('%Y%m%d_%H%M%S') + f'_{score:.3f}.wav'
        try:
            with wave.open(os.path.join(self.save_dir, name), 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(np.concatenate(list(self._history)).tobytes())
        except OSError as e:
            self.get_logger().warning(f'Could not save activation audio: {e}')

    # destroy_node replaced by lifecycle callbacks


def main():
    rclpy.init()
    node = WakeWordNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
