#!/usr/bin/env python3

"""
openwakeword_node.py
=====================
Wake word detection ("Эй Лёня") on a custom openWakeWord (ONNX) model.

Subscribes:
  raw_audio  (Float32MultiArray) — 16kHz float32 audio chunks from audio_source_node

Publishes:
  wake_detected (Bool)    — True on activation (debounced)
  wake_score    (Float32) — raw model score for every chunk, for tuning/diagnostics

Parameters:
  model_path    (str)   — path to the custom .onnx wake word model
  threshold     (float) — activation score threshold (default 0.2)
  debounce_sec  (float) — minimum interval between two activations (default 1.5)

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import os
import time

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from std_msgs.msg import Bool, Float32, Float32MultiArray
import numpy as np
from openwakeword.model import Model


class WakeWordNode(LifecycleNode):
    def __init__(self):
        super().__init__('wakeword_node')
        self.model            = None
        self.model_key        = None
        self.threshold        = 0.2
        self.debounce_sec     = 1.5
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
        self._dp('threshold',    0.2)
        self._dp('debounce_sec', 1.5)

        model_path        = self.get_parameter('model_path').value
        self.threshold    = self.get_parameter('threshold').value
        self.debounce_sec = self.get_parameter('debounce_sec').value

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
            f"Wake word ready (threshold={self.threshold}, debounce={self.debounce_sec}s)")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.wake_pub.on_activate(state)
        self.score_pub.on_activate(state)
        self._active = True
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self._active = False
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

        prediction = self.model.predict(audio)

        # Determine the model key on the first call
        if self.model_key is None and prediction:
            self.model_key = list(prediction.keys())[0]

        if self.model_key is None:
            return

        score = float(prediction[self.model_key])

        score_msg = Float32()
        score_msg.data = score
        self.score_pub.publish(score_msg)

        now = time.time()

        if score >= self.threshold and (now - self.last_activation) >= self.debounce_sec:
            self.last_activation  = now
            self.activation_count += 1
            wake_msg = Bool()
            wake_msg.data = True
            self.wake_pub.publish(wake_msg)
            self.get_logger().info(
                f"Wake word detected! Score={score:.3f} "
                f"(activation #{self.activation_count})"
            )

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
