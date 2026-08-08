#!/usr/bin/env python3

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

    def _dp(self, name, default=None):
        """Безопасный declare_parameter: игнорирует повторное объявление при re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp(
            'model_path',
            '/home/artur/openWakeWord/my_custom_model/ey_lyonya.onnx'
        )
        self._dp('threshold',    0.2)
        self._dp('debounce_sec', 1.5)

        model_path        = self.get_parameter('model_path').value
        self.threshold    = self.get_parameter('threshold').value
        self.debounce_sec = self.get_parameter('debounce_sec').value

        self.wake_pub  = self.create_lifecycle_publisher(Bool,    'wake_detected', 10)
        self.score_pub = self.create_lifecycle_publisher(Float32, 'wake_score',    10)

        self.get_logger().info(f'Загрузка модели: {model_path}')
        self.model = Model(
            wakeword_models=[model_path],
            inference_framework='onnx',
        )
        self.model_key = None
        self.create_subscription(Float32MultiArray, 'raw_audio', self._audio_callback, 20)
        self.get_logger().info(
            f"Wake word готов (порог={self.threshold}, debounce={self.debounce_sec}с)")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.wake_pub.on_activate(state)
        self.score_pub.on_activate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
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
        # float32 [-1, 1] → int16 (openWakeWord ожидает int16)
        audio = (np.array(msg.data, dtype=np.float32) * 32768.0).astype(np.int16)

        prediction = self.model.predict(audio)

        # Определяем ключ модели при первом вызове
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
                f"Wake word обнаружен! Score={score:.3f} "
                f"(активация #{self.activation_count})"
            )

    # destroy_node заменён на lifecycle callbacks


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
