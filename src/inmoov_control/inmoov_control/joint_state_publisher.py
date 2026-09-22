"""
joint_state_publisher.py — Publishes /joint_states and /face_joint_states.

Reads the current servo positions via an echo channel (the Arduino publishes
its values only on change, or on request). For debugging and visualisation in
rviz2 — subscribes to /joint_command and /face_command and re-publishes them
as /joint_states / /face_joint_states with a proper header.stamp.

NOTE: in the current implementation no echo channel is read — the node only
re-publishes the last *commanded* positions (see the JointStatePublisher
class docstring).

Usage:
  Run alongside arduino_right_node and arduino_left_node.
  Subscribers: policy inference node, rviz2, rosbag for data collection.

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import math
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from sensor_msgs.msg import JointState

from .arduino_right_node import ArduinoRightNode
from .arduino_left_node  import ArduinoLeftNode


BODY_JOINT_NAMES = (
    [n for n, _, _ in ArduinoRightNode.BODY_JOINTS] +
    [n for n, _, _ in ArduinoLeftNode.BODY_JOINTS]
)

FACE_JOINT_NAMES = (
    [n for n, _, _ in ArduinoRightNode.FACE_JOINTS] +
    [n for n, _, _ in ArduinoLeftNode.FACE_JOINTS]
)


class JointStatePublisher(LifecycleNode):
    """
    Tracks the last commanded positions and re-publishes them as /joint_states
    and /face_joint_states at 50 Hz. This gives downstream nodes (policy,
    rviz2) a consistent timestamped view of robot state.

    NOTE: This reflects *commanded* state, not measured state (no encoders).
    For true state estimation you need joint sensors or motor feedback.
    """

    def __init__(self):
        super().__init__('joint_state_publisher')
        self._body     = {n: 0.0 for n in BODY_JOINT_NAMES}
        self._face     = {n: 0.0 for n in FACE_JOINT_NAMES}
        self._js_pub   = None
        self._face_pub = None
        self._timer    = None

    # ── Lifecycle: Phase 2 ─────────────────────────────────────────────────

    def on_configure(self, state):
        self.create_subscription(JointState, '/joint_command', self._body_cb, 10)
        self.create_subscription(JointState, '/face_command',  self._face_cb, 10)
        self._js_pub   = self.create_lifecycle_publisher(JointState, '/joint_states',      10)
        self._face_pub = self.create_lifecycle_publisher(JointState, '/face_joint_states', 10)
        self.get_logger().info(
            f'joint_state_publisher: tracking {len(self._body)} body + '
            f'{len(self._face)} face joints')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._js_pub.on_activate(state)
        self._face_pub.on_activate(state)
        self._timer = self.create_timer(0.02, self._publish)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        if self._timer:
            self.destroy_timer(self._timer)
            self._timer = None
        self._js_pub.on_deactivate(state)
        self._face_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

    def _body_cb(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            if name in self._body:
                self._body[name] = pos

    def _face_cb(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            if name in self._face:
                self._face[name] = pos

    def _publish(self) -> None:
        now = self.get_clock().now().to_msg()

        js = JointState()
        js.header.stamp    = now
        js.header.frame_id = 'base_link'
        js.name     = list(self._body.keys())
        js.position = list(self._body.values())
        self._js_pub.publish(js)

        fs = JointState()
        fs.header.stamp    = now
        fs.header.frame_id = 'head_link'
        fs.name     = list(self._face.keys())
        fs.position = list(self._face.values())
        self._face_pub.publish(fs)



def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
