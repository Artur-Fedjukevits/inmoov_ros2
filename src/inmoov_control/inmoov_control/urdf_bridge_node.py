"""
urdf_bridge_node.py — servo world <-> URDF world.

The motion stack of inmoov_control speaks *servo* joints (thumb_R, bicep_L, ...)
in the "radians around 90 deg" convention; the robot model (inmoov_i2.urdf, the
Android app, RViz / robot_state_publisher, MoveIt) speaks *URDF* joints in URDF
radians. This node converts both ways with the calibration table
servo_urdf_map.yaml (see servo_urdf_map.py):

  /joint_states, /face_joint_states  (servo names, commanded pose)
        └─> /urdf_joint_states       (sensor_msgs/JointState, URDF names, 41 joints)

  /urdf_joint_cmd                    (inmoov_msgs/JointCommand, URDF names, URDF rad)
        └─> /joint_cmd               (same source / lease / release, priority capped
                                      at PRIORITY_REMOTE, servo names, servo rad;
                                      velocity converted; right eye dropped, see below)

  ~/get_map        (std_srvs/Trigger)  message = JSON of the whole table
  ~/set_calibration (std_msgs/String)  JSON, see _set_calibration_cb; saved to
                                       ~/.config/inmoov/servo_urdf_map.yaml
                                       (or to map_file, if that parameter is set)
  ~/status         (std_msgs/String, latched) JSON: map source, last calibration result

The eyes are mirrored by arduino_comm_node (EYE_SYNC): a command for one eye
moves both. /urdf_joint_cmd therefore only drives the left (leading) eye — the
right-eye joints are ignored, otherwise the two eyes in one command would race
and the last one would win for both.

There is no servo feedback: /urdf_joint_states is the *commanded* pose after
arbitration (what joint_state_publisher reports), clamped to the URDF limits.

The Android app talks to these topics through rosbridge (ws://<robot>:9090).
For RViz on the real robot: robot_state_publisher --ros-args -r joint_states:=/urdf_joint_states

Author: Artur Fedjukevits
Assisted by: Claude (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import json
import math
import os

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from inmoov_msgs.msg import JointCommand

from .servo_urdf_map import (USER_MAP_PATH, JointMap, ServoUrdfMap,
                             servo_deg_to_rad, servo_rad_to_deg)


def _default_map_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('inmoov_control'),
                            'config', 'servo_urdf_map.yaml')
    except Exception:  # running from the source tree
        return os.path.join(os.path.dirname(__file__), '..', 'config', 'servo_urdf_map.yaml')


# Mirrored by arduino_comm_node from eye_lr_L / eye_ud_L (EYE_SYNC); the left eye
# leads, as in vision_head_tracker_node
_EYE_FOLLOWERS = frozenset({'eye_lr_R', 'eye_ud_R'})


class UrdfBridgeNode(LifecycleNode):

    def __init__(self):
        super().__init__('urdf_bridge')
        self.declare_parameter('map_file', '')          # '' = shipped table (+ user overrides)
        self.declare_parameter('publish_rate_hz', 25.0)
        self._map: ServoUrdfMap = None
        self._shipped_path = ''
        self._map_file = ''       # the map_file parameter; '' = shipped + user file
        self._servo_deg = {}      # servo name -> last reported servo angle (deg)
        self._subs = []
        self._srv = None
        self._state_pub = None
        self._cmd_pub = None
        self._status_pub = None
        self._timer = None
        self._last_calib = ''

    # ------------------------------------------------------------ lifecycle
    def on_configure(self, state):
        self._map_file = self.get_parameter('map_file').value
        self._shipped_path = _default_map_path()
        try:
            self._map = (ServoUrdfMap.load(self._map_file) if self._map_file
                         else ServoUrdfMap.load_default(self._shipped_path))
        except Exception as e:   # noqa: BLE001 — bad YAML must not crash the stack
            self.get_logger().error(f'urdf_bridge: cannot load map: {e}')
            return TransitionCallbackReturn.FAILURE
        # until something is reported, servos are at their firmware rest angles
        self._servo_deg = {j.servo: j.servo_rest for j in self._map.joints}

        self._subs = [
            self.create_subscription(JointState, '/joint_states', self._servo_state_cb, 10),
            self.create_subscription(JointState, '/face_joint_states', self._servo_state_cb, 10),
            self.create_subscription(JointCommand, '/urdf_joint_cmd', self._urdf_cmd_cb, 20),
            self.create_subscription(String, '~/set_calibration', self._set_calibration_cb, 10),
        ]
        self._srv = self.create_service(Trigger, '~/get_map', self._get_map_cb)
        self._state_pub = self.create_lifecycle_publisher(JointState, '/urdf_joint_states', 10)
        self._cmd_pub = self.create_lifecycle_publisher(JointCommand, '/joint_cmd', 20)
        self._status_pub = self.create_lifecycle_publisher(
            String, '~/status', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        n_ok = sum(j.verified for j in self._map.joints)
        self.get_logger().info(
            f'urdf_bridge: {len(self._map.joints)} joints ({n_ok} calibrated), '
            f'map: {self._map.source}')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        for p in (self._state_pub, self._cmd_pub, self._status_pub):
            p.on_activate(state)
        rate = max(1.0, float(self.get_parameter('publish_rate_hz').value))
        self._timer = self.create_timer(1.0 / rate, self._publish_state)
        self._publish_status()
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        if self._timer:
            self.destroy_timer(self._timer)
            self._timer = None
        for p in (self._state_pub, self._cmd_pub, self._status_pub):
            p.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        for s in self._subs:
            self.destroy_subscription(s)
        self._subs = []
        if self._srv:
            self.destroy_service(self._srv)
            self._srv = None
        for p in (self._state_pub, self._cmd_pub, self._status_pub):
            if p is not None:
                self.destroy_lifecycle_publisher(p)
        self._state_pub = self._cmd_pub = self._status_pub = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        if self._timer:
            self.destroy_timer(self._timer)
            self._timer = None
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

    # ----------------------------------------------------- servo -> URDF
    def _servo_state_cb(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            if name in self._servo_deg and math.isfinite(pos):
                self._servo_deg[name] = servo_rad_to_deg(pos)

    def _publish_state(self) -> None:
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        for j in self._map.joints:
            js.name.append(j.urdf)
            js.position.append(j.servo_to_urdf(self._servo_deg[j.servo]))
        self._state_pub.publish(js)

    # ----------------------------------------------------- URDF -> servo
    def _urdf_cmd_cb(self, msg: JointCommand) -> None:
        out = JointCommand()
        out.source = msg.source or 'remote'
        # A network client must not outrank the behavior tree's own overrides;
        # calibration (priority 90) goes straight to /joint_cmd, not through here
        out.priority = min(msg.priority, JointCommand.PRIORITY_REMOTE)
        out.lease_sec = msg.lease_sec
        out.release = msg.release
        cmd = msg.cmd
        has_vel = len(cmd.velocity) == len(cmd.name)
        unknown = []
        for i, name in enumerate(cmd.name):
            j = self._map.by_urdf.get(name)
            if j is None:
                unknown.append(name)
                continue
            if j.servo in _EYE_FOLLOWERS:
                continue    # mirrored from the left eye by the firmware side
            if msg.release:
                out.cmd.name.append(j.servo)
                continue
            if i >= len(cmd.position) or not math.isfinite(cmd.position[i]):
                continue
            rad = cmd.position[i]
            out.cmd.name.append(j.servo)
            out.cmd.position.append(servo_deg_to_rad(j.urdf_to_servo(rad)))
            if has_vel:
                v = cmd.velocity[i]
                out.cmd.velocity.append(
                    self._map.urdf_velocity_to_servo(name, rad, v)
                    if math.isfinite(v) and v != 0.0 else 0.0)
        if out.cmd.velocity and len(out.cmd.velocity) != len(out.cmd.name):
            out.cmd.velocity = []
        if unknown:
            self.get_logger().warn(
                f'urdf_bridge: ignored non-actuated joints {unknown[:5]}',
                throttle_duration_sec=5.0)
        if out.cmd.name:
            out.cmd.header.stamp = self.get_clock().now().to_msg()
            self._cmd_pub.publish(out)

    # ------------------------------------------------------- calibration
    def _get_map_cb(self, request, response):
        d = self._map.to_dict()
        d['source'] = self._map.source
        d['servo_deg'] = {k: round(v, 2) for k, v in self._servo_deg.items()}
        response.success = True
        response.message = json.dumps(d)
        return response

    def _set_calibration_cb(self, msg: String) -> None:
        """JSON: {"servo": "bicep_R", "points": [[deg, rad], ...], "verified": true}
        or {"servo": "bicep_R", "reset": true} — back to the shipped table.
        Calibrated (verified) joints are saved to the user file; the status
        reports "persisted": false for a "verified": false table (applied only
        until restart). With map_file set, the whole table is saved there."""
        try:
            d = json.loads(msg.data)
            servo = d['servo']
            old = self._map.by_servo[servo]
            if d.get('reset'):
                shipped = ServoUrdfMap.load(self._shipped_path).by_servo[servo]
                new = shipped
            else:
                new = JointMap(old.servo, old.urdf, [old.servo_min, old.servo_max],
                               old.servo_rest, d['points'], old.urdf_limits,
                               d.get('verified', True), old.note, old.board)
            joints = [new if j.servo == servo else j for j in self._map.joints]
            if self._map_file:
                # an explicit map_file is loaded whole (no merge) — save it whole
                self._map = ServoUrdfMap(joints, self._map.unmapped_servos, self._map_file)
                self._map.save(self._map_file)
                persisted = True
            else:
                self._map = ServoUrdfMap(joints, self._map.unmapped_servos,
                                         f'{USER_MAP_PATH} + {self._shipped_path}')
                # only calibrated joints go to the user file — the rest keep following
                # the shipped table (so its future fixes are not frozen)
                ServoUrdfMap([j for j in joints if j.verified]).save(USER_MAP_PATH)
                # "verified": false is applied now but not kept across a restart
                persisted = new.verified or bool(d.get('reset'))
            self._last_calib = json.dumps({'ok': True, 'servo': servo,
                                           'points': new.to_dict()['points'],
                                           'persisted': persisted})
            self.get_logger().info(
                f'urdf_bridge: calibration of {servo} applied'
                f'{"" if persisted else " (not saved: verified=false)"}: {new.points}')
        except Exception as e:   # noqa: BLE001 — report to the app, keep running
            self._last_calib = json.dumps({'ok': False, 'error': str(e)})
            self.get_logger().error(f'urdf_bridge: bad calibration message: {e}')
        self._publish_status()

    def _publish_status(self) -> None:
        if self._status_pub is None:
            return
        s = String()
        s.data = json.dumps({
            'map_source': self._map.source,
            'joints': len(self._map.joints),
            'calibrated': sum(j.verified for j in self._map.joints),
            'last_calibration': json.loads(self._last_calib) if self._last_calib else None,
        })
        self._status_pub.publish(s)


def main(args=None):
    rclpy.init(args=args)
    node = UrdfBridgeNode()
    executor = SingleThreadedExecutor()
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
