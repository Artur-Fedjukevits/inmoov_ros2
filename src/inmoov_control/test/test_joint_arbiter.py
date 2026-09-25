"""
test_joint_arbiter.py — per-joint arbitration (no hardware needed).

  JointArbiter: free / lease / priority / same-source refresh / expiry /
  release / write-only (lease 0) / shared eye group.
  ArduinoLeftNode._handle_cmd: the real conflicts it was built for —
  tracker REST vs a BT head command, expression vs tracker eyes, expression
  vs TTS jaw, legacy topic — plus the default-speed rule.

Run:
  cd /home/artur/ros2_ws && source install/setup.bash
  python3 -m pytest src/inmoov_control/test/test_joint_arbiter.py -v
"""

import math
import os
import sys

import pytest
import rclpy
from sensor_msgs.msg import JointState

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inmoov_control.joint_arbiter import JointArbiter      # noqa: E402
from inmoov_control.arduino_left_node import ArduinoLeftNode  # noqa: E402
from inmoov_msgs.msg import JointCommand as JC                # noqa: E402

GROUPS = {'eye_lr_L': 'eye_lr', 'eye_lr_R': 'eye_lr'}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


# ─────────────────────────────────────────────────────────────────────────────
# JointArbiter

def test_free_joint_and_lease_blocks_lower_and_equal():
    c = Clock()
    a = JointArbiter(clock=c)
    assert a.claim('rothead', 'bt_command', 70, 5.0)[0]
    ok, owner = a.claim('rothead', 'head_tracker', 40, 0.5)
    assert not ok and owner.source == 'bt_command'
    assert not a.claim('rothead', 'other_70', 70, 1.0)[0]      # equal priority: owner keeps it
    assert a.claim('rothead', 'bt_command', 70, 5.0)[0]       # same source refreshes
    assert a.claim('rothead', 'calibration', 90, 2.0)[0]      # higher priority takes over
    assert a.owners()['rothead']['source'] == 'calibration'
    assert a.rejected == 2


def test_lease_expires():
    c = Clock()
    a = JointArbiter(clock=c)
    a.claim('neck', 'bt_scan', 60, 3.0)
    c.t = 2.9
    assert not a.claim('neck', 'head_tracker', 40, 0.5)[0]
    c.t = 3.0
    assert a.claim('neck', 'head_tracker', 40, 0.5)[0]


def test_release_only_by_owner():
    a = JointArbiter(clock=Clock())
    a.claim('midstom', 'bt_scan', 60, 10.0)
    assert not a.release('midstom', 'head_tracker')
    assert not a.claim('midstom', 'head_tracker', 40, 0.5)[0]
    assert a.release('midstom', 'bt_scan')
    assert a.claim('midstom', 'head_tracker', 40, 0.5)[0]


def test_write_only_takes_no_ownership():
    a = JointArbiter(clock=Clock())
    assert a.claim('jaw', 'expression', 30, 0.0)[0]
    assert a.owners() == {}
    assert a.claim('jaw', 'blink', 20, 0.0)[0]                # nothing held → lower prio writes


def test_eye_group_shared():
    a = JointArbiter(GROUPS, clock=Clock())
    a.claim('eye_lr_L', 'head_tracker', 40, 0.5)
    assert not a.claim('eye_lr_R', 'expression', 30, 0.0)[0]  # mirror is covered


# ─────────────────────────────────────────────────────────────────────────────
# ArduinoLeftNode._handle_cmd (no serial, no executor)

@pytest.fixture(scope='module', autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = ArduinoLeftNode(serial_port='/dev/null')
    yield n
    n.destroy_node()


def _js(**deg):
    js = JointState()
    js.name = list(deg)
    js.position = [(v - 90.0) * math.pi / 180.0 for v in deg.values()]
    return js


def _deg(node, name):
    if name in node._body_map:
        return node._body_degs[node._body_map[name]]
    return node._face_degs[node._face_map[name]]


def test_tracker_rest_does_not_override_bt_head_command(node):
    """Replaces the old 200 ms 'publish after the tracker's REST' hack."""
    node._handle_cmd('bt_command', JC.PRIORITY_BT_COMMAND, 8.0, False, _js(rothead=130, neck=60))
    node._handle_cmd('head_tracker', JC.PRIORITY_HEAD_TRACKER, 0.0, False, _js(rothead=90, neck=40))
    assert _deg(node, 'rothead') == 130 and _deg(node, 'neck') == 60
    node._handle_cmd('bt_command', 0, 0.0, True, _js(rothead=0, neck=0))   # release
    node._handle_cmd('head_tracker', JC.PRIORITY_HEAD_TRACKER, 0.5, False, _js(rothead=95, neck=45))
    assert _deg(node, 'rothead') == 95


def test_expression_cannot_move_tracked_eyes_or_speaking_jaw(node):
    node._handle_cmd('head_tracker', JC.PRIORITY_HEAD_TRACKER, 0.5, False, _js(eye_lr_L=85))
    node._handle_cmd('tts_jaw', JC.PRIORITY_TTS_JAW, 0.5, False, _js(jaw=60))
    node._handle_cmd('expression', JC.PRIORITY_EXPRESSION, 1.0, False,
                     _js(eye_lr_L=90, jaw=10, eyebrow_L=100))
    assert _deg(node, 'eye_lr_L') == 85        # tracker keeps the eyes
    assert _deg(node, 'jaw') == 60             # TTS keeps the jaw
    assert _deg(node, 'eyebrow_L') == 100      # free joint → expression applies


def test_legacy_topic_is_lowest_priority(node):
    node._legacy_cmd_cb(_js(midstom=100))
    assert _deg(node, 'midstom') == 100        # nobody holds it
    node._handle_cmd('bt_scan', JC.PRIORITY_BT_SCAN, 5.0, False, _js(midstom=60))
    node._legacy_cmd_cb(_js(midstom=120))
    assert _deg(node, 'midstom') == 60


def test_speed_default_when_no_velocity(node):
    i = node._body_map['rothead']
    js = _js(rothead=120)
    js.velocity = [1.0]                        # scan speed ≈ 57°/s → step 3
    node._handle_cmd('bt_scan', JC.PRIORITY_BT_SCAN, 0.0, False, js)
    assert node._body_speeds[i] == 3
    node._handle_cmd('head_tracker', JC.PRIORITY_HEAD_TRACKER, 0.0, False, _js(rothead=100))
    assert node._body_speeds[i] == ArduinoLeftNode.DEFAULT_STEPS[i] == 1   # not inherited


def test_eye_mirror_and_right_eye_command(node):
    """Right-camera fallback publishes eye_lr_R: the left board applies it to eye_lr_L."""
    node._handle_cmd('head_tracker', JC.PRIORITY_HEAD_TRACKER, 0.5, False, _js(eye_lr_R=95))
    assert _deg(node, 'eye_lr_L') == 95
