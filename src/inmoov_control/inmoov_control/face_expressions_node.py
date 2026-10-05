"""
face_expressions_node.py — InMoov face expression library for ROS2.

Source files ported from MRL InMoov2:
  gestures/faceExpressions.py
  EyebrowMovements.py, EyelidMovements.py, CheekMovements.py, EyeMovements.py

Usage as a library (from behavior_manager_node or similar):
  from inmoov_control.face_expressions_node import FaceExpressions
  fe = FaceExpressions(node)
  fe.happy()

Usage as a standalone node:
  ros2 run inmoov_control face_expressions_node
  ros2 topic pub /face_expression std_msgs/msg/String "data: happy"
  ros2 topic pub /face_expression_hold std_msgs/msg/String "data: happy"

Calibration:
  Run `ros2 run inmoov_control face_expression_calibrator` to tune expression
  positions interactively. Results are saved to USER_CALIB_FILE
  ($INMOOV_FACE_CALIBRATION or ~/.config/inmoov/face_expressions_calibration.json),
  which survives rebuilds and is loaded on import. Without it the
  face_expressions_calibration.json shipped with the package is used.

Commands go out as inmoov_msgs/JointCommand on /joint_cmd (source 'expression',
priority 30, see FaceExpressions._ANIM_LEASE_SEC). The firmware treats every
angle literally and clamps it to the joint's min/max (0 is 0°, not "rest").

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import json
import math
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from inmoov_msgs.msg import JointCommand

# ─────────────────────────────────────────────────────────────────────────────
# Servo hardware limits & rest positions
# ─────────────────────────────────────────────────────────────────────────────

# Rest (neutral) positions in degrees — from arduino_left_node.py / arduino_right_node.py
FACE_REST: dict[str, int] = {
    'eyelid_L_Upper':  85,    # PCA ch 6,  INV, min=70,  max=95
    'eyelid_L_Lower':  85,    # PCA ch 7,       min=75,  max=95
    'eyelid_R_Upper':  85,    # PCA ch 8,       min=65,  max=100
    'eyelid_R_Lower':  85,    # PCA ch 9,  INV, min=70,  max=95
    'eyebrow_L':       90,    # PCA ch 10, INV, min=60,  max=110
    'eyebrow_R':       80,    # PCA ch 11,      min=70,  max=105
    'cheek_L':        100,    # PCA ch 14, INV, min=75,  max=115
    'cheek_R':         87,    # PCA ch 15,      min=68,  max=105
    'forhead_L':       90,    # PCA ch 12, INV, min=90,  max=110
    'forhead_R':       85,    # PCA ch 13,      min=85,  max=105
    'eye_lr_L':        90,    # GPIO pin 22, min=80,  max=100
    'eye_ud_L':       100,    # GPIO pin 24, min=80,  max=110  (INV in firmware)
    'jaw':             10,    # GPIO pin 26, min=10,  max=90   (firmware limit)
    'eye_lr_R':        90,    # GPIO pin 22, min=80,  max=100
    'eye_ud_R':       100,    # GPIO pin 24, min=85,  max=115
    'upperLip':        90,    # GPIO pin 26, min=90,  max=105
}

# Shorthand for firmware min/max (val=0 → rest quirk: never send 0)
_MN = {
    'eyelid_L_Upper':  70, 'eyelid_L_Lower':  75,
    'eyelid_R_Upper':  65, 'eyelid_R_Lower':  70,
    'eyebrow_L':       60, 'eyebrow_R':       70,
    'cheek_L':         75, 'cheek_R':         68,
    'forhead_L':       90, 'forhead_R':       85,
    'eye_lr_L':        80, 'eye_ud_L':        80,
    'eye_lr_R':        80, 'eye_ud_R':        85,
    'upperLip':        90, 'jaw':             10,   # jaw closed (rest = min)
}
_MX = {
    'eyelid_L_Upper':  95, 'eyelid_L_Lower':  95,
    'eyelid_R_Upper': 100, 'eyelid_R_Lower':  95,
    'eyebrow_L':      110, 'eyebrow_R':      105,
    'cheek_L':        115, 'cheek_R':        105,
    'forhead_L':      110, 'forhead_R':      105,
    'eye_lr_L':       100, 'eye_ud_L':       110,
    'eye_lr_R':       100, 'eye_ud_R':       115,
    'upperLip':       105, 'jaw':             90,   # jaw fully open (firmware max)
}

# ─────────────────────────────────────────────────────────────────────────────
# Expression data — default positions (translated from MRL InMoov2)
# Each entry: {joint_name: degrees}.  Joints absent → use FACE_REST.
# Override with face_expressions_calibration.json (written by calibrator GUI).
# ─────────────────────────────────────────────────────────────────────────────

EXPRESSIONS_DATA: dict[str, dict[str, int]] = {
    'neutral': {},
    'angry': {
        'forhead_L':      _MX['forhead_L'],  'forhead_R':      _MX['forhead_R'],
        'eyelid_L_Upper': 70,                'eyelid_L_Lower': 70,
        'eyelid_R_Upper': 70,                'eyelid_R_Lower': 70,
        'upperLip':       _MX['upperLip'],
        'cheek_L':        _MN['cheek_L'],    'cheek_R':        _MN['cheek_R'],
        'eyebrow_L':      _MN['eyebrow_L'],  'eyebrow_R':      _MN['eyebrow_R'],
    },
    'wink': {
        'eyelid_L_Upper': _MN['eyelid_L_Upper'],
        'eyelid_L_Lower': _MN['eyelid_L_Lower'],
    },
    'disgust': {
        'upperLip':       _MX['upperLip'],
        'forhead_L':      _MN['forhead_L'],  'forhead_R':      _MN['forhead_R'],
        'eyelid_L_Lower': _MN['eyelid_L_Lower'],
        'eyelid_R_Lower': _MN['eyelid_R_Lower'],
        'cheek_R':        _MX['cheek_R'],
        'eyebrow_L':      _MN['eyebrow_L'],  'eyebrow_R':      _MN['eyebrow_R'],
    },
    'fear': {
        'eyelid_L_Upper': _MX['eyelid_L_Upper'], 'eyelid_L_Lower': _MX['eyelid_L_Lower'],
        'eyelid_R_Upper': _MX['eyelid_R_Upper'], 'eyelid_R_Lower': _MX['eyelid_R_Lower'],
        'cheek_L':        _MN['cheek_L'],        'cheek_R':        _MN['cheek_R'],
        'eyebrow_L':      _MX['eyebrow_L'],      'eyebrow_R':      _MX['eyebrow_R'],
        'forhead_L':      _MX['forhead_L'],      'forhead_R':      _MX['forhead_R'],
    },
    'happy': {
        'eyebrow_L':      _MX['eyebrow_L'],   'eyebrow_R':      _MX['eyebrow_R'],
        'cheek_L':        _MX['cheek_L'],      'cheek_R':        _MX['cheek_R'],
        'upperLip':       90,
        'eyelid_L_Lower': _MN['eyelid_L_Lower'],
        'eyelid_R_Lower': _MN['eyelid_R_Lower'],
        'jaw':            _MX['jaw'],
    },
    'smile': {
        'cheek_L':        _MX['cheek_L'],     'cheek_R':        _MX['cheek_R'],
        'eyelid_L_Lower': _MN['eyelid_L_Lower'],
        'eyelid_R_Lower': _MN['eyelid_R_Lower'],
        'jaw':            _MX['jaw'],
    },
    'sad': {
        'eyelid_L_Upper': _MN['eyelid_L_Upper'], 'eyelid_R_Upper': _MN['eyelid_R_Upper'],
        'cheek_L':        _MN['cheek_L'],        'cheek_R':        _MN['cheek_R'],
    },
    'sigh': {
        'eye_lr_L':       _MX['eye_lr_L'],   'eye_ud_L':       _MX['eye_ud_L'],
        'eye_lr_R':       _MX['eye_lr_R'],   'eye_ud_R':       _MX['eye_ud_R'],
    },
    'sorry': {
        'eyebrow_L':      _MX['eyebrow_L'],  'eyebrow_R':      _MX['eyebrow_R'],
        'cheek_L':        _MN['cheek_L'],    'cheek_R':        _MN['cheek_R'],
    },
    'suspicious': {
        'upperLip':       _MN['upperLip'],
        'forhead_R':      _MX['forhead_R'],  'forhead_L':      _MN['forhead_L'],
        'cheek_L':        _MX['cheek_L'],
        'eyebrow_R':      _MN['eyebrow_R'],  'eyebrow_L':      _MX['eyebrow_L'],
        'eyelid_L_Upper': _MX['eyelid_L_Upper'], 'eyelid_L_Lower': _MX['eyelid_L_Lower'],
        'eyelid_R_Upper': 70,                    'eyelid_R_Lower': 70,
    },
    'thinking': {
        'eyebrow_L':      _MN['eyebrow_L'],  'eyebrow_R':      _MN['eyebrow_R'],
        'forhead_L':      _MX['forhead_L'],  'forhead_R':      _MX['forhead_R'],
        'eyelid_L_Lower': _MN['eyelid_L_Lower'],
        'eyelid_R_Lower': _MN['eyelid_R_Lower'],
        'eye_lr_L':       _MX['eye_lr_L'],   'eye_ud_L':       _MX['eye_ud_L'],
        'eye_lr_R':       _MX['eye_lr_R'],   'eye_ud_R':       _MX['eye_ud_R'],
    },
    'unamused': {
        'eyebrow_R':      _MN['eyebrow_R'],
        'forhead_L':      _MX['forhead_L'],  'forhead_R':      _MN['forhead_R'],
        'eyelid_L_Upper': 70,  'eyelid_L_Lower': 70,
        'eyelid_R_Upper': 70,  'eyelid_R_Lower': 70,
        'eye_lr_L':       _MN['eye_lr_L'],   'eye_lr_R':       _MN['eye_lr_R'],
        'cheek_L':        _MN['cheek_L'],    'cheek_R':        _MN['cheek_R'],
    },
    'surprise': {
        'eyebrow_L':      _MX['eyebrow_L'],  'eyebrow_R':      _MX['eyebrow_R'],
        'eyelid_L_Upper': _MX['eyelid_L_Upper'], 'eyelid_R_Upper': _MX['eyelid_R_Upper'],
        'forhead_L':      _MN['forhead_L'],  'forhead_R':      _MN['forhead_R'],
        'jaw':            _MX['jaw'],
    },
    'sleeping': {
        'eyelid_L_Upper': _MN['eyelid_L_Upper'], 'eyelid_L_Lower': _MN['eyelid_L_Lower'],
        'eyelid_R_Upper': _MN['eyelid_R_Upper'], 'eyelid_R_Lower': _MN['eyelid_R_Lower'],
    },
}

# Load calibration overrides from JSON (written by face_expression_calibrator GUI).
# The user file wins; otherwise the calibration shipped with the package is used.
_PKG_CALIB_FILE = os.path.join(os.path.dirname(__file__), 'face_expressions_calibration.json')
USER_CALIB_FILE = os.environ.get(
    'INMOOV_FACE_CALIBRATION',
    os.path.expanduser('~/.config/inmoov/face_expressions_calibration.json'))
_CALIB_FILE = USER_CALIB_FILE if os.path.exists(USER_CALIB_FILE) else _PKG_CALIB_FILE
if os.path.exists(_CALIB_FILE):
    with open(_CALIB_FILE, encoding='utf-8') as _f:
        for _expr, _pos in json.load(_f).items():
            if _expr in EXPRESSIONS_DATA:
                EXPRESSIONS_DATA[_expr] = _pos


def _r(deg: float) -> float:
    """Convert servo degree (0–180, center=90) to radians for face_command."""
    return (float(deg) - 90.0) * math.pi / 180.0


# ─────────────────────────────────────────────────────────────────────────────
# FaceExpressions class
# ─────────────────────────────────────────────────────────────────────────────

class FaceExpressions:
    """
    Publishes face servo commands to /face_command (sensor_msgs/JointState).

    Expression positions are loaded from EXPRESSIONS_DATA (defaults) overridden
    by the calibration JSON (_CALIB_FILE, see the module docstring).

    Use the face_expression_calibrator GUI to tune positions interactively.
    """

    EXPRESSION_MAP: dict[str, str] = {
        'neutral':          'neutral',
        'angry':            'angry',      'anger':      'angry',
        'wink':             'wink',
        'disgust':          'disgust',
        'fear':             'fear',
        'happy':            'happy',
        'smile':            'smile',
        'sad':              'sad',
        'sigh':             'sigh',
        'sorry':            'sorry',
        'suspicious':       'suspicious',
        'thinking':         'thinking',
        'unamused':         'unamused',
        'surprise':         'surprise',   'surprised':  'surprise',
        'sleeping':         'sleeping',
        # MRL aliases
        'contempt':         'happy',      'anxiety':    'angry',
        'disappointment':   'sad',        'frown':      'sad',
        'gasp':             'surprise',   'excited':    'surprise',
        'chuckle':          'smile',      'grin':       'smile',
        'helplessness':     'sorry',
    }

    # Arbitration (inmoov_msgs/JointCommand, priority 30): animated expressions hold
    # their joints briefly (blink can't cut in mid-animation); static holds take no
    # lease. The head tracker (eyes) and TTS (jaw) outrank us, so e.g. neutral()
    # at the end of speech no longer yanks the eyes away from a tracked face.
    _ANIM_LEASE_SEC = 1.0

    def __init__(self, node: Node):
        self._node = node
        self._pub = node.create_publisher(JointCommand, '/joint_cmd', 10)
        self._lease_sec = self._ANIM_LEASE_SEC

    # ── Internal ──────────────────────────────────────────────────────────────

    def _send(self, positions: dict[str, int]) -> None:
        """Publish an arbitrated command for given {joint: degrees} dict."""
        if not positions:
            return
        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = list(positions.keys())
        msg.position = [_r(v) for v in positions.values()]
        self._pub.publish(JointCommand(
            source='expression', priority=JointCommand.PRIORITY_EXPRESSION,
            lease_sec=self._lease_sec, cmd=msg))

    def _rest(self, joints: list[str] | None = None) -> None:
        if joints is None:
            joints = list(FACE_REST.keys())
        self._send({j: FACE_REST[j] for j in joints})

    def _expr(self, name: str, hold_sec: float = 0.0) -> None:
        """
        Send expression from EXPRESSIONS_DATA.
        Only sends joints defined in the expression (others keep current position).
        If hold_sec > 0, sleep then return to neutral.
        Contains time.sleep() when hold_sec > 0 — call from non-callback thread.
        """
        positions = EXPRESSIONS_DATA.get(name, {})
        self._send(positions)
        if hold_sec > 0:
            time.sleep(hold_sec)
            self.neutral()

    def hold(self, name: str) -> None:
        """Static, non-animated apply — no built-in revert timer. Used by
        /face_expression_hold (tts_node): the pose is meant to stay exactly
        until the caller (speech duration) explicitly changes/reverts it,
        unlike the one-shot animated methods below (happy()/surprise()/etc.)
        used by /face_expression."""
        self._lease_sec = 0.0   # static pose — write only, don't hold (serialised executor)
        try:
            self._hold(name)
        finally:
            self._lease_sec = self._ANIM_LEASE_SEC

    def _hold(self, name: str) -> None:
        if name == 'neutral':
            # EXPRESSIONS_DATA['neutral'] == {} → _expr('neutral') would send
            # an empty positions dict, which _send() no-ops on. Without this,
            # holding 'neutral' never actually moves the servos back to rest
            # — the previous expression's pose (e.g. happy) stays stuck
            # forever instead of relaxing when speech ends.
            self.neutral()
            return
        self._expr(name)

    # ── Expressions ───────────────────────────────────────────────────────────

    def neutral(self) -> None:
        """All face servos to rest positions."""
        self._rest()

    def angry(self) -> None:
        """MRL: forheadsU, eyelidsHalfShut, upperLipU, cheeksD, browsD"""
        self._expr('angry')

    def wink(self) -> None:
        """MRL: winkLeft — close left eyelids then restore.
        Contains time.sleep() — call from non-callback thread only."""
        self._expr('wink')
        time.sleep(0.3)
        self._rest(['eyelid_L_Upper', 'eyelid_L_Lower'])

    def disgust(self) -> None:
        """MRL: upperLipU, forheadsD, lowerEyelidsClose, cheekRightU, browsD"""
        self._expr('disgust')

    def fear(self) -> None:
        """MRL: eyelidsOpen, cheeksD, browsU, forheadsU"""
        self._expr('fear')

    def happy(self) -> None:
        """MRL: browsU, cheeksU, jaw open → relax → neutral.
        Animated. Contains time.sleep() — call from non-callback thread only."""
        self._expr('happy')
        time.sleep(1.0)
        self._rest(['eyebrow_L', 'eyebrow_R',
                    'eyelid_L_Upper', 'eyelid_L_Lower',
                    'eyelid_R_Upper', 'eyelid_R_Lower',
                    'cheek_L', 'cheek_R', 'jaw'])
        time.sleep(1.0)
        self.neutral()

    def smile(self) -> None:
        """MRL: cheeksU, lowerEyelidsClose, jaw → cheeksC → neutral.
        Animated. Contains time.sleep() — call from non-callback thread only."""
        self._expr('smile')
        time.sleep(1.0)
        self._send({'cheek_L': 90, 'cheek_R': 90,
                    'jaw': FACE_REST['jaw']})
        time.sleep(0.5)
        self.neutral()

    def sad(self) -> None:
        """MRL: upperEyelidsClose, cheeksD"""
        self._expr('sad')

    def sigh(self) -> None:
        """MRL: eyes roll up/right, then neutral.
        Contains time.sleep() — call from non-callback thread only."""
        self._expr('sigh')
        time.sleep(3.0)
        self._rest(['eye_lr_L', 'eye_ud_L', 'eye_lr_R', 'eye_ud_R'])
        self.neutral()

    def sorry(self) -> None:
        """MRL: browsU, cheeksD → wait → browsC, cheeksC → neutral.
        Animated. Contains time.sleep() — call from non-callback thread only."""
        self._expr('sorry')
        time.sleep(0.8)
        self._send({'eyebrow_L': FACE_REST['eyebrow_L'],
                    'eyebrow_R': FACE_REST['eyebrow_R'],
                    'cheek_L':   90, 'cheek_R': 90})
        time.sleep(0.5)
        self.neutral()

    def suspicious(self) -> None:
        """MRL: upperLipD, forheadsRULD, cheekLeftU, browsRULD,
        eyelidsLeftOpen, eyelidsRightHalfShut"""
        self._expr('suspicious')

    def thinking(self) -> None:
        """MRL: browsD, forheadsU, lowerEyelidsClose, eyes glance up.
        Animated. Contains time.sleep() — call from non-callback thread only."""
        self._expr('thinking')
        time.sleep(1.5)
        self._rest(['eye_lr_L', 'eye_ud_L', 'eye_lr_R', 'eye_ud_R'])
        self.neutral()

    def unamused(self) -> None:
        """MRL: browRightD, forheadsLURD, eyelidsHalfShut, eyesR, cheeksD"""
        self._expr('unamused')

    def surprise(self) -> None:
        """MRL: browsU, upperEyelidsOpen, forheadsD, jaw → relax.
        Animated. Contains time.sleep() — call from non-callback thread only."""
        self._expr('surprise')
        time.sleep(0.5)
        self._rest(['jaw'])
        self.neutral()

    def sleeping(self) -> None:
        """MRL: all eyelids closed."""
        self._expr('sleeping')

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def execute(self, expression: str) -> bool:
        """Execute expression by name. Returns False if unknown."""
        method_name = self.EXPRESSION_MAP.get(expression.lower())
        if not method_name:
            return False
        method = getattr(self, method_name, None)
        if not method:
            return False
        method()
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Standalone ROS2 node
# ─────────────────────────────────────────────────────────────────────────────

class FaceExpressionsNode(LifecycleNode):
    """
    Subscribes to /face_expression (std_msgs/String) and executes
    the named expression via FaceExpressions.

    Example:
      ros2 topic pub --once /face_expression std_msgs/msg/String "data: happy"
    """

    def __init__(self):
        super().__init__('face_expressions_node')
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._fe = None

    def on_configure(self, state):
        self._fe = FaceExpressions(self)
        self.create_subscription(String, '/face_expression', self._cb, 10)
        # Held expression for the duration of speech (tts_node) — a static
        # pose, no animation / auto-revert, see FaceExpressions.hold().
        self.create_subscription(String, '/face_expression_hold', self._cb_hold, 10)
        self.get_logger().info(
            f'face_expressions_node configured | calib: '
            f'{_CALIB_FILE if os.path.exists(_CALIB_FILE) else "defaults"}')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

    def _cb(self, msg: String) -> None:
        name = msg.data.strip().lower()
        if name not in self._fe.EXPRESSION_MAP:
            self.get_logger().warn(f'Unknown expression: "{name}"')
            return
        self.get_logger().info(f'face expression: {name}')
        self._executor.submit(self._fe.execute, name)

    def _cb_hold(self, msg: String) -> None:
        name = msg.data.strip().lower()
        if name not in EXPRESSIONS_DATA:
            self.get_logger().warn(f'Unknown hold expression: "{name}"')
            return
        self.get_logger().info(f'face expression (hold): {name}')
        # Same single-worker executor as /face_expression — serialises with
        # the animated gestures (greet/farewell), avoiding a servo race.
        self._executor.submit(self._fe.hold, name)


def main(args=None):
    rclpy.init(args=args)
    node = FaceExpressionsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
