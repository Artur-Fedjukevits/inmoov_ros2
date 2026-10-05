#!/usr/bin/env python3
"""
behavior_manager_node.py  (v2 — Data → Decision → Action)
==========================================================
A single "Eternal Tree" @ 10 Hz.
The Blackboard is the single source of truth.

World data flows from ROS topics → Blackboard → the BT makes decisions.

Blackboard keys (BehaviorManagerNode fills them in from subscriptions):
  /robot/sleep             bool  — active sleep mode
  /robot/sleep_requested   bool  — sleep transition requested (from LLM robot_control)
  /robot/sleep_text        str   — farewell phrase before sleeping
  /robot/command           dict  — pending physical command {action, ...} from LLM
  /llm/text                str   — LLM response text to speak ('' while streaming)
  /llm/voice_style         str   — voice instruction (CosyVoice instruct)
  /llm/emotion             str   — desired facial emotion while speaking
  /llm/has_content         bool  — True when there is content for DialogueBranch (text or streamed)
  /social/person_present   bool  — a person is in frame
  /social/name             str   — the person's name
  /social/emotion          str   — the person's current emotion
  /social/should_greet     bool  — a greeting is needed
  /social/greet_text       str   — greeting text
  /social/farewell_pending bool  — the person just left
  /social/farewell_text    str   — farewell text
  /social/introducing      bool  — name collection is in progress
  /social/introduce_pending bool — IdentityManager requests the introduction phrase be spoken
  /social/introduce_text   str   — text to speak during the introduction flow
  /social/face_search_pending bool — time to retry the face search (see FaceSearchAttempt)
  /search/query            str   — search query
  /search/result           dict  — search result

Tree (Selector, no-memory — re-evaluated from the top every tick):
  Root
  ├── SleepTransition  — if sleep was requested: farewell → sleep
  ├── SleepActive      — if already asleep: blocks everything below
  ├── RobotCommand     — pending physical command from LLM (move/arm/head)
  ├── WebSearch        — pending search query from LLM
  ├── SocialBranch     — if a person is in frame:
  │     IntroducingBlock / GreetBranch / FaceSearchBranch / DialogueBranch / IdleGaze
  │     (FaceSearchBranch — retries the face search by sound/voice hint on every
  │      utterance until head_tracker locks a face; lives here rather than in
  │      SoundScanBranch below because that branch is never ticked while a
  │      dialogue is in progress)
  ├── FarewellBranch   — if the person just left: farewell
  ├── SoundScanBranch  — wake word: turn the TORSO toward the voice (/sound_direction) until /human_detected
  ├── PIRScanBranch    — pure PIR motion (no voice): head left→right→center
  └── GlobalIdle       — nothing happening (blinking)

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import json
import math
import random
import threading
import time
import requests

import rclpy
from rclpy.node import Node
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import JointState
from inmoov_msgs.msg import JointCommand
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist

import py_trees

from inmoov_msgs.action import Speak
from inmoov_msgs.msg import SoundDirection


# ══════════════════════════════════════════════════════════════════════════════
# HELPER LEAVES
# ══════════════════════════════════════════════════════════════════════════════

class CheckBB(py_trees.behaviour.Behaviour):
    """Condition leaf: SUCCESS if check_fn(bb[key]) is truthy."""

    def __init__(self, name: str, key: str, check_fn=None):
        super().__init__(name)
        self._attr  = key.strip('/').replace('/', '.')
        self._check = check_fn if check_fn is not None else bool
        self._bb    = py_trees.blackboard.Client(name=f'Chk:{name}')
        self._bb.register_key(key=key, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        try:
            val = self._bb
            for part in self._attr.split('.'):
                val = getattr(val, part)
            ok = self._check(val)
        except KeyError:
            ok = False
        return (py_trees.common.Status.SUCCESS if ok
                else py_trees.common.Status.FAILURE)


class SetBB(py_trees.behaviour.Behaviour):
    """Action leaf: writes a value into the BB. Always SUCCESS."""

    def __init__(self, name: str, key: str, value):
        super().__init__(name)
        self._parts = key.strip('/').replace('/', '.').split('.')
        self._value = value
        self._bb    = py_trees.blackboard.Client(name=f'Set:{name}')
        self._bb.register_key(key=key, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        obj = self._bb
        for part in self._parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, self._parts[-1], self._value)
        return py_trees.common.Status.SUCCESS


class CheckFn(py_trees.behaviour.Behaviour):
    """Condition leaf: SUCCESS if fn() is truthy (for state that lives on the node, not in the BB)."""

    def __init__(self, name: str, fn):
        super().__init__(name)
        self._fn = fn

    def update(self) -> py_trees.common.Status:
        return (py_trees.common.Status.SUCCESS if self._fn()
                else py_trees.common.Status.FAILURE)


class AlwaysRunning(py_trees.behaviour.Behaviour):
    """Blocks the subtree — returns RUNNING forever."""

    def __init__(self, name: str = 'AlwaysRunning'):
        super().__init__(name)

    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.RUNNING


# ══════════════════════════════════════════════════════════════════════════════
# ACTIONS
# ══════════════════════════════════════════════════════════════════════════════

class ExecuteRobotCommand(py_trees.behaviour.Behaviour):
    """Reads /robot/command from the BB, publishes the physical command, clears the key."""

    # Absolute rest angles for the head (must match arduino_left_node and vision_head_tracker)
    _REST_ROTHEAD = 90.0
    _REST_NECK    = 40.0

    # Torso (midstom): rest=90, min=60, max=120 (arduino_left_node) — same sign
    # convention as rothead: LEFT=60/RIGHT=120 (verified by hand 2026-08-24).
    # scope='partial' adds part of
    # this range in the same direction as pan — used when the head alone isn't
    # enough to aim the camera at a conversation partner standing to the side.
    _REST_MIDSTOM           = 90.0
    _TORSO_MIN              = 60.0
    _TORSO_MAX              = 120.0
    _TORSO_HALF_RANGE       = 30.0   # _REST_MIDSTOM ± this = _TORSO_MIN/_TORSO_MAX
    _TORSO_PARTIAL_FRACTION = 0.3    # scope='partial'
    # scope='head' with a turn at least this wide still brings the torso along
    # (as 'partial') — the head and torso must never face different ways.
    # Smaller glances only re-center a torso that points the other way.
    _TORSO_FOLLOW_MIN_PAN   = 20.0

    # Arbitration (inmoov_msgs/JointCommand): the override holds the head for
    # 5 s + 1.5 s (see _send_head_cmd / _resume_head_tracker) — the lease covers
    # it with a margin; _enable_tracker_now releases explicitly.
    _SOURCE             = 'bt_command'
    _HEAD_JOINTS        = ('rothead', 'neck', 'midstom')
    _OVERRIDE_LEASE_SEC = 8.0

    def __init__(self, node: Node):
        super().__init__('ExecuteRobotCommand')
        self._node       = node
        self._pub_vel    = node.create_publisher(Twist,      'cmd_vel',          10)
        self._pub_arm    = node.create_publisher(String,     'arm_command',      10)
        self._pub_stat   = node.create_publisher(String,     'status_request',   10)
        self._timer      = None   # timer that stops motion
        self._head_timer = None   # timer that hands control back to head_tracker after a manual command
        self._torso_moved = False  # the last head command moved the torso — it needs to return to center

        # Track the current rothead/neck from ANY accepted command (/joint_commanded, including
        # from head_tracker itself while it's following a face) — so that before a
        # manual command (robot_control/look_direction) we know WHERE the face was,
        # and afterward can return the head there instead of to REST. Same pattern
        # as vision_head_tracker_node._external_joint_cb.
        self._known_rothead     = self._REST_ROTHEAD
        self._known_neck        = self._REST_NECK
        self._known_midstom     = self._REST_MIDSTOM
        self._pre_turn_rothead  = self._REST_ROTHEAD
        self._pre_turn_neck     = self._REST_NECK
        self._pre_turn_midstom  = self._REST_MIDSTOM
        self._override_active   = False   # True between a manual command and resume
        # Was a track actually locked BEFORE the manual command (not just "the
        # head was at rest/center") — see _resume_head_tracker.
        self._had_lock_before_turn = False
        # /joint_commanded = commands the Arduino nodes actually accepted (after arbitration)
        self._sub_joint = node.create_subscription(
            JointState, '/joint_commanded', self._track_joint_cb, 10)

        node.add_deactivate_hook(self.cancel_pending)

        self._bb = py_trees.blackboard.Client(name='ExecCmd')
        self._bb.register_key(key='/robot/command',
                               access=py_trees.common.Access.READ)
        self._bb.register_key(key='/robot/command',
                               access=py_trees.common.Access.WRITE)

    def initialise(self):
        cmd    = self._bb.robot.command
        action = cmd.get('action', '')

        if action == 'move':
            self._do_move(cmd)
        elif action == 'arm':
            msg = String()
            msg.data = json.dumps(
                {'command': cmd.get('command', 'home'), 'target': cmd.get('target', '')},
                ensure_ascii=False,
            )
            self._pub_arm.publish(msg)
            self._node.get_logger().info(f'Arm: {msg.data}')
        elif action == 'head':
            self._do_head(cmd)
        elif action == 'status':
            msg = String()
            msg.data = cmd.get('query', 'all')
            self._pub_stat.publish(msg)
        else:
            self._node.get_logger().warn(f'ExecuteRobotCommand: unknown action={action}')

        self._bb.robot.command = {}

    def _track_joint_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name == 'rothead':
                self._known_rothead = pos * 180.0 / math.pi + 90.0
            elif name == 'neck':
                self._known_neck    = pos * 180.0 / math.pi + 90.0
            elif name == 'midstom':
                self._known_midstom = pos * 180.0 / math.pi + 90.0

    def _torso_for_pan(self, pan: float, scope: str):
        """midstom for a head command, or None to leave the torso alone.
        The torso always ends up on the same side as the head (or centered):
        - scope partial/full, or 'head' with |pan| >= _TORSO_FOLLOW_MIN_PAN → turn it that way;
        - a small 'head' glance → only re-center a torso that points the other way;
        - never pull it back against the turn: if it is already turned further
          that way (e.g. by FaceSearch), keep it. Live bug 2026-09-30: torso at 120°
          after FaceSearch, "turn right" (pan=+45 partial) sent it back to 99°."""
        if pan == 0:
            return None
        rest    = self._REST_MIDSTOM
        current = self._known_midstom - rest
        if scope == 'head' and abs(pan) < self._TORSO_FOLLOW_MIN_PAN:
            return rest if current * pan < 0 else None
        fraction = 1.0 if scope == 'full' else self._TORSO_PARTIAL_FRACTION
        target = math.copysign(self._TORSO_HALF_RANGE * fraction, pan)
        if current * pan > 0 and abs(current) > abs(target):
            target = current
        return max(self._TORSO_MIN, min(self._TORSO_MAX, rest + target))

    def _do_head(self, cmd: dict):
        pan   = float(cmd.get('pan',  0))
        tilt  = float(cmd.get('tilt', 0))
        scope = str(cmd.get('scope') or 'head').strip().lower()
        if scope not in ('head', 'partial', 'full'):
            scope = 'head'

        if not self._override_active:
            # First manual command in a series — remember where the face was
            # BEFORE us, so that after resume we return the head there (not to
            # REST). If several commands come in a row (look right, then look
            # left) — don't overwrite the snapshot with an intermediate turned position.
            self._pre_turn_rothead = self._known_rothead
            self._pre_turn_neck    = self._known_neck
            self._pre_turn_midstom = self._known_midstom
            self._had_lock_before_turn = getattr(self._node, '_face_locked', False)
            self._override_active  = True

        rothead = max(30.0, min(140.0, self._REST_ROTHEAD + pan))
        neck    = max(0.0,  min(100.0, self._REST_NECK    + tilt))

        joints = {'rothead': rothead, 'neck': neck}

        midstom = self._torso_for_pan(pan, scope)
        if midstom is not None:
            joints['midstom'] = midstom
        # The torso has to go back to center on resume only if it is off center now
        self._torso_moved = midstom is not None and midstom != self._REST_MIDSTOM
        self._node.note_manual_turn()
        torso_log = f' midstom={midstom:.0f}°' if midstom is not None else ''
        self._node.get_logger().info(
            f'Head: pan={pan:+.0f}° tilt={tilt:+.0f}° scope={scope} → '
            f'rothead={rothead:.0f}° neck={neck:.0f}°{torso_log}')

        if self._head_timer:
            self._head_timer.cancel()
        # Command first, holding the head (priority 70) for the whole override;
        # the REST pose the tracker sends when disabled (priority 40, no lease) is
        # then rejected by the arbiter — no more "wait 200 ms and overwrite it".
        self._send_head_cmd(joints)
        self._node.enable_head_tracker(False)

    def cancel_pending(self):
        """Node deactivating: drop the head-command timer chain, stop any timed move."""
        if self._head_timer:
            self._head_timer.cancel()
            self._head_timer = None
        if self._override_active:
            self._node.release_joints(self._SOURCE, self._HEAD_JOINTS)
        self._override_active = False
        self._torso_moved     = False
        if self._timer:
            self._timer.cancel()
            self._timer = None
            self._pub_vel.publish(Twist())

    def _send_head_cmd(self, joints: dict):
        if 'midstom' in joints:
            self._node.note_own_torso_move()
        self._node.send_joints(self._SOURCE, JointCommand.PRIORITY_BT_COMMAND,
                               self._OVERRIDE_LEASE_SEC, joints)
        # After 5s hand control back to head_tracker (it will resume following the face)
        self._head_timer = threading.Timer(5.0, self._resume_head_tracker)
        self._head_timer.daemon = True
        self._head_timer.start()

    def _resume_head_tracker(self):
        # If a face was actually locked BEFORE the manual command (not just "the
        # head was at rest/center") — return the head to where the face was,
        # otherwise head_tracker will turn on wherever we pointed it, fail to
        # find a face there within return_timeout_sec, and drift back to REST
        # on its own, losing the person.
        #
        # BUT if there was no face BEFORE the command either (typical case:
        # "I'm to your left" / look_direction, where head_tracker just sat at
        # rest for the whole dialogue) — do NOT roll back: the manual command
        # just physically confirmed (e.g. via photo/description) that the
        # person is EXACTLY at the current turned position. Rolling back to
        # the old position (in practice — to REST) would undo that discovery
        # and reliably lose the person again. Live bug 2026-08-31 (found by
        # the user): "looked left, took a photo, described it — and then
        # immediately turned back."
        if not self._node.lc_active:
            return
        if self._had_lock_before_turn:
            back = {'rothead': self._pre_turn_rothead, 'neck': self._pre_turn_neck}
            if self._torso_moved:
                # The torso was moved by a manual command (see _torso_for_pan) —
                # head_tracker only controls the head, so the torso won't return
                # on its own. Back to where it was with the face, not to center —
                # otherwise head and torso would end up facing different ways.
                back['midstom'] = self._pre_turn_midstom
                self._node.note_own_torso_move()
            # Hold it until _enable_tracker_now releases (1.5 s below)
            self._node.send_joints(self._SOURCE, JointCommand.PRIORITY_BT_COMMAND, 3.0, back)
            self._node.get_logger().info(
                f'Head: returning to pre-command position rothead={self._pre_turn_rothead:.0f}° '
                f'neck={self._pre_turn_neck:.0f}°')
        else:
            self._node.get_logger().info(
                'Head: there was no face before the command either — staying at the '
                'current (just-confirmed) position, not rolling back')
        self._torso_moved      = False
        self._override_active  = False

        # Give the servos time to physically finish turning (the return trip can
        # be up to ~70°, the same order of magnitude as the original turn), then
        # enable head_tracker — it picks up tracking from the correct position
        # (or, if the person really did leave, it will calmly drift to REST on
        # its own timeout).
        self._head_timer = threading.Timer(1.5, self._enable_tracker_now)
        self._head_timer.daemon = True
        self._head_timer.start()

    def _enable_tracker_now(self):
        if not self._node.lc_active:
            return
        self._node.release_joints(self._SOURCE, self._HEAD_JOINTS)
        self._node.enable_head_tracker(True)
        self._head_timer = None
        self._node.get_logger().info('Head: control returned to head_tracker')

    def _do_move(self, cmd: dict):
        direction = cmd.get('direction', 'stop')
        speed     = float(cmd.get('speed', 0.3))
        duration  = float(cmd.get('duration', 2.0))
        twist     = Twist()
        mapping   = {
            'forward':  ('linear.x',   speed),
            'backward': ('linear.x',  -speed),
            'left':     ('angular.z',  speed),
            'right':    ('angular.z', -speed),
        }
        if direction in mapping:
            attr, val = mapping[direction]
            obj, field = attr.split('.')
            setattr(getattr(twist, obj), field, val)
        self._pub_vel.publish(twist)
        self._node.get_logger().info(f'Move: {direction} speed={speed}')
        if direction != 'stop' and duration > 0:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(duration, self._stop_vel)
            self._timer.start()

    def _stop_vel(self):
        self._pub_vel.publish(Twist())
        self._timer = None

    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        if self._timer:
            self._timer.cancel()
            self._timer = None


class ExpressEmotion(py_trees.behaviour.Behaviour):
    """Publishes an emotion to /face_expression. A fixed string or from the BB."""

    VALID = {
        'neutral', 'angry', 'wink', 'disgust', 'fear', 'happy', 'smile',
        'sad', 'sigh', 'sorry', 'suspicious', 'thinking', 'unamused',
        'surprise', 'sleeping', 'anger', 'surprised', 'contempt',
        'anxiety', 'disappointment', 'frown', 'gasp', 'excited',
        'chuckle', 'grin', 'helplessness',
    }

    def __init__(self, node: Node, emotion: str | None = None,
                 bb_key: str | None = None):
        super().__init__(f'ExpressEmotion({emotion or bb_key})')
        self._node    = node
        self._emotion = emotion
        self._pub     = node.create_publisher(String, '/face_expression', 10)
        self._bb      = None
        if bb_key:
            self._bb      = py_trees.blackboard.Client(name=f'Expr:{bb_key}')
            self._bb_attr = bb_key.strip('/').replace('/', '.')
            self._bb.register_key(key=bb_key, access=py_trees.common.Access.READ)

    def initialise(self):
        if self._bb is not None:
            try:
                val = self._bb
                for part in self._bb_attr.split('.'):
                    val = getattr(val, part)
                emotion = str(val).lower().strip()
            except Exception:
                emotion = 'neutral'
        else:
            emotion = (self._emotion or 'neutral').lower()

        if emotion not in self.VALID:
            emotion = 'neutral'
        msg = String()
        msg.data = emotion
        self._pub.publish(msg)
        self._node.get_logger().info(f'Face emotion: {emotion}')

    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS


class SpeakBehaviour(py_trees.behaviour.Behaviour):
    """
    TTS leaf: RUNNING while speech is in progress.
    Supports preemption via terminate().
    Text is either a fixed string or comes from the BB (bb_key).
    """

    def __init__(self, node: Node, text: str | None = None,
                 bb_key: str | None = None, voice_bb_key: str | None = None,
                 voice: str | None = None):
        super().__init__('Speak')
        self._node          = node
        self._text          = text
        self._bb_key        = bb_key
        self._voice_bb_key  = voice_bb_key
        self._voice         = voice
        self._client        = ActionClient(node, Speak, 'speak')
        self._goal_handle   = None
        self._done          = threading.Event()
        self._success       = False
        self._bb            = None
        if bb_key or voice_bb_key:
            self._bb = py_trees.blackboard.Client(name=f'Speak:{bb_key or voice_bb_key}')
            if bb_key:
                self._bb_attr = bb_key.strip('/').replace('/', '.')
                self._bb.register_key(key=bb_key, access=py_trees.common.Access.READ)
            if voice_bb_key:
                self._voice_bb_attr = voice_bb_key.strip('/').replace('/', '.')
                self._bb.register_key(key=voice_bb_key, access=py_trees.common.Access.READ)

    def initialise(self):
        self._done.clear()
        self._success     = False
        self._goal_handle = None

        text  = self._text
        voice = self._voice or ''
        if self._bb is not None:
            try:
                if self._bb_key:
                    val = self._bb
                    for part in self._bb_attr.split('.'):
                        val = getattr(val, part)
                    text = str(val)
            except Exception as e:
                self._node.get_logger().error(f'SpeakBehaviour: BB text error: {e}')
                self._done.set()
                return
            try:
                if self._voice_bb_key:
                    val = self._bb
                    for part in self._voice_bb_attr.split('.'):
                        val = getattr(val, part)
                    voice = str(val)
            except Exception:
                voice = ''

        if not text or not text.strip():
            self._success = True
            self._done.set()
            return

        if not self._client.wait_for_server(timeout_sec=1.0):
            self._node.get_logger().error('SpeakBehaviour: TTS Action Server unavailable')
            self._done.set()
            return

        goal = Speak.Goal()
        goal.text  = text
        goal.voice = voice
        future = self._client.send_goal_async(goal)
        future.add_done_callback(self._goal_accepted_cb)

    def _goal_accepted_cb(self, future):
        gh = future.result()
        if not gh.accepted:
            self._node.get_logger().warn('SpeakBehaviour: goal rejected')
            self._done.set()
            return
        self._goal_handle = gh
        gh.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future):
        self._success = future.result().result.success
        self._done.set()

    def update(self) -> py_trees.common.Status:
        if not self._done.is_set():
            return py_trees.common.Status.RUNNING
        return (py_trees.common.Status.SUCCESS if self._success
                else py_trees.common.Status.FAILURE)

    def terminate(self, new_status: py_trees.common.Status):
        if (new_status == py_trees.common.Status.INVALID
                and self._goal_handle is not None):
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None


class SetSleepMode(py_trees.behaviour.Behaviour):
    """Activates sleep mode: publishes /robot_sleep True (latched)."""

    def __init__(self, node: Node):
        super().__init__('SetSleepMode')
        self._node = node
        lqos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._pub = node.create_publisher(Bool, '/robot_sleep', lqos)
        self._bb  = py_trees.blackboard.Client(name='SetSleep')
        self._bb.register_key(key='/robot/sleep',
                               access=py_trees.common.Access.WRITE)
        self._bb.register_key(key='/robot/sleep_requested',
                               access=py_trees.common.Access.WRITE)

    def initialise(self):
        self._node.enable_face_detection(False)
        self._node.enable_head_tracker(False)
        msg = Bool()
        msg.data = True
        self._pub.publish(msg)
        self._bb.robot.sleep           = True
        self._bb.robot.sleep_requested = False
        self._node.get_logger().info('Sleep mode activated — face_detection and head_tracker disabled')

    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS


class WebSearchBehaviour(py_trees.behaviour.Behaviour):
    """Asynchronous search via Tavily. RUNNING until it completes."""

    def __init__(self, node: Node, api_key: str):
        super().__init__('WebSearch')
        self._node       = node
        self._api_key    = api_key
        self._result_pub = node.create_publisher(String, 'search_result', 10)
        self._bb         = py_trees.blackboard.Client(name='WebSearch')
        self._bb.register_key(key='/search/query',
                               access=py_trees.common.Access.READ)
        self._bb.register_key(key='/search/query',
                               access=py_trees.common.Access.WRITE)
        self._bb.register_key(key='/search/result',
                               access=py_trees.common.Access.WRITE)
        self._done    = threading.Event()
        self._snippet = ''
        self._success = False

    def initialise(self):
        self._done.clear()
        self._snippet = ''
        self._success = False
        query = self._bb.search.query
        self._bb.search.query = ''  # clear immediately — prevents a repeat trigger
        threading.Thread(target=self._search, args=(query,), daemon=True).start()

    def _search(self, query: str):
        try:
            self._node.get_logger().info(f'Search (Tavily): "{query}"')
            r = requests.post(
                'https://api.tavily.com/search',
                json={
                    'api_key':        self._api_key,
                    'query':          query,
                    'search_depth':   'basic',
                    'max_results':    3,
                    'include_answer': True,
                },
                timeout=20.0,
            )
            r.raise_for_status()
            data    = r.json()
            answer  = data.get('answer', '')
            results = data.get('results', [])
            self._snippet = (
                answer.strip() if answer else
                (results[0]['content'].strip()[:500] if results else 'Ничего не найдено')
            )
            self._success = True
            # Publish immediately from this thread — don't wait for the next BT tick
            self._bb.search.result = {'snippet': self._snippet}
            msg = String()
            msg.data = self._snippet
            self._result_pub.publish(msg)
            self._node.get_logger().info(
                f'Search completed: {len(self._snippet)} characters → published')
        except Exception as e:
            self._node.get_logger().error(f'Tavily search error: {e}')
        finally:
            self._done.set()

    def update(self) -> py_trees.common.Status:
        if not self._done.is_set():
            return py_trees.common.Status.RUNNING
        return py_trees.common.Status.SUCCESS if self._success else py_trees.common.Status.FAILURE


class GesticulationAction(py_trees.behaviour.Behaviour):
    """A gesture run in parallel with speech. Publishes arm_command based on emotion/context.

    Currently implemented as a stub — always SUCCESS.
    Future version: analyze /llm/emotion and pick a matching gesture.
    """

    def __init__(self, node: Node):
        super().__init__('Gesticulate')
        self._node    = node
        self._pub_arm = node.create_publisher(String, 'arm_command', 10)
        self._bb      = py_trees.blackboard.Client(name='Gesticulate')
        self._bb.register_key(key='/llm/emotion', access=py_trees.common.Access.READ)

    def initialise(self):
        try:
            emotion = self._bb.llm.emotion
        except KeyError:
            emotion = 'neutral'
        # TODO: map emotion → a specific gesture (grab/extend/wave/etc.)
        # For now — a neutral "home" pose
        if emotion in ('happy', 'excited', 'smile'):
            cmd = {'command': 'wave', 'target': ''}
        else:
            cmd = {'command': 'home', 'target': ''}
        msg = String()
        msg.data = json.dumps(cmd, ensure_ascii=False)
        self._pub_arm.publish(msg)

    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS


class PIRScanBehaviour(py_trees.behaviour.Behaviour):
    """
    Turns the head left→right→center when the PIR sensor triggers.
    The BT is the sole orchestrator: it enables face_detection for the search.

    If a face is found — SocialBranch (higher priority in the Selector) interrupts
    the scan via terminate(INVALID), which enables head_tracker (hands it control).
    If no face is found — _done() disables face_detection and returns SUCCESS.
    """

    # rothead: rest=90, min=30, max=140 (arduino); rest_rothead=90 (vision_head_tracker)
    # IMPORTANT 2026-08-24: the old LEFT=120/RIGHT=60 note was WRONG (never
    # properly verified physically) — verified by hand: rothead=120 physically
    # turns RIGHT, the same convention as midstom (LEFT=60/RIGHT=120).
    _CENTER   = 90.0   # looking forward (hardware rest arduino_left_node: rothead rest=90)
    _LEFT     = 60.0   # full left (verified by hand 2026-08-24)
    _RIGHT    = 120.0  # full right (verified by hand 2026-08-24)
    _NECK     = 40.0   # neck hardware rest (arduino_left_node: neck rest=40)

    # vel=1.0 rad/s → deg_per_sec_to_step(57°/s) = step=3 → ~50°/s
    # CENTER(90)→LEFT(60): 30°/50°/s≈0.6s, LEFT→RIGHT(120): 60°/50°/s≈1.2s, RIGHT→CENTER: 30°/50°/s≈0.6s
    _SCAN_VEL = 1.0

    # (name, target_rothead | None=dwell, duration_sec)
    _PHASES = (
        ('go_left',      _LEFT,   0.6),
        ('dwell_left',   None,    1.5),   # was 0.8
        ('go_right',     _RIGHT,  1.2),
        ('dwell_right',  None,    1.5),   # was 0.8
        ('go_center',    _CENTER, 0.6),
        ('dwell_center', None,    1.2),   # new: dwell at center
    )

    def __init__(self, node: Node):
        super().__init__('PIRScan')
        self._node      = node
        self._bb = py_trees.blackboard.Client(name='PIRScan')
        self._bb.register_key(key='/pir/scan_active',
                               access=py_trees.common.Access.WRITE)
        self._phase      = 0
        self._phase_end  = 0.0
        self._completed  = False   # True once _done() has been called (scan finished without a face)

    def _head_cmd(self, rothead: float) -> None:
        # Lease covers the longest go+dwell phase pair; released when the scan ends
        self._node.send_joints('bt_scan', JointCommand.PRIORITY_BT_SCAN, 3.0,
                               {'rothead': rothead, 'neck': self._NECK}, vel=self._SCAN_VEL)

    def initialise(self) -> None:
        self._phase     = 0
        self._completed = False
        _, target, dur  = self._PHASES[0]
        self._phase_end = time.monotonic() + dur
        self._node.enable_face_detection(True)
        self._head_cmd(target)
        self._node.get_logger().info(
            f'PIRScan: starting — turning left (rothead={target:.0f}°), face_detection enabled')

    def update(self) -> py_trees.common.Status:
        if time.monotonic() < self._phase_end:
            return py_trees.common.Status.RUNNING

        self._phase += 1
        if self._phase >= len(self._PHASES):
            return self._done()

        name, target, dur = self._PHASES[self._phase]
        self._phase_end = time.monotonic() + dur
        if target is not None:
            self._head_cmd(target)
            self._node.get_logger().info(f'PIRScan: {name} (rothead={target:.0f}°)')

        return py_trees.common.Status.RUNNING

    def _done(self) -> py_trees.common.Status:
        """Scan finished — no face found. Disable detection."""
        self._completed = True
        self._node.release_joints('bt_scan', ('rothead', 'neck'))
        self._node.enable_face_detection(False)
        self._bb.pir.scan_active = False
        self._node.get_logger().info('PIRScan: finished — no face detected, face_detection disabled')
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status == py_trees.common.Status.INVALID:
            self._bb.pir.scan_active = False
            if self._completed:
                # py_trees calls terminate(INVALID) when a Sequence(memory=False)
                # re-ticks right after our SUCCESS — ignore it, the scan already
                # finished normally.
                return
            # A real preempt: SocialBranch has taken over control (a face was found),
            # or the NoActiveDialogue gate closed (a dialogue is in progress).
            # Hand the head to vision_head_tracker, leave face_detection enabled.
            self._node.release_joints('bt_scan', ('rothead', 'neck'))
            self._node.enable_head_tracker(True)
            self._node.get_logger().info(
                'PIRScan: interrupted (face found / dialogue) — head_tracker enabled')


class SoundScanBehaviour(py_trees.behaviour.Behaviour):
    """
    On wake word — turn the TORSO (midstom), not the head, toward wherever
    the voice came from (per /sound_direction — TDOA sign-vote, see
    sound_localization_node).
    Replaces the old PIRScan head-scan specifically for the "heard a voice"
    case (PIR motion without voice still uses PIRScanBehaviour — there is no
    direction to work with there).

    Stops EXACTLY where it is as soon as OAK-D has seen a person
    (/human_detected — faster and coarser than full face recognition
    /social/person_present) — the torso does NOT return to center, and
    control is handed to head_tracker (which reads face_tracker) for precise
    visual fine-tuning. If no person is found within DWELL_TIMEOUT — the
    torso returns to center, face_detection is disabled, failure (same as
    the old PIRScan).

    The midstom sign was verified by hand 2026-08-22: 60°=left, 120°=right —
    the OPPOSITE convention from rothead (there 120=left, 60=right).
    Don't mix these up in future edits.
    """

    _CENTER = 90.0   # midstom rest (arduino_left_node: midstom rest=90, min=60, max=120)
    _LEFT   = 60.0    # verified by hand 2026-08-22: midstom=60° — torso fully LEFT
    _RIGHT  = 120.0   # verified by hand 2026-08-22: midstom=120° — torso fully RIGHT (opposite convention from rothead!)
    _TURN_VEL = 1.0

    _GO_DURATION    = 0.6   # ~30° at ~50°/s (see the _SCAN_VEL calculation in PIRScanBehaviour)
    _DWELL_TIMEOUT  = 8.0   # how long to wait for /human_detected after turning before giving up
    _MIN_CONFIDENCE = 0.15  # below this the direction is "unknown", don't turn (stay centered)
    _MIN_ANGLE_DEG  = 15.0  # |angle_deg| smaller than this also counts as "near center", don't turn

    # rothead: rest=90, min=30, max=140 — same side convention as midstom:
    # >90 = right (aim_head_at_human: OAK-D angle +28° → rothead 129°; pan>0 = right)
    _ROTHEAD_CENTER = 90.0
    _ROTHEAD_MIN    = 30.0
    _ROTHEAD_MAX    = 140.0
    _NECK_REST      = 40.0
    # The head turns the same way as the torso on a sound/voice-hint turn
    # (see BehaviorManagerNode.turn_toward) — torso 30° + head 25° ≈ 55° off-axis.
    _HEAD_TURN_OFFSET = 25.0
    _JOINTS = ('midstom', 'rothead', 'neck')
    _AIM_GAIN       = 1.4  # 2026-08-24: 1:1 undershot — the head caught the face for a
                            # moment at the edge of the frame and immediately lost it
                            # (OAK-D is physically offset from the eye axis, parallax
                            # requires a bit of angle overshoot). Tuned empirically, not
                            # from geometry — refine based on real-world results.

    def __init__(self, node: Node):
        super().__init__('SoundScan')
        self._node      = node
        self._bb = py_trees.blackboard.Client(name='SoundScan')
        self._bb.register_key(key='/sound/scan_active',
                               access=py_trees.common.Access.WRITE)
        self._phase      = 0     # 0=moving to target, 1=waiting for human_detected, 2=returning to center
        self._phase_end  = 0.0
        self._completed  = False
        self._target     = self._CENTER

    def _turn_cmd(self, midstom: float) -> None:
        # Lease covers go + the whole dwell; released on found / done / preempt
        self._node.turn_toward(midstom, 'bt_scan', JointCommand.PRIORITY_BT_SCAN,
                               self._GO_DURATION + self._DWELL_TIMEOUT + 1.0,
                               vel=self._TURN_VEL)

    def initialise(self) -> None:
        self._completed = False
        self._phase      = 0
        self._started_mono = time.monotonic()
        self._moved      = True

        angle      = getattr(self._node, '_last_sound_angle', 0.0)
        confidence = getattr(self._node, '_last_sound_confidence', 0.0)

        if self._node.manual_turn_recent():
            # The user has just turned us explicitly — don't override it with a guess
            self._target = self._CENTER
            self._moved  = False
            self._node.get_logger().info(
                'SoundScan: a manual turn command was just executed — not turning')
        elif confidence < self._MIN_CONFIDENCE or abs(angle) < self._MIN_ANGLE_DEG:
            self._target = self._CENTER
            self._node.get_logger().info(
                f'SoundScan: direction is unconfident (angle={angle:.0f}° '
                f'conf={confidence:.2f}) — staying centered')
        elif angle > 0:
            self._target = self._RIGHT
            self._node.get_logger().info(
                f'SoundScan: voice on the right (angle={angle:.0f}° conf={confidence:.2f}) '
                f'— turning torso right')
        else:
            self._target = self._LEFT
            self._node.get_logger().info(
                f'SoundScan: voice on the left (angle={angle:.0f}° conf={confidence:.2f}) '
                f'— turning torso left')

        self._node.enable_face_detection(True)
        if self._moved:
            self._turn_cmd(self._target)
        self._phase_end = time.monotonic() + self._GO_DURATION

    def update(self) -> py_trees.common.Status:
        if getattr(self._node, '_human_detected', False):
            return self._found()

        now = time.monotonic()
        if self._phase == 0:
            if now < self._phase_end:
                return py_trees.common.Status.RUNNING
            self._phase     = 1
            self._phase_end = now + self._DWELL_TIMEOUT
            self._node.get_logger().info('SoundScan: holding position, waiting for /human_detected')
            return py_trees.common.Status.RUNNING
        elif self._phase == 1:
            if now < self._phase_end:
                return py_trees.common.Status.RUNNING
            if self._target == self._CENTER:
                # Already standing at center (the direction was unconfident from the very
                # start, we never turned anywhere) — nothing to return, an extra
                # _torso_cmd and a confusing "return to center" log. Live bug
                # 2026-08-28 (noticed by the user).
                return self._done()
            if self._node._last_manual_turn_mono > self._started_mono:
                # A manual turn command arrived during the dwell — leave the robot
                # where the user pointed it, don't swing back to center.
                return self._done()
            self._phase = 2
            self._turn_cmd(self._CENTER)
            self._phase_end = now + self._GO_DURATION
            self._node.get_logger().info('SoundScan: no person found before the timeout — returning to center')
            return py_trees.common.Status.RUNNING
        else:
            if now < self._phase_end:
                return py_trees.common.Status.RUNNING
            return self._done()

    def _found(self) -> py_trees.common.Status:
        self._completed = True
        self._node.release_joints('bt_scan', self._JOINTS)
        self._bb.sound.scan_active = False
        self._node.aim_head_at_human()   # aim the head using OAK-D right away, don't wait for face_detection to find it
        self._node.enable_head_tracker(True)
        self._node.get_logger().info(
            'SoundScan: OAK-D saw a person — stopping, handing control to head_tracker')
        return py_trees.common.Status.SUCCESS

    def _done(self) -> py_trees.common.Status:
        self._completed = True
        self._node.release_joints('bt_scan', self._JOINTS)
        self._node.enable_face_detection(False)
        self._bb.sound.scan_active = False
        # First failure of the session (right after the wake word) — count it as
        # attempt #1 in the shared face-search retry counter, so that the "second
        # time" (as the user phrased it) naturally coincides with the first attempt
        # during the dialogue itself (see FaceSearchAttempt/record_face_search_attempt).
        self._node._face_search_attempts = 1
        self._node._face_search_pending_check = {
            'direction': None if self._target == self._CENTER else self._target
        }
        self._node._publish_face_search_status(active=True, attempts=1, ask_now=False)
        self._node.get_logger().info('SoundScan: finished — no person found, face_detection disabled')
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status == py_trees.common.Status.INVALID:
            self._bb.sound.scan_active = False
            if self._completed:
                # py_trees calls terminate(INVALID) when Sequence(memory=False) re-ticks
                # right after our SUCCESS — ignore it, already handled in _found()/_done().
                return
            # A real preempt (e.g. /social/person_present became True some other way,
            # SocialBranch intercepted before we ourselves saw /human_detected).
            self._node.release_joints('bt_scan', self._JOINTS)
            self._node.enable_head_tracker(True)
            self._node.get_logger().info('SoundScan: interrupted (person found) — head_tracker enabled')


class FaceSearchAttempt(py_trees.behaviour.Behaviour):
    """One quick NON-BLOCKING torso turn per dialogue utterance, until
    head_tracker locks a face (see project memory: face search retry).

    Unlike SoundScanBehaviour (a one-shot scan right after the wake word,
    which stops being ticked as soon as a dialogue starts — SocialBranch
    has higher priority than SoundScanBranch) — this leaf lives INSIDE
    social_selector, so it keeps getting a chance on every utterance
    for as long as the conversation goes on.

    Does NOT wait for /human_detected (no dwell) — returns SUCCESS in the same tick,
    so as not to delay speech by more than one BT tick (~100ms @ 10Hz).
    Direction priority: the interlocutor's voice hint (/voice/direction_hint,
    parsed in llm_node) > /sound_direction > nothing (just wait for the next
    utterance). Delegates the direction choice / counter / status logic to the
    node — see BehaviorManagerNode.record_face_search_attempt().
    """

    def __init__(self, node: Node):
        super().__init__('FaceSearchAttempt')
        self._node = node

    def update(self) -> py_trees.common.Status:
        node = self._node
        now = time.monotonic()
        if now - node._face_search_last_attempt_mono < node._FACE_SEARCH_MIN_RETRY_INTERVAL_SEC:
            return py_trees.common.Status.SUCCESS  # too soon after the previous turn — skip

        target, source = node.record_face_search_attempt()
        node.enable_face_detection(True)
        if target is not None:
            # Short lease: the turn itself (~0.6 s) with a margin; no release needed.
            # head_tracker keeps the head there afterwards (it syncs to /joint_commanded).
            rothead = node.turn_toward(target, 'bt_scan', JointCommand.PRIORITY_BT_SCAN,
                                       1.5, vel=1.0)
            node.get_logger().info(
                f'FaceSearch: attempt #{node._face_search_attempts} — '
                f'{source} → midstom={target:.0f}° rothead={rothead:.0f}°')
        elif source == 'manual_command':
            node.get_logger().info(
                f'FaceSearch: attempt #{node._face_search_attempts} — '
                f'no turn, a manual turn command was just executed')
        else:
            node.get_logger().info(
                f'FaceSearch: attempt #{node._face_search_attempts} — '
                f'no direction, waiting for the next utterance')
        return py_trees.common.Status.SUCCESS


class IdleBlinkBehaviour(py_trees.behaviour.Behaviour):
    """Eye blinking in the idle state. Always RUNNING."""

    _CLOSED = {
        'eyelid_L_Upper': 70, 'eyelid_L_Lower': 75,
        'eyelid_R_Upper': 65, 'eyelid_R_Lower': 70,
    }
    _OPEN = {
        'eyelid_L_Upper': 85, 'eyelid_L_Lower': 85,
        'eyelid_R_Upper': 85, 'eyelid_R_Lower': 85,
    }
    # Firmware: step=2, SMOOTH_INTERVAL_MS=60 → default 33°/s (too slow for blink).
    # We send vel=3.0 rad/s → deg_per_sec_to_step(171°/s) = step=10 → ~170°/s.
    # Worst-case travel: 20° (R_Upper 85→65) / 170°/s ≈ 120ms to fully close.
    _BLINK_VEL_RADS = 3.0   # rad/s → firmware step≈10 via CMD_SET_SPEEDS
    _BLINK_DURATION = 0.12  # seconds — matches 20° travel at step=10 (2 firmware ticks)
    _INTERVAL_MIN   = 3.0   # min pause between blinks
    _INTERVAL_MAX   = 7.0   # max pause between blinks

    def __init__(self, node: Node, name: str = 'IdleBlink'):
        super().__init__(name)
        self._node       = node
        self._next_blink = time.monotonic() + random.uniform(self._INTERVAL_MIN, self._INTERVAL_MAX)
        self._open_at    = None

    def _send(self, positions: dict, vel: float = 0.0) -> None:
        # Lowest priority, no lease: an expression holding the eyelids wins
        self._node.send_joints('blink', JointCommand.PRIORITY_BLINK, 0.0, positions, vel=vel)
        # The eyelids pass in front of the eye cameras — face_capture drops those
        # frames (/eyes/blink: True on close, False on reopen)
        self._node.publish_blink(positions is self._CLOSED)

    def initialise(self) -> None:
        self._next_blink = time.monotonic() + random.uniform(self._INTERVAL_MIN, self._INTERVAL_MAX)
        self._open_at    = None

    def update(self) -> py_trees.common.Status:
        now = time.monotonic()
        if self._open_at is not None:
            if now >= self._open_at:
                self._send(self._OPEN, vel=self._BLINK_VEL_RADS)
                self._open_at    = None
                self._next_blink = now + random.uniform(self._INTERVAL_MIN, self._INTERVAL_MAX)
        elif now >= self._next_blink:
            self._send(self._CLOSED, vel=self._BLINK_VEL_RADS)
            self._open_at = now + self._BLINK_DURATION
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status == py_trees.common.Status.INVALID and self._open_at is not None:
            self._send(self._OPEN, vel=self._BLINK_VEL_RADS)
            self._open_at    = None
            self._next_blink = time.monotonic() + random.uniform(self._INTERVAL_MIN, self._INTERVAL_MAX)


# ══════════════════════════════════════════════════════════════════════════════
# BUILDING THE ETERNAL TREE
# ══════════════════════════════════════════════════════════════════════════════

def build_tree(node: Node, tavily_key: str) -> py_trees.behaviour.Behaviour:
    """
    Builds the single eternal priority tree.

    Root (Selector, no-memory) — re-checked from the start on every tick:
      1. SleepTransition  — transition to sleep on request
      2. SleepActive      — block everything while asleep
      3. RobotCommand     — execute a physical command from the LLM
      4. WebSearch        — internet search
      5. SocialBranch     — social interaction (gate: person_present)
      6. FarewellBranch   — farewell when the person has left
      7a. SoundScanBranch — wake word: turn the torso (midstom) toward the voice until /human_detected
      7b. PIRScanBranch   — pure PIR motion (no voice): head scan
      8. GlobalIdle       — idle

    Interrupt Buffer: SocialBranch — Sequence(no-memory):
      CheckPersonPresent → FAILURE if the person left → Sequence is interrupted
      → terminate(INVALID) is called on SpeakBehaviour → cancels the TTS goal.
    """

    # ── 1. Sleep transition ───────────────────────────────────────────────
    sleep_transition = py_trees.composites.Sequence(
        'SleepTransition', memory=False, children=[
            CheckBB('IsSleepRequested', '/robot/sleep_requested',
                    check_fn=lambda v: v is True),
            py_trees.composites.Sequence('FarewellAndSleep', memory=True, children=[
                ExpressEmotion(node, emotion='sleeping'),
                SpeakBehaviour(node, bb_key='/robot/sleep_text'),
                SetSleepMode(node),
            ]),
        ]
    )

    # ── 2. Sleep active block ─────────────────────────────────────────────
    sleep_active = py_trees.composites.Sequence(
        'SleepActive', memory=False, children=[
            CheckBB('IsAsleep', '/robot/sleep', check_fn=lambda v: v is True),
            AlwaysRunning('SleepBlock'),
        ]
    )

    # ── 3. Robot command (move/arm/head/status) ───────────────────────────
    robot_command = py_trees.composites.Sequence(
        'RobotCommand', memory=False, children=[
            CheckBB('HasCommand', '/robot/command',
                    check_fn=lambda v: bool(v) and bool(v.get('action'))),
            ExecuteRobotCommand(node),
        ]
    )

    # ── 4. Web search ─────────────────────────────────────────────────────
    web_search = py_trees.composites.Sequence(
        'WebSearch', memory=False, children=[
            CheckBB('HasQuery', '/search/query', check_fn=lambda v: bool(v)),
            WebSearchBehaviour(node, tavily_key),
        ]
    )

    # ── 5. Social branch ──────────────────────────────────────────────────
    #
    # Structure of meeting a new person:
    #   IntroducingBlock (no-memory Sequence):
    #     - Checks the introducing flag (name-collection mode)
    #     - SpeakOrSkip: if there is a pending phrase — speaks it (memory=True inner),
    #       otherwise skips (Success fallback)
    #     - AlwaysRunning: blocks while introducing=True
    #
    # When introducing becomes False → IsIntroducing FAILURE → Sequence FAILURE
    # → SocialSelector moves on to the next branch.

    speak_if_pending = py_trees.composites.Sequence(
        'SpeakIntroIfPending', memory=True, children=[
            CheckBB('HasIntroText', '/social/introduce_pending',
                    check_fn=lambda v: v is True),
            SetBB('ClearIntroPending', '/social/introduce_pending', False),
            SpeakBehaviour(node, bb_key='/social/introduce_text'),
        ]
    )
    speak_or_skip = py_trees.composites.Selector(
        'SpeakOrSkip', memory=False, children=[
            speak_if_pending,
            py_trees.behaviours.Success(name='NoPendingIntro'),
        ]
    )
    introducing_block = py_trees.composites.Sequence(
        'IntroducingBlock', memory=False, children=[
            CheckBB('IsIntroducing', '/social/introducing',
                    check_fn=lambda v: v is True),
            speak_or_skip,
            AlwaysRunning('WaitIntroduce'),
        ]
    )

    # Greeting: only once (should_greet → True → BT greets → clears the flag)
    # Facial expression — via /face_expression_hold (tts_node), not ExpressEmotion/BT:
    # previously a parallel ExpressEmotion('happy') triggered the animated happy() with
    # its own hold_sec=1s and rolled the face back to rest before the end of the
    # greeting phrase (5-7s), see dialogue_branch below — the same bug that had
    # already been fixed there; it was missed here during the e0ddd51 migration.
    greet_branch = py_trees.composites.Sequence(
        'GreetBranch', memory=True, children=[
            CheckBB('ShouldGreet', '/social/should_greet',
                    check_fn=lambda v: v is True),
            SpeakBehaviour(node, bb_key='/social/greet_text', voice='happy'),
            SetBB('ClearGreet', '/social/should_greet', False),
        ]
    )

    # Dialogue branch: the BT orchestrates speech + gesture in parallel.
    # Starts when the LLM has put text into the BB via /llm_response.
    # Interrupted automatically if the person leaves (SocialBranch gate above).
    # Facial expression here no longer goes through ExpressEmotion/BT: tts_node
    # itself holds it for the whole time the phrase is being spoken (see
    # /face_expression_hold) — previously the BT twitched the face for ~1s in
    # parallel with speech and it rolled back to rest before the end of the phrase.
    dialogue_branch = py_trees.composites.Sequence(
        'DialogueBranch', memory=True, children=[
            # has_content=True for a normal reply (text non-empty) and when streaming (text='')
            CheckBB('HasLLMContent', '/llm/has_content', check_fn=lambda v: v is True),
            py_trees.composites.Parallel(
                'DialogueParallel',
                policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
                children=[
                    # with streamed=True text='' → SpeakBehaviour is SUCCESS at once (no TTS goal)
                    SpeakBehaviour(node,
                                   bb_key='/llm/text',
                                   voice_bb_key='/llm/voice_style'),
                    GesticulationAction(node),
                ]
            ),
            # Clear the BB after completion — so the reply isn't repeated
            SetBB('ClearLLMText',       '/llm/text',        ''),
            SetBB('ClearLLMEmo',        '/llm/emotion',     'neutral'),
            SetBB('ClearLLMVoice',      '/llm/voice_style', ''),
            SetBB('ClearLLMContent',    '/llm/has_content', False),
        ]
    )

    # Repeated face search by sound/voice hint — lives INSIDE
    # social_selector (not in SoundScanBranch/PIRScanBranch lower in the tree —
    # those are never ticked while a dialogue is in progress, see project memory face
    # search retry). All three children return SUCCESS synchronously within a single
    # tick, so dialogue_branch starts speech with a delay of no more than ~100ms.
    face_search_branch = py_trees.composites.Sequence(
        'FaceSearchBranch', memory=False, children=[
            CheckBB('HasFaceSearchPending', '/social/face_search_pending',
                    check_fn=lambda v: v is True),
            FaceSearchAttempt(node),
            SetBB('ClearFaceSearchPending', '/social/face_search_pending', False),
        ]
    )

    social_selector = py_trees.composites.Selector(
        'SocialSelector', memory=False, children=[
            introducing_block,
            greet_branch,
            face_search_branch,
            dialogue_branch,
            IdleBlinkBehaviour(node, 'IdleGaze'),  # eye tracking + blinking
        ]
    )

    # Gate: Sequence(no-memory) — if the person leaves, CheckPersonPresent is FAILURE
    # → the whole branch is interrupted → terminate(INVALID) on the RUNNING SpeakBehaviour → TTS cancelled
    social_branch = py_trees.composites.Sequence(
        'SocialBranch', memory=False, children=[
            CheckBB('IsPersonPresent', '/social/person_present',
                    check_fn=lambda v: v is True),
            social_selector,
        ]
    )

    # ── 6. Farewell branch (the person has just left) ────────────────────
    farewell_branch = py_trees.composites.Sequence(
        'FarewellBranch', memory=True, children=[
            CheckBB('FarewellPending', '/social/farewell_pending',
                    check_fn=lambda v: v is True),
            SetBB('ClearFarewell', '/social/farewell_pending', False),
            py_trees.composites.Parallel(
                'FarewellParallel',
                policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
                children=[
                    SpeakBehaviour(node, bb_key='/social/farewell_text'),
                    ExpressEmotion(node, emotion='neutral'),
                ]
            ),
        ]
    )

    # ── 7a. Sound scan (wake word → turn the TORSO toward the voice) ──────
    sound_scan = py_trees.composites.Sequence(
        'SoundScanBranch', memory=False, children=[
            CheckBB('IsSoundScanActive', '/sound/scan_active',
                    check_fn=lambda v: v is True),
            SoundScanBehaviour(node),
        ]
    )

    # ── 7b. PIR scan (pure motion without voice — search for a face with the head) ────
    pir_scan = py_trees.composites.Sequence(
        'PIRScanBranch', memory=False, children=[
            CheckBB('IsPIRScanActive', '/pir/scan_active',
                    check_fn=lambda v: v is True),
            # Never scan over an ongoing dialogue — person_present alone is not
            # enough (see BehaviorManagerNode.is_dialogue_active). A scan already
            # running is preempted here too: terminate(INVALID) hands the head
            # back to head_tracker.
            CheckFn('NoActiveDialogue', node.pir_scan_allowed),
            PIRScanBehaviour(node),
        ]
    )

    # ── 8. Global idle ────────────────────────────────────────────────────
    global_idle = IdleBlinkBehaviour(node, 'GlobalIdle')

    # ── Root ──────────────────────────────────────────────────────────────
    root = py_trees.composites.Selector(
        'Root', memory=False, children=[
            sleep_transition,
            sleep_active,
            robot_command,
            web_search,
            social_branch,
            farewell_branch,
            sound_scan,
            pir_scan,
            global_idle,
        ]
    )
    return root


# ══════════════════════════════════════════════════════════════════════════════
# BEHAVIOR MANAGER NODE
# ══════════════════════════════════════════════════════════════════════════════

class BehaviorManagerNode(LifecycleNode):
    # Face-search retry thresholds (see FaceSearchAttempt) — no more often than once per
    # _FACE_SEARCH_MIN_RETRY_INTERVAL_SEC (protection against duplicate turns if
    # several utterances/track losses arrive faster than one torso motion
    # cycle), ask "where are you" no earlier than _FACE_SEARCH_ASK_MIN_ATTEMPTS
    # failed attempts and no more than once per _FACE_SEARCH_ASK_COOLDOWN_SEC.
    _FACE_SEARCH_MIN_RETRY_INTERVAL_SEC = 2.5
    _FACE_SEARCH_ASK_MIN_ATTEMPTS       = 2
    _FACE_SEARCH_ASK_COOLDOWN_SEC       = 15.0
    # Grace before reacting to a face lost mid-dialogue (_face_locked_cb) —
    # /head_tracker/face_locked flips to False only after 2s of track
    # "staleness" anyway (_STALE_SEC in vision_head_tracker_node), but short
    # losses (a blink, looking away for a second) are commonplace in conversation and
    # recover on their own. Live bug 2026-08-28: without this grace every
    # such trifle triggered a real torso turn every few seconds
    # ("alternating between losing the face and finding it"). We wait the same
    # amount on top of the standard 2s — ~5s in total, still noticeably faster than
    # head_tracker's full return to rest (return_timeout_sec, in practice ~7s).
    _FACE_LOST_GRACE_SEC = 3.0
    # The PIR sensor is physically located IN THE TORSO — when we turn the torso
    # ourselves (SoundScan/FaceSearch/a manual robot_control command with scope=partial|full),
    # the sensor sees the robot's own motion and mistakes it for a person,
    # launching PIRScan right in the middle of a dialogue. Live bug 2026-08-28 (found
    # by the user). The window is chosen with a margin over the turn time
    # (~60°/1.0rad/s ≈ 1s) + the typical PIR/Arduino heartbeat delay.
    _PIR_SELF_MOTION_BLANK_SEC = 2.5
    # How long after the last voice exchange (LLM reply / wake word) a dialogue
    # still counts as active for the PIR gate (is_dialogue_active). Needed because
    # /social/person_present is NOT reliable mid-dialogue: with an unrecognized
    # face identity_manager keeps sending person_present=False @ 2Hz and
    # overwrites the voice-presence True from _llm_response_cb. Live bug
    # 2026-09-26: PIRScan started while head_tracker was following the speaker,
    # and the head went back to rest.
    _DIALOGUE_ACTIVE_HOLD_SEC = 30.0
    # An explicit turn command from the user (robot_control head / look_direction)
    # outranks the robot's own guesses (sound direction, voice hint): for this long
    # after it, SoundScan/FaceSearch do not turn anything. Live bug 2026-09-30:
    # FaceSearch turned the torso 0.2 s after "turn right" was executed.
    _MANUAL_TURN_PRIORITY_SEC = 20.0

    def __init__(self):
        super().__init__('behavior_manager_node')

        self._pir_cooldown         = 20.0  # will be overwritten in on_configure
        self._pir_next_scan_at     = 0.0
        self._pir_prev_state       = False
        self._last_own_torso_move_mono = float('-inf')
        self._last_manual_turn_mono    = float('-inf')  # see manual_turn_recent
        self._last_dialogue_mono   = float('-inf')  # see is_dialogue_active

        # Cache of the last /sound_direction and /human_detected for SoundScanBehaviour
        # (it reads them via getattr(node, ...), not via a topic subscription in the behaviour itself)
        self._last_sound_angle      = 0.0
        self._last_sound_confidence = 0.0
        self._human_detected        = False
        self._last_human_angle      = 0.0   # /human_angle_deg (OAK-D, atan2(x_mm,z_mm)) — SoundScanBehaviour aims the head before enable_head_tracker

        # ── Face-search retry (repeated face search on every utterance until
        # head_tracker locks a face — see FaceSearchAttempt/_face_locked_cb) ──
        self._face_locked                   = False   # /head_tracker/face_locked
        self._face_ever_locked              = False   # have we ever locked on in this session?
        self._last_direction_hint           = 'none'  # /voice/direction_hint
        self._face_search_attempts          = 0
        self._face_search_pending_check     = None    # {'direction': deg|None} — outcome of the PREVIOUS attempt
        self._face_search_last_attempt_mono = 0.0
        self._face_search_last_ask_mono     = float('-inf')
        self._face_lost_since               = None    # monotonic ts of the loss — grace before reacting

        # ── Blackboard initialization ──────────────────────────────────────
        self._bb = py_trees.blackboard.Client(name='BehaviorManager')
        _bb_defaults = {
            '/robot/sleep':             False,
            '/robot/sleep_requested':   False,
            '/robot/sleep_text':        '',
            '/robot/command':           {},
            '/llm/text':                '',
            '/llm/voice_style':         '',
            '/llm/emotion':             'neutral',
            '/llm/has_content':         False,
            '/social/person_present':   False,
            '/social/name':             '',
            '/social/emotion':          'neutral',
            '/social/should_greet':     False,
            '/social/greet_text':       '',
            '/social/farewell_pending': False,
            '/social/farewell_text':    '',
            '/social/introducing':      False,
            '/social/introduce_pending': False,
            '/social/introduce_text':   '',
            '/search/query':            '',
            '/search/result':           {},
            '/pir/scan_active':         False,
            '/sound/scan_active':       False,
            '/social/face_search_pending': False,
            '/scene/person_count':      0,
            '/scene/objects_summary':   '',
            '/scene/location':          '',
        }
        for key in _bb_defaults:
            self._bb.register_key(key=key, access=py_trees.common.Access.WRITE)
        self._bb.robot.sleep             = False
        self._bb.robot.sleep_requested   = False
        self._bb.robot.sleep_text        = ''
        self._bb.robot.command           = {}
        self._bb.llm.text                = ''
        self._bb.llm.voice_style         = ''
        self._bb.llm.emotion             = 'neutral'
        self._bb.llm.has_content         = False
        self._bb.social.person_present   = False
        self._bb.social.name             = ''
        self._bb.social.emotion          = 'neutral'
        self._bb.social.should_greet     = False
        self._bb.social.greet_text       = ''
        self._bb.social.farewell_pending = False
        self._bb.social.farewell_text    = ''
        self._bb.social.introducing      = False
        self._bb.social.introduce_pending = False
        self._bb.social.introduce_text   = ''
        self._bb.search.query            = ''
        self._bb.search.result           = {}
        self._bb.pir.scan_active         = False
        self._bb.sound.scan_active       = False
        self._bb.social.face_search_pending = False
        self._bb.scene.person_count      = 0
        self._bb.scene.objects_summary   = ''
        self._bb.scene.location          = ''

        # To detect the person_present True→False transition
        self._person_was_present = False

        # Farewell delay timer — give TTS time to finish before person_present=False
        self._farewell_timer: threading.Timer | None = None
        self._farewell_delay_sec = 8.0  # max time to wait for TTS to end

        # After say_goodbye: _social_ctx_cb must NOT overwrite person_present=False
        # from identity_manager (it is in IDLE), otherwise the BT gate kills the farewell TTS.
        # _finalize_farewell() will set False itself after the timer.
        self._suppress_social_present_until: float = 0.0  # time.monotonic()

        self._tree       = None
        self._tick_timer = None

        # Lifecycle ACTIVE flag. Subscriptions and threading timers outlive
        # deactivate, and the motor publishers here are plain (non-lifecycle) —
        # everything that can move a servo outside the BT tick checks this flag.
        self.lc_active = False
        self._deactivate_hooks: list = []   # BT leaves' cancel callbacks

    def send_joints(self, source: str, priority: int, lease_sec: float,
                    joints_deg: dict, vel: float = 0.0) -> None:
        """Arbitrated servo command (inmoov_msgs/JointCommand on /joint_cmd).
        joints_deg: joint → angle in degrees (0..180, center 90)."""
        if not self.lc_active:
            return
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name     = list(joints_deg.keys())
        js.position = [(float(v) - 90.0) * math.pi / 180.0 for v in joints_deg.values()]
        if vel != 0.0:
            js.velocity = [float(vel)] * len(js.name)
        self._joint_cmd_pub.publish(JointCommand(
            source=source, priority=priority, lease_sec=float(lease_sec), cmd=js))

    def release_joints(self, source: str, names) -> None:
        """Give up this source's leases (works while INACTIVE too — used on deactivate)."""
        js = JointState()
        js.name = list(names)   # positions are ignored on release
        self._joint_cmd_pub.publish(JointCommand(source=source, release=True, cmd=js))

    def add_deactivate_hook(self, fn) -> None:
        """BT leaves register cleanup here (pending timers etc.), run on deactivate."""
        self._deactivate_hooks.append(fn)

    # ── Vision control (BT is the sole orchestrator) ─────────────────────

    def enable_face_detection(self, enabled: bool) -> None:
        msg = Bool()
        msg.data = enabled
        self._face_det_pub.publish(msg)

    def enable_head_tracker(self, enabled: bool) -> None:
        msg = Bool()
        msg.data = enabled
        self._head_tracker_pub.publish(msg)

    def publish_blink(self, closing: bool) -> None:
        if self.lc_active:
            self._blink_pub.publish(Bool(data=closing))

    def note_own_torso_move(self) -> None:
        """Call on EVERY command to the midstom joint — the PIR is physically in
        the torso and sees the robot's own rotation as "human motion"
        (see _PIR_SELF_MOTION_BLANK_SEC/_pir_cb)."""
        self._last_own_torso_move_mono = time.monotonic()

    def note_manual_turn(self) -> None:
        """An explicit head/torso turn command from the user was executed."""
        self._last_manual_turn_mono = time.monotonic()

    def manual_turn_recent(self) -> bool:
        return time.monotonic() - self._last_manual_turn_mono < self._MANUAL_TURN_PRIORITY_SEC

    def turn_toward(self, midstom: float, source: str, priority: int,
                    lease_sec: float, vel: float = 0.0) -> float:
        """Turn the torso AND the head toward the same side (a sound/voice-hint
        guess). The head adds SoundScanBehaviour._HEAD_TURN_OFFSET on top of the
        torso in the same direction; midstom at center → head straight too.
        Live bug 2026-09-30: only the torso turned toward the voice, the head
        stayed at 90°. Returns the rothead sent."""
        side = (midstom > SoundScanBehaviour._CENTER) - (midstom < SoundScanBehaviour._CENTER)
        rothead = SoundScanBehaviour._ROTHEAD_CENTER + side * SoundScanBehaviour._HEAD_TURN_OFFSET
        self.note_own_torso_move()
        self.send_joints(source, priority, lease_sec,
                         {'midstom': midstom, 'rothead': rothead,
                          'neck': SoundScanBehaviour._NECK_REST}, vel=vel)
        return rothead

    def aim_head_at_human(self) -> None:
        """One-shot precise aiming of the head (rothead) at the person using
        /human_angle_deg (OAK-D atan2(x_mm,z_mm)) — the narrow FOV of the eye cameras
        may fail to catch the face after just a coarse TORSO turn
        (SoundScanBehaviour/FaceSearchAttempt turn only midstom,
        to the fixed extreme 60°/120°, without a precise follow-up with the head).

        Previously called only from SoundScanBehaviour._found() (the one-shot
        scan right after the wake word). Moved up to the node level and also wired
        into _human_detected_cb (see below) — live bug 2026-08-31: during
        a dialogue FaceSearchAttempt turned the torso all the way, but rothead
        stood at 90° for the whole dialogue (no track for 7s → rest), because the
        precise follow-up with the head did not happen at all if OAK-D saw the person NOT
        right after the wake word but later, in the middle of an ongoing conversation.

        Idea from the AIR2025 paper (Saini et al.). Formula/sign — see
        SoundScanBehaviour._ROTHEAD_* / _AIM_GAIN, not redefined here
        again, the same ones are used."""
        if not self.lc_active:   # called from _human_detected_cb, which runs while INACTIVE too
            return
        angle = self._last_human_angle
        rothead = max(SoundScanBehaviour._ROTHEAD_MIN, min(
            SoundScanBehaviour._ROTHEAD_MAX,
            SoundScanBehaviour._ROTHEAD_CENTER + angle * SoundScanBehaviour._AIM_GAIN))
        # Short lease: the turn itself; head_tracker takes over right after
        self.send_joints('bt_scan', JointCommand.PRIORITY_BT_SCAN, 1.5,
                         {'rothead': rothead, 'neck': SoundScanBehaviour._NECK_REST},
                         vel=SoundScanBehaviour._TURN_VEL)
        self.get_logger().info(
            f'AimHead: aiming the head at the person (OAK-D angle={angle:.0f}° → rothead={rothead:.0f}°)')

    # ── Face-search retry (see FaceSearchAttempt) ─────────────────────────

    def _reset_face_search(self) -> None:
        """Reset the face-search session — a new session (wake word/sleep), the person
        left, or the face was actually locked. Do NOT call on intermediate successes
        such as SoundScanBehaviour._found() (OAK-D coarsely saw a body) — there
        head_tracker may still fail to lock the face for a few seconds."""
        self._face_search_attempts      = 0
        self._face_ever_locked          = False
        self._face_search_pending_check = None
        self._face_search_last_ask_mono = float('-inf')
        self._last_direction_hint       = 'none'
        self._face_lost_since           = None
        # In case the flag managed to be set before the reset (e.g.
        # _direction_hint_cb fired on a hint from the outgoing session) —
        # otherwise it would "fire" in the next, unrelated session.
        self._bb.social.face_search_pending = False
        self._publish_face_search_status(active=False, attempts=0, ask_now=False)

    def _publish_face_search_status(self, active: bool, attempts: int, ask_now: bool) -> None:
        msg = String()
        msg.data = json.dumps({
            'active': active,
            'attempts': attempts,
            'ask_now': ask_now,
            'kind': 'never_found' if not self._face_ever_locked else 'lost_again',
            # Live bug 2026-08-31: the LLM answered "yes, I see you!" to a direct
            # question even though head_tracker was not holding a face — we weave the truth into
            # the prompt (_build_face_search_block) so it does not hallucinate.
            'locked': self._face_locked,
        })
        self._face_search_status_pub.publish(msg)

    def is_dialogue_active(self) -> bool:
        """A dialogue is in progress: head_tracker holds a face, an LLM reply is
        pending/being spoken, or the last voice exchange was recent. Deliberately
        independent of /social/person_present (see _DIALOGUE_ACTIVE_HOLD_SEC)."""
        if self._face_locked:
            return True
        try:
            if self._bb.llm.has_content or self._bb.social.person_present:
                return True
        except Exception:
            pass
        return time.monotonic() - self._last_dialogue_mono < self._DIALOGUE_ACTIVE_HOLD_SEC

    def pir_scan_allowed(self) -> bool:
        """BT gate for PIRScanBranch. During a dialogue also drops a pending
        /pir/scan_active, so a stale request doesn't fire after the dialogue ends."""
        if not self.is_dialogue_active():
            return True
        if self._bb.pir.scan_active:
            self._bb.pir.scan_active = False
            self.get_logger().info('PIR: scan request dropped — dialogue in progress')
        return False

    def _face_locked_cb(self, msg: Bool):
        was_locked = self._face_locked
        self._face_locked = msg.data

        if msg.data:
            self._face_lost_since = None
            if not was_locked:
                # Found (or re-locked) — the search session is over.
                self._face_ever_locked = True
                self._face_search_attempts = 0
                self._face_search_pending_check = None
                self._face_search_last_ask_mono = float('-inf')
                # If the flag managed to be set (an utterance/loss arrived slightly earlier
                # than head_tracker re-locked the face itself) — clear it, otherwise
                # FaceSearchBranch would do a redundant turn right after the find.
                self._bb.social.face_search_pending = False
                self._publish_face_search_status(active=False, attempts=0, ask_now=False)
                self.get_logger().info('FaceSearch: HeadTracker locked the face — counter reset')
            return

        # msg.data == False. Lost — do NOT react instantly: short losses
        # (a blink, looking away for a second) are commonplace in conversation and
        # recover on their own in under a second (see the live bug 2026-08-28 —
        # without the grace every such trifle jerked the torso). We give _FACE_LOST_GRACE_SEC
        # on top of the standard 2s track "staleness", and react only if the loss
        # lasted longer — that is no longer a blink.
        if was_locked:
            self._face_lost_since = time.monotonic()
            # locked=False must reach llm_node IMMEDIATELY (not after the grace) —
            # honesty about "I see / I don't see" matters more than the physical reaction
            # (torso turn), which we still hold back for the grace.
            self._publish_face_search_status(
                active=True, attempts=self._face_search_attempts, ask_now=False)
            return
        if self._face_lost_since is None:
            return  # this loss was already handled (or there was none) — don't spam
        if time.monotonic() - self._face_lost_since < self._FACE_LOST_GRACE_SEC:
            return

        self._face_lost_since = None  # handled — the next trigger only on a new loss/utterance
        if self._bb.social.person_present:
            self._bb.social.face_search_pending = True
            self.get_logger().info(
                f'FaceSearch: face lost for >{self._FACE_LOST_GRACE_SEC:.1f}s '
                f'during a dialogue — scheduling an attempt')

    def _direction_hint_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._last_direction_hint = data.get('direction', 'none')
        if self._face_locked:
            return
        # Do NOT gate on the current /social/person_present: when a person returns
        # (after a silent leave/OakD veto) this hint arrives EARLIER than
        # person_present manages to become True (that is set later — either by
        # _llm_response_cb AFTER the LLM's reply to the same utterance, or
        # _social_ctx_cb still holds False while identity_manager is in IDLE).
        # Live bug 2026-08-28: "I am to your left" was silently dropped
        # precisely by this gate. FaceSearchBranch gates on person_present itself
        # at the BT level (social_branch) — setting the flag here is safe, it
        # just waits for the next tick when the presence is confirmed.
        self._bb.social.face_search_pending = True

    def record_face_search_attempt(self):
        """Called from FaceSearchAttempt.update(). Resolves the outcome of the
        PREVIOUS attempt (increments the counter on failure), picks the target of
        this attempt (voice hint > /sound_direction > nothing),
        publishes the status. Returns (torso_target_deg|None, source|None)."""
        prev = self._face_search_pending_check
        if prev is not None:
            if prev['direction'] is None or not self._human_detected:
                self._face_search_attempts += 1

        if self.manual_turn_recent():
            # The user has just told us explicitly where to turn — that wins over
            # our own guesses; the hint came from the same utterance, drop it.
            self._last_direction_hint = 'none'
            target, source = None, 'manual_command'
        elif self._last_direction_hint == 'right':
            target, source = SoundScanBehaviour._RIGHT, 'voice_hint'
            # A one-shot hint — consume it right away, otherwise on the next failure
            # (or after look_direction/a photo ALREADY disproved it) it
            # would be reused until the next utterance with a direction.
            # Live bug 2026-08-31: "I'm on the left" kept being applied even
            # after a photo confirmed that there was nobody on the left.
            self._last_direction_hint = 'none'
        elif self._last_direction_hint == 'left':
            target, source = SoundScanBehaviour._LEFT, 'voice_hint'
            self._last_direction_hint = 'none'
        elif (self._last_sound_confidence >= SoundScanBehaviour._MIN_CONFIDENCE
              and abs(self._last_sound_angle) >= SoundScanBehaviour._MIN_ANGLE_DEG):
            target = (SoundScanBehaviour._RIGHT if self._last_sound_angle > 0
                      else SoundScanBehaviour._LEFT)
            source = 'sound'
        else:
            target, source = None, None

        self._face_search_pending_check = {'direction': target}
        self._face_search_last_attempt_mono = time.monotonic()

        ask_now = False
        if self._face_search_attempts >= self._FACE_SEARCH_ASK_MIN_ATTEMPTS:
            now = time.monotonic()
            if now - self._face_search_last_ask_mono >= self._FACE_SEARCH_ASK_COOLDOWN_SEC:
                ask_now = True
                self._face_search_last_ask_mono = now
        self._publish_face_search_status(
            active=True, attempts=self._face_search_attempts, ask_now=ask_now)
        return target, source

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _llm_response_cb(self, msg: String):
        """LLM response → Blackboard. The BT DialogueBranch will see the text and orchestrate speech."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Invalid /llm_response JSON: {e}')
            return

        streamed = data.get('streamed', False)
        via_telegram = data.get('telegram', False)
        self._bb.llm.text        = '' if streamed else data.get('text', '')
        self._bb.llm.voice_style = data.get('voice_instruct', '')
        # /llm/emotion no longer arrives as a separate field — the payload merged
        # emotion and voice style into a single voice_instruct (see llm_node.py
        # _tool_set_voice_style). The value here is needed only by GesticulationAction
        # for choosing a gesture (wave on happy) — the facial expression is now synchronized
        # with speech directly in tts_node (/face_expression_hold), not through the BT.
        self._bb.llm.emotion     = data.get('voice_instruct', '') or 'neutral'
        # has_content=True triggers the BT DialogueBranch for speech/gesture
        # (with streamed=True the text is empty, but SpeakBehaviour succeeds at once → Gesture runs)
        self._bb.llm.has_content = bool(
            self._bb.llm.text or streamed
        )
        if self._bb.llm.has_content and not via_telegram:
            self._last_dialogue_mono = time.monotonic()
            # Voice = confirmation of presence: allow DialogueBranch even without a face.
            # Telegram requests do not set person_present — the person is not physically present.
            was_present = self._bb.social.person_present
            self._bb.social.person_present = True
            if not was_present:
                # Presence is confirmed by voice, not by a successful scan — without
                # this face_detection/head_tracker may stay disabled,
                # and face-search retry (FaceSearchAttempt) would be useless.
                self.enable_face_detection(True)
                self.enable_head_tracker(True)
                self.get_logger().info(
                    'LLM response → person_present=True (voice presence), vision enabled')
            else:
                self.get_logger().info('LLM response → person_present=True (voice presence)')
        self.get_logger().debug(
            f'LLM→BB: "{self._bb.llm.text[:60]}" '
            f'[{self._bb.llm.emotion}]'
        )

    def _event_cb(self, msg: String):
        """Robot events from LLM tool calls (move/arm/head/sleep/search)."""
        if not self.lc_active:
            # INACTIVE: don't queue it — it would run on the next activation
            self.get_logger().warn(f'robot_events while inactive — dropped: {msg.data[:80]}')
            return
        try:
            event = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Invalid JSON: {e}')
            return

        action = event.get('action', '')
        self.get_logger().info(f'Event: {action}')

        if action in ('move', 'arm', 'head') and self._bb.robot.sleep:
            # SleepActive blocks the tree, so it would sit in the BB and run on WAKE —
            # a stale motion out of nowhere. Asleep means no motion.
            self.get_logger().warn(f'robot_events: {action} while asleep — dropped')
            return
        if action in ('move', 'arm', 'head', 'status'):
            self._bb.robot.command = event
        elif action == 'sleep':
            self._bb.robot.sleep_requested = True
            self._bb.robot.sleep_text = event.get(
                'text', 'Спокойной ночи! Скажи «Эй Лёня» чтобы разбудить меня.')
        elif action == 'goodbye':
            # The LLM said goodbye explicitly. The farewell `text` is sent straight to TTS by llm_node —
            # FarewellBranch is not needed (otherwise the farewell would be spoken twice).
            # Just switch IM to IDLE and start the timer to turn vision off.
            self._bb.social.should_greet      = False
            self._bb.social.introducing       = False
            self._bb.social.introduce_pending = False
            # Switch IdentityManager to IDLE (tear down the session immediately)
            go_idle_msg = Bool()
            go_idle_msg.data = True
            self._go_idle_pub.publish(go_idle_msg)
            # Turn face_detection and head_tracker off immediately — the person is saying goodbye,
            # a repeat greeting must not happen while TTS has not finished yet.
            self.enable_face_detection(False)
            self.enable_head_tracker(False)
            # Block overwriting person_present from social_ctx until the end of the farewell TTS.
            # Without this identity_manager (in IDLE) sends person_present=False every 0.5s,
            # and the BT gate kills SpeakBehaviour via terminate(INVALID) before the first word.
            self._suppress_social_present_until = (
                time.monotonic() + self._farewell_delay_sec + 2.0)
            # Timer for the final cleanup after TTS
            if self._farewell_timer is not None:
                self._farewell_timer.cancel()
            self._farewell_timer = threading.Timer(
                self._farewell_delay_sec, self._finalize_farewell)
            self._farewell_timer.daemon = True
            self._farewell_timer.start()
            self.get_logger().info(
                f'say_goodbye → IDLE, face_detection turned off immediately, '
                f'final cleanup in {self._farewell_delay_sec:.0f}s')
        elif action == 'search':
            self._bb.search.query = event.get('query', '')
        else:
            self.get_logger().warn(f'Unknown action in event_cb: "{action}"')

    def _social_ctx_cb(self, msg: String):
        """Extended social context from IdentityManager → Blackboard."""
        try:
            ctx = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        # After say_goodbye: identity_manager is in IDLE and sends person_present=False @ 2Hz.
        # Writing this into the BB would make the BT gate kill the farewell TTS via terminate(INVALID).
        # _finalize_farewell() will set False itself after the timer.
        if time.monotonic() < self._suppress_social_present_until:
            pass  # don't update person_present — let it stay True from _llm_response_cb
        else:
            self._bb.social.person_present = ctx.get('person_present', False)
        self._bb.social.name           = ctx.get('name', '')
        self._bb.social.emotion        = ctx.get('emotion', 'neutral')
        self._bb.social.introducing    = ctx.get('introducing', False)

        # should_greet — a one-shot signal: set only when IM says so
        if ctx.get('should_greet', False):
            self._bb.social.should_greet = True
            self._bb.social.greet_text   = ctx.get('greet_text', '')

        # introduce_pending — a one-shot signal: IM wants to speak the introduction phrase
        if ctx.get('introduce_pending', False):
            self._bb.social.introduce_pending = True
            self._bb.social.introduce_text    = ctx.get('introduce_text', '')

    def _scene_ctx_cb(self, msg: String):
        """Scene summary (objects + people) from scene_manager_node → Blackboard."""
        try:
            ctx = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        self._bb.scene.person_count    = ctx.get('person_count', 0)
        self._bb.scene.location        = ctx.get('location', '')
        self._bb.scene.objects_summary = ', '.join(
            f"{o.get('label')}:{o.get('count')}" for o in ctx.get('objects', [])
        )

    def _person_present_cb(self, msg: Bool):
        """The person left → let TTS finish speaking, then say goodbye."""
        was_present = self._person_was_present
        now_present = msg.data
        self._person_was_present = now_present

        if now_present and not was_present:
            # The person came back — cancel the deferred farewell
            if self._farewell_timer is not None:
                self._farewell_timer.cancel()
                self._farewell_timer = None
            self._bb.social.person_present   = True
            self._bb.social.farewell_pending = False
            self._bb.social.farewell_text    = ''
            # Presence is confirmed by this topic (not necessarily by a successful
            # scan) — turn vision on, otherwise face-search retry is useless
            # (see _llm_response_cb, the same bug fix).
            self.enable_face_detection(True)
            self.enable_head_tracker(True)

        elif was_present and not now_present:
            # Silent leave (timeout / OakD veto) — do NOT speak a farewell.
            # A farewell only in response to an explicit "bye" via the say_goodbye tool call.
            name = self._bb.social.name
            self._bb.social.should_greet      = False
            self._bb.social.introducing       = False
            self._bb.social.introduce_pending = False
            self.get_logger().info(
                f'Person left ({name}) — silent IDLE (no farewell)')

            # After farewell_delay_sec turn vision off and finish the dialogue
            if self._farewell_timer is not None:
                self._farewell_timer.cancel()
            self._farewell_timer = threading.Timer(
                self._farewell_delay_sec, self._finalize_farewell)
            self._farewell_timer.daemon = True
            self._farewell_timer.start()

    def _finalize_farewell(self):
        """Called by the timer: close the dialogue and turn vision off."""
        self._farewell_timer = None
        if not self.lc_active:
            return
        self._bb.social.person_present = False
        self.enable_head_tracker(False)
        self.enable_face_detection(False)
        # Reset pending LLM — there is no one left to speak to
        self._bb.llm.text        = ''
        self._bb.llm.emotion     = 'neutral'
        self._bb.llm.voice_style = ''
        self._reset_face_search()  # the person left — the face-search session is over
        self._last_dialogue_mono = float('-inf')  # dialogue over — PIR may scan again
        self.get_logger().info('Farewell: person_present=False, vision turned off')

    def _robot_sleep_cb(self, msg: Bool):
        """Synchronize the sleep mode from the latched topic."""
        self._bb.robot.sleep = msg.data
        self._bb.robot.command = {}   # a pre-sleep command must not fire on WAKE
        if not msg.data:
            self._bb.robot.sleep_requested = False
            self._pir_next_scan_at = 0.0  # reset the cooldown — the wake word is more important
            # Waking always goes through the wake word (voice_detector publishes
            # /robot_sleep False on it) — so there is a direction, use
            # SoundScan (torso turn), not the old PIR head scan.
            self._bb.sound.scan_active = True
            self._reset_face_search()  # new session — the old counter is stale
            self.get_logger().info('Waking up (by wake word) — starting SoundScan')

    def _sound_direction_cb(self, msg: SoundDirection):
        """Cache of the last /sound_direction for SoundScanBehaviour."""
        self._last_sound_angle      = msg.angle_deg
        self._last_sound_confidence = msg.confidence

    def _human_detected_cb(self, msg: Bool):
        """Cache of the last /human_detected (OAK-D) for SoundScanBehaviour.

        On the rising edge (False→True), if a face is not yet locked and
        a person is nearby (dialogue active) — immediately aim the head precisely using
        OAK-D (aim_head_at_human), without waiting for head_tracker to do it
        with its narrow eye cameras. Previously this follow-up happened
        ONLY in SoundScanBehaviour._found() (right after the wake word) — if
        OAK-D saw the person later, in the middle of an already ongoing dialogue (the typical
        case for FaceSearchAttempt, which turns only the torso), the head
        stayed at rest. Live bug 2026-08-31."""
        was_detected = self._human_detected
        self._human_detected = msg.data
        if msg.data and not was_detected and not self._face_locked:
            try:
                person_present = self._bb.social.person_present
            except Exception:
                person_present = False
            if person_present:
                self.aim_head_at_human()

    def _human_angle_cb(self, msg: Float32):
        """Cache of the last /human_angle_deg (OAK-D atan2(x_mm,z_mm)) for aiming the head."""
        self._last_human_angle = msg.data

    def _wake_detected_cb(self, msg: Bool):
        """Wake word in IDLE → start SoundScan (turn the torso toward the voice)."""
        if not msg.data:
            return
        if self._bb.robot.sleep:
            return  # sleep mode is handled via /robot_sleep False
        self._last_dialogue_mono = time.monotonic()
        try:
            person_present = self._bb.social.person_present
        except Exception:
            person_present = False

        if person_present:
            # The user said the wake word while the robot was greeting — an interruption.
            # Reset should_greet so the BT doesn't loop on a repeated greeting.
            try:
                if self._bb.social.should_greet:
                    self._bb.social.should_greet = False
                    self._bb.social.greet_text   = ''
                    self.get_logger().info(
                        'Wake word interrupted the greeting → should_greet reset')
            except Exception:
                pass
            return

        if self._bb.pir.scan_active or self._bb.sound.scan_active:
            return
        self._pir_next_scan_at = 0.0   # the wake word beats the cooldown
        self._bb.sound.scan_active = True
        self._reset_face_search()  # new session — the old counter is stale
        self.get_logger().info('Wake word in IDLE → starting SoundScan (torso turn toward the voice)')

    def _pir_cb(self, msg: Bool):
        """PIR signal: react only to the rising edge (False→True).
        The Arduino sends a heartbeat every 5s — repeated Trues with a stuck HIGH are ignored.
        """
        prev             = self._pir_prev_state
        self._pir_prev_state = msg.data

        if not msg.data:
            return
        if prev:
            return  # heartbeat or sustained HIGH — not a new motion event

        # Rising edge: new motion detected
        if self._bb.robot.sleep:
            return
        try:
            person_present = self._bb.social.person_present
        except Exception:
            person_present = False
        # sound.scan_active covers only the window of SoundScan itself; human_detected
        # covers the wider window "OAK-D has already found a body, head_tracker is already
        # leading" — between the handover and the moment person_present actually
        # becomes True after recognition (which can take several seconds).
        # Without this check PIR motion (a person just standing/moving in front of
        # the robot) manages to start a SECOND, competing head scan on top of the
        # already-working head_tracker — the two fight over rothead/neck.
        # Found 2026-08-24: PIRScan started 100ms after a successful
        # SoundScan._found(), and the head thrashed between two command sources.
        if person_present or self._bb.pir.scan_active or self._bb.sound.scan_active \
                or self._human_detected:
            return
        if self.is_dialogue_active():
            self.get_logger().debug('PIR: motion during an active dialogue — ignoring')
            return
        now = time.monotonic()
        if now - self._last_own_torso_move_mono < self._PIR_SELF_MOTION_BLANK_SEC:
            self.get_logger().debug(
                'PIR: motion right after our own torso turn — treating as a self-trigger, ignoring')
            return
        if now < self._pir_next_scan_at:
            self.get_logger().debug('PIR: motion (cooldown active, ignoring)')
            return
        self._bb.pir.scan_active = True
        self._pir_next_scan_at   = now + self._pir_cooldown
        self.get_logger().info('PIR: motion — starting head scan')

    # ── Tick ─────────────────────────────────────────────────────────────

    def _tick(self):
        if self._tree is not None:
            self._tree.tick_once()


    # ── Lifecycle callbacks ────────────────────────────────────────────────

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores a repeated declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('tavily_api_key',        '')
        self._dp('bt_tick_rate_hz',       10.0)
        self._dp('pir_scan_cooldown_sec', 20.0)

        tavily_key          = self.get_parameter('tavily_api_key').value
        self._tick_rate     = self.get_parameter('bt_tick_rate_hz').value
        self._pir_cooldown  = self.get_parameter('pir_scan_cooldown_sec').value

        # All servo commands of this node (BT leaves + aim_head_at_human) — plain
        # publisher, gated by lc_active in send_joints()
        self._joint_cmd_pub = self.create_publisher(JointCommand, '/joint_cmd', 20)
        self._blink_pub     = self.create_publisher(Bool, '/eyes/blink', 10)

        self._tree = build_tree(self, tavily_key)
        self._tree.setup_with_descendants()

        lqos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._face_det_pub     = self.create_lifecycle_publisher(Bool, '/face_detection/enable', lqos)
        self._head_tracker_pub = self.create_lifecycle_publisher(Bool, '/head_tracker/enable',   lqos)
        self._go_idle_pub      = self.create_lifecycle_publisher(Bool, '/go_idle', 10)
        # Face-search retry status for llm_node (the "ask where you are" prompt) — latched,
        # like /robot_sleep, so that a restarted llm_node immediately sees the current status.
        self._face_search_status_pub = self.create_lifecycle_publisher(
            String, '/behavior/face_search_status', lqos)

        self.create_subscription(String, '/llm_response',   self._llm_response_cb,   10)
        self.create_subscription(String, 'robot_events',    self._event_cb,           10)
        self.create_subscription(String, '/social_context', self._social_ctx_cb,      10)
        self.create_subscription(String, '/scene/objects',  self._scene_ctx_cb,       10)
        self.create_subscription(Bool,   '/person_present', self._person_present_cb,  10)
        self.create_subscription(Bool,   '/pir_state',      self._pir_cb,             10)
        self.create_subscription(Bool,   'wake_detected',   self._wake_detected_cb,   10)
        self.create_subscription(Bool,   '/robot_sleep',    self._robot_sleep_cb,     lqos)
        self.create_subscription(SoundDirection, '/sound_direction', self._sound_direction_cb, 10)
        self.create_subscription(Bool,   '/human_detected', self._human_detected_cb,  10)
        self.create_subscription(Float32, '/human_angle_deg', self._human_angle_cb,   10)
        self.create_subscription(Bool,   '/head_tracker/face_locked', self._face_locked_cb, 10)
        self.create_subscription(String, '/voice/direction_hint', self._direction_hint_cb, 10)

        self.get_logger().info('BehaviorManager configured')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._face_det_pub.on_activate(state)
        self._head_tracker_pub.on_activate(state)
        self._go_idle_pub.on_activate(state)
        self._face_search_status_pub.on_activate(state)
        self.lc_active = True
        self.enable_face_detection(False)
        self.enable_head_tracker(False)
        self._tick_timer = self.create_timer(1.0 / self._tick_rate, self._tick)
        self.get_logger().info(
            f'BehaviorManager v2 ready (eternal tree @ {self._tick_rate:.0f} Hz). '
            f'Tavily: {"configured" if self.get_parameter("tavily_api_key").value else "not configured"}'
        )
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.lc_active = False
        if self._tick_timer:
            self.destroy_timer(self._tick_timer)
            self._tick_timer = None
        if self._farewell_timer is not None:
            self._farewell_timer.cancel()
            self._farewell_timer = None
        for hook in self._deactivate_hooks:
            try:
                hook()
            except Exception as e:
                self.get_logger().warn(f'deactivate hook {hook}: {e}')
        self._face_det_pub.on_deactivate(state)
        self._head_tracker_pub.on_deactivate(state)
        self._go_idle_pub.on_deactivate(state)
        self._face_search_status_pub.on_deactivate(state)
        self._bb.robot.command       = {}
        self._bb.llm.has_content     = False
        self._bb.social.person_present = False
        self._bb.pir.scan_active     = False
        self._bb.sound.scan_active   = False
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        if self._tick_timer:
            self.destroy_timer(self._tick_timer)
            self._tick_timer = None
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        if self._tick_timer:
            self.destroy_timer(self._tick_timer)
            self._tick_timer = None
        return TransitionCallbackReturn.SUCCESS


def main():
    rclpy.init()
    node = BehaviorManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
