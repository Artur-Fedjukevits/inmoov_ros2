#!/usr/bin/env python3
"""
vision_head_tracker_node.py
===========================
Controls the robot's gaze. Supports two cameras with automatic fallback.

Leader/follower logic:
  Leader — the left eye (when available):
    drives the head (rothead + neck) and publishes eye_lr_L / eye_ud_L.
    EYE_SYNC in arduino_comm_node mirrors the right eye automatically.

  Fallback — the right eye (if the left has sent no track message for
  _FALLBACK_SEC = 5 s):
    takes over head control and publishes eye_lr_R / eye_ud_R.
    EYE_SYNC mirrors the left eye.

  Both sets must not be published at the same time — EYE_SYNC creates a race
  (the last joint in the packet overwrites the mirror of the previous one).

Publishes:
  /joint_command  (JointState) — rothead, neck
  /face_command   (JointState) — eye_lr_L + eye_ud_L  | eye_lr_R + eye_ud_R
  /head_tracker/face_locked (Bool) — whether there is a fresh bbox RIGHT NOW
    (not stale, see _STALE_SEC). Not latched — False is published explicitly
    on disable. The single source of truth for "face caught" for the
    re-search behaviors in behavior_manager_node.

Subscribes:
  /face/tracks/left    (String JSON)
  /face/tracks/right   (String JSON)
  /head_tracker/enable (Bool latched)

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import json
import math
import threading
import time

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Bool

_STALE_SEC    = 2.0  # bbox goes stale (s) — the camera has not delivered a face
_FALLBACK_SEC = 5.0  # the right camera drives the head only if the left has been absent for 5+ s


def _bbox_area(bbox) -> float:
    if len(bbox) < 4:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _deg_to_rad(deg: float, center: float = 90.0) -> float:
    return (deg - center) * math.pi / 180.0


def _rad_to_deg(rad: float, center: float = 90.0) -> float:
    return rad * 180.0 / math.pi + center


class VisionHeadTrackerNode(LifecycleNode):
    def __init__(self):
        super().__init__('vision_head_tracker_node')
        self._head_pub        = None
        self._face_pub        = None
        self._timer           = None
        self._lock            = threading.Lock()
        self._enabled         = False
        self._rothead_range   = (30.0, 130.0)
        self._neck_range      = (1.0,  100.0)
        self._eye_lr_range    = (80.0, 100.0)
        self._eye_ud_range    = (80.0, 110.0)
        self._left_bbox       = self._right_bbox = None
        self._last_left_t     = self._last_right_t = 0.0
        # Separate timestamp: when the left CAMERA last sent any message
        # (even one without a face). Used to decide "camera is alive", not "face is present".
        self._last_left_msg_t: float = 0.0
        self._active_side     = 'left'
        # Live bug 2026-09-01 (found by the user): _tick() runs at
        # track_hz (15 Hz), while new detections arrive at detection_hz
        # (2.5 Hz by default) — previously, between two detections
        # _step_head() was applied to THE SAME offset up to _max_stale (5)
        # times in a row, which acted as a hidden 5x amplification of
        # gain_head/max_step_deg and as a cumulative error when a camera
        # frame "froze" (see face_detection_node._trigger_detection). Now
        # the head makes exactly ONE step per new detection (bbox_seq), not
        # per tick — the counter is compared with the last processed one.
        self._bbox_seq        = 0
        self._last_stepped_seq = -1
        # Drift diagnostics during active tracking (live bug 2026-08-31: the
        # user observed the head/torso gradually drifting AWAY from the face
        # that was actually caught, with no logs at all — the P controller
        # logged nothing by default). Throttled — otherwise it would spam at ~15 Hz.
        self._last_track_log_mono = 0.0
        self._TRACK_LOG_INTERVAL_SEC = 3.0
        # For the WARNING on a suspicious jump of target_track_id (see _tick()) —
        # live bug 2026-08-31: at the moment identity_manager confirmed the
        # identity and issued target_track_id, head_tracker switched to a
        # DIFFERENT track (not the one that was already well centered), the
        # offset jumped to the edge of the frame (e.g. -0.13 -> -0.79) and the
        # head was slowly pulled away from the real face for the whole dialogue
        # until the track was lost.
        self._prev_target_track_id = None
        # Runaway safeguard for the head (live bug 2026-08-31, confirmed by
        # the user): the offset did not decrease for 27 s in a row while
        # rothead honestly travelled ALMOST THE WHOLE physical range
        # (41° -> 130°, to the stop) — the cumulative head correction (unlike
        # the eye, which aims at an absolute target and does not run away by
        # itself) had no safety fuse of the kind "offset is not decreasing —
        # something is wrong, stop winding up steps".
        # If several detections in a row do NOT bring the offset closer to
        # zero — stop accumulating, before reaching the hardware limit.
        #
        # IMPORTANT (live bug 2026-09-01): comparing mag with the PREVIOUS
        # step (step by step) is not viable — the real bbox noise between two
        # neighbouring detections is easily +-0.05-0.15 (see
        # project_face_search_retry.md), so "did not improve by at least EPS
        # on EVERY single step" is almost never satisfied even with genuine
        # convergence — the head froze for ~10 s on a perfectly normal (just
        # noisy) track. Instead we compare with a BASELINE recorded at the
        # start of a series of failures — that way noise within the series
        # does not interfere, while a genuinely "flat" offset (as in the
        # 2026-08-31 bug — 27 s without movement) is still caught.
        self._runaway_streak      = 0
        self._streak_ref_mag      = None   # mag at the start of the current series
        self._RUNAWAY_MAX_STREAK  = 8       # detections in a row without progress from baseline
        self._RUNAWAY_IMPROVE_EPS = 0.04    # progress from baseline smaller than this does not count as improvement
        self._at_rest         = False
        self._target_person_id = self._target_track_id = None

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores re-declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('image_width',        640)
        self._dp('image_height',       480)
        self._dp('fov_h_deg',          60.0)
        self._dp('fov_v_deg',          45.0)
        self._dp('gain_head',           0.5)
        self._dp('gain_eye',            0.2)
        self._dp('dead_zone_px',        20)
        self._dp('head_dead_zone_px',   80)
        self._dp('eye_limit_deg',       8.0)
        self._dp('return_timeout_sec', 12.0)
        self._dp('track_hz',           15.0)
        self._dp('min_det_score',       0.50)
        self._dp('head_pan_dir',    1)
        self._dp('head_tilt_dir',  -1)
        self._dp('eye_pan_dir',    -1)
        self._dp('eye_tilt_dir',   -1)
        self._dp('max_step_deg',    2.0)
        self._dp('rest_rothead',   90.0)
        self._dp('rest_neck',      40.0)
        self._dp('rest_eye_lr',    90.0)
        self._dp('rest_eye_ud',   100.0)
        self._dp('bbox_ema_alpha',  0.4)
        self._dp('max_stale_ticks', 5)  # no longer used (see _bbox_seq), kept so as not to break launch files

        self._img_w          = self.get_parameter('image_width').value
        self._img_h          = self.get_parameter('image_height').value
        self._gain_head      = self.get_parameter('gain_head').value
        self._gain_eye       = self.get_parameter('gain_eye').value
        self._dead_zone      = self.get_parameter('dead_zone_px').value
        self._head_dead_zone = self.get_parameter('head_dead_zone_px').value
        self._ret_tmo        = self.get_parameter('return_timeout_sec').value
        self._hz             = self.get_parameter('track_hz').value
        self._min_det_score  = self.get_parameter('min_det_score').value
        self._head_pan_dir   = float(self.get_parameter('head_pan_dir').value)
        self._head_tilt_dir  = float(self.get_parameter('head_tilt_dir').value)
        self._eye_pan_dir    = float(self.get_parameter('eye_pan_dir').value)
        self._eye_tilt_dir   = float(self.get_parameter('eye_tilt_dir').value)
        self._max_step       = self.get_parameter('max_step_deg').value
        self._ema_alpha      = self.get_parameter('bbox_ema_alpha').value
        self._rothead        = self.get_parameter('rest_rothead').value
        self._neck           = self.get_parameter('rest_neck').value
        self._eye_lr         = self.get_parameter('rest_eye_lr').value
        self._eye_ud         = self.get_parameter('rest_eye_ud').value

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(String, '/face/tracks/left',    self._left_tracks_cb,  10)
        self.create_subscription(String, '/face/tracks/right',   self._right_tracks_cb, 10)
        self.create_subscription(Bool,   '/head_tracker/enable', self._enable_cb, latched_qos)
        self.create_subscription(String, '/face/identity',       self._identity_cb,     10)
        self.create_subscription(String, '/social_context',      self._social_context_cb, 10)
        # Synchronization with external one-off head commands (for example,
        # SoundScanBehaviour._aim_head_at_human() aims rothead using the OAK-D
        # BEFORE enable_head_tracker) — without this, self._rothead/_neck stay
        # stale (90°/40° from the last _return_to_rest), and the first step of
        # the P controller is computed from a wrong base, jerking the head
        # back towards the centre before it catches the real track. Found 2026-08-24.
        self.create_subscription(JointState, '/joint_command', self._external_joint_cb, 10)
        self._head_pub   = self.create_lifecycle_publisher(JointState, '/joint_command', 10)
        self._face_pub   = self.create_lifecycle_publisher(JointState, '/face_command',  10)
        # /head_tracker/face_locked — the single source of truth for "is a face
        # actually caught right now" (there is a fresh, non-stale bbox). Used
        # by behavior_manager_node to decide whether a repeated
        # sound_localization-based search attempt is needed on every utterance
        # (see project memory: face search retry). NOT latched — therefore we
        # publish False explicitly when the tracker is disabled (_enable_cb),
        # otherwise a subscriber would keep a stale True forever.
        self._locked_pub = self.create_lifecycle_publisher(Bool, '/head_tracker/face_locked', 10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._head_pub.on_activate(state)
        self._face_pub.on_activate(state)
        self._locked_pub.on_activate(state)
        self._timer = self.create_timer(1.0 / self._hz, self._tick)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        if self._timer:
            self.destroy_timer(self._timer)
            self._timer = None
        self._head_pub.on_deactivate(state)
        self._face_pub.on_deactivate(state)
        self._locked_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

    # ── Enable ────────────────────────────────────────────────────────────

    def _enable_cb(self, msg: Bool):
        with self._lock:
            self._enabled = msg.data
            if msg.data:
                # Reset stale timestamps so as not to log "no track for 348 s"
                # right after enabling (the tracker was disabled, the old timestamps remained)
                now = time.time()
                self._last_left_t  = now
                self._last_right_t = now
                self._at_rest      = False
        if not msg.data:
            self._return_to_rest()
            self._locked_pub.publish(Bool(data=False))
            self.get_logger().info('HeadTracker: disabled -> rest')
        else:
            self.get_logger().info('HeadTracker: enabled')

    def _external_joint_cb(self, msg: JointState):
        """Picks up rothead/neck from ANY source on /joint_command
        (including our own _publish_head() — harmless, same value) —
        keeps the internal state in sync with the real position so that the
        P controller does not compute a step from a stale base after someone
        else's one-off command (see the comment at the subscription)."""
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                if name == 'rothead':
                    self._rothead = _rad_to_deg(pos, center=90.0)
                elif name == 'neck':
                    self._neck = _rad_to_deg(pos, center=90.0)

    # ── Target interlocutor ───────────────────────────────────────────────

    def _social_context_cb(self, msg: String):
        """Get the current interlocutor from social_context."""
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        state     = data.get('state', 'idle')
        person_id = data.get('person_id')
        with self._lock:
            if state == 'interacting' and person_id:
                if person_id != self._target_person_id:
                    self._target_person_id = person_id
                    self._target_track_id  = None  # waiting for confirmation from identity
            else:
                self._target_person_id = None
                self._target_track_id  = None

    def _identity_cb(self, msg: String):
        """Update the interlocutor's target track_id from the recognition results."""
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        if not data.get('locked') or not data.get('is_known'):
            return
        with self._lock:
            if data.get('person_id') == self._target_person_id:
                new_track_id = data.get('track_id')
                if new_track_id != self._target_track_id:
                    # New binding — give the full correction budget again,
                    # do not inherit the "runaway" counter from the previous track.
                    self._runaway_streak  = 0
                    self._streak_ref_mag  = None
                self._target_track_id = new_track_id

    # ── Track callbacks ───────────────────────────────────────────────────

    def _left_tracks_cb(self, msg: String):
        if not self._enabled:
            return
        # Update "camera is alive" on ANY message — even if there is no face.
        # This prevents a false fallback to the right camera when the left is
        # active but temporarily does not see a face (bad angle, blinking).
        with self._lock:
            self._last_left_msg_t = time.time()
        bbox = self._extract_primary(msg)
        if bbox is None:
            return
        with self._lock:
            self._last_left_t = time.time()
            self._left_bbox   = self._ema(self._left_bbox, bbox)
            self._bbox_seq   += 1  # new detection -> the head may make ONE step (see _tick)
            self._at_rest     = False

    def _right_tracks_cb(self, msg: String):
        if not self._enabled:
            return
        bbox = self._extract_primary(msg)
        if bbox is None:
            return
        with self._lock:
            now = time.time()
            self._last_right_t = now
            self._right_bbox   = self._ema(self._right_bbox, bbox)
            # If the right is the leader (left unavailable) — this is also a new detection
            if (now - self._last_left_t) >= _STALE_SEC:
                self._bbox_seq += 1
            self._at_rest = False

    def _extract_primary(self, msg: String) -> list | None:
        try:
            tracks = json.loads(msg.data).get('tracks', [])
        except Exception:
            return None
        tracks = [t for t in tracks if t.get('det_score', 1.0) >= self._min_det_score]
        if not tracks:
            return None

        with self._lock:
            target_id = self._target_track_id

        if target_id is not None:
            # In a dialogue — follow only the interlocutor's track_id
            target = next((t for t in tracks if t.get('track_id') == target_id), None)
            if target is not None:
                return target.get('bbox')
            # The interlocutor's face is not in frame — do not move towards someone else's
            return None

        # We tried here to prefer continuity with the previous bbox over area
        # (live bug 2026-08-31: hijack by someone else's / a false object) —
        # reverted 2026-08-31: it did not fix the head drift itself (see
        # project_face_search_retry.md), extra complexity. Left as it
        # was — the largest face.
        return max(tracks, key=lambda t: _bbox_area(t.get('bbox', [0, 0, 0, 0]))).get('bbox')

    def _ema(self, current: list | None, new: list) -> list:
        if current is None:
            return list(new)
        a = self._ema_alpha
        return [a * n + (1.0 - a) * c for n, c in zip(new, current)]

    # ── Main tick ────────────────────────────────────────────────────────

    def _tick(self):
        with self._lock:
            if not self._enabled:
                return
            now         = time.time()
            left_fresh   = (now - self._last_left_t)  < _STALE_SEC
            right_fresh  = (now - self._last_right_t) < _STALE_SEC
            # "Left is absent" = it has sent NO messages at all (even without a face)
            # for longer than FALLBACK_SEC. If the left is active but does not see a face — that is NOT absence,
            # it is just a bad angle. We use _last_left_msg_t instead of _last_left_t.
            left_msg_ref = self._last_left_msg_t if self._last_left_msg_t > 0.0 \
                           else self._last_left_t
            left_absent  = (now - left_msg_ref) >= _FALLBACK_SEC
            left_bbox    = self._left_bbox  if left_fresh  else None
            right_bbox   = self._right_bbox if (right_fresh and left_absent) else None
            bbox_seq    = self._bbox_seq
            last_any    = max(self._last_left_t, self._last_right_t)

        self._locked_pub.publish(Bool(data=bool(left_bbox is not None or right_bbox is not None)))

        if left_bbox is None and right_bbox is None:
            elapsed = time.time() - last_any
            if elapsed > self._ret_tmo:
                if not self._at_rest:
                    self.get_logger().info(
                        f'HeadTracker: no track for {elapsed:.0f} s -> rest '
                        f'(rothead={self._rothead:.1f}°, neck={self._neck:.1f}°)')
                    self._return_to_rest()   # resets angles, clears bboxes, publishes
                else:
                    # Already at rest — just hold the position (no log and no state reset)
                    self._publish_head()
                    self._publish_eyes('left')
            return

        # Leader — the left if available
        if left_bbox is not None:
            lead_bbox  = left_bbox
            new_side   = 'left'
        else:
            lead_bbox  = right_bbox
            new_side   = 'right'

        # Side change — log
        if new_side != self._active_side:
            self.get_logger().info(
                f'HeadTracker: {self._active_side} -> {new_side} (EYE_SYNC active) '
                f'rothead={self._rothead:.1f}° neck={self._neck:.1f}°')
            self._active_side = new_side

        norm_x, norm_y = self._bbox_to_norm(lead_bbox)

        # Diagnostics: track_id changed (for example, identity_manager has just
        # confirmed the identity and issued target_track_id) AND the new bbox
        # ended up at the edge of the frame, although a moment ago tracking
        # was fine — suspicion of a switch to a DIFFERENT, wrong track instead
        # of the already well-centered face. See live bug 2026-08-31.
        if self._target_track_id != self._prev_target_track_id:
            if abs(norm_x) > 0.5 or abs(norm_y) > 0.5:
                self.get_logger().warn(
                    f'HeadTracker: SUSPICIOUS track_id JUMP '
                    f'{self._prev_target_track_id} → {self._target_track_id}, '
                    f'new bbox at the edge of the frame offset=({norm_x:+.2f},{norm_y:+.2f}) — '
                    f'possibly switched to the wrong face')
            self._prev_target_track_id = self._target_track_id

        # ── Head: exactly ONE step per new detection, not per tick ──
        # (see the comment at self._bbox_seq in __init__ — previously there was
        # a head_stale < max_stale check here, which allowed up to 5 repeated
        # steps on the same stale offset between two detections).
        if bbox_seq != self._last_stepped_seq:
            self._step_head(norm_x, norm_y)
            self._last_stepped_seq = bbox_seq

        # ── Eyes: one set of joint names, EYE_SYNC mirrors the other ──
        self._set_eye(norm_x, norm_y)

        # Diagnostics: once a second show where and why the head is moving —
        # previously the P controller logged NOTHING between "enabled" and "no
        # track -> rest", drift away from the real face was not visible in the logs.
        # (NOTE: the actual throttle interval is _TRACK_LOG_INTERVAL_SEC = 3.0 s.)
        now_mono = time.monotonic()
        if now_mono - self._last_track_log_mono >= self._TRACK_LOG_INTERVAL_SEC:
            self._last_track_log_mono = now_mono
            self.get_logger().info(
                f'HeadTracker: tracking ({self._active_side}, track_id='
                f'{self._target_track_id}) offset=({norm_x:+.2f},{norm_y:+.2f}) '
                f'→ rothead={self._rothead:.1f}° neck={self._neck:.1f}° '
                f'eye_lr={self._eye_lr:.1f}°')

        self._publish_head()
        self._publish_eyes(self._active_side)

    # ── P controller ──────────────────────────────────────────────────────

    def _bbox_to_norm(self, bbox: list) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        norm_x = ((x1 + x2) / 2.0 - self._img_w / 2.0) / (self._img_w / 2.0)
        norm_y = ((y1 + y2) / 2.0 - self._img_h / 2.0) / (self._img_h / 2.0)
        return norm_x, norm_y

    def _step_head(self, norm_x: float, norm_y: float):
        head_dz_x = self._head_dead_zone / (self._img_w / 2.0)
        head_dz_y = self._head_dead_zone / (self._img_h / 2.0)
        step_h, step_v = 0.0, 0.0
        if abs(norm_x) >= head_dz_x:
            step_h = _clamp(
                self._head_pan_dir * norm_x * self._max_step * self._gain_head,
                -self._max_step, self._max_step)
        if abs(norm_y) >= head_dz_y:
            step_v = _clamp(
                self._head_tilt_dir * norm_y * self._max_step * self._gain_head,
                -self._max_step, self._max_step)

        # Runaway safeguard: the offset (error of the face position in the
        # frame) must DECREASE as the head turns towards the face. If several
        # detections in a row fail to do so — something is wrong (a stale/
        # incorrect bbox, or the head physically cannot compensate for the
        # shift) — stop accumulating the correction instead of driving to
        # the servo's hardware limit.
        mag = abs(norm_x) + abs(norm_y)
        if self._streak_ref_mag is None:
            self._streak_ref_mag = mag
        if mag <= self._streak_ref_mag - self._RUNAWAY_IMPROVE_EPS:
            # Genuine progress relative to the start of the series — reset
            # the counter and move the baseline to the current (now better) value.
            self._runaway_streak = 0
            self._streak_ref_mag = mag
        else:
            self._runaway_streak += 1

        if self._runaway_streak >= self._RUNAWAY_MAX_STREAK:
            if self._runaway_streak == self._RUNAWAY_MAX_STREAK:
                self.get_logger().warn(
                    f'HeadTracker: RUNAWAY — offset has not decreased for '
                    f'{self._runaway_streak} detections in a row '
                    f'(offset=({norm_x:+.2f},{norm_y:+.2f}), track_id='
                    f'{self._target_track_id}) — stopping accumulation '
                    f'of head correction')
            return

        self._rothead = _clamp(self._rothead + step_h, *self._rothead_range)
        self._neck    = _clamp(self._neck    + step_v, *self._neck_range)

    def _set_eye(self, norm_x: float, norm_y: float):
        eye_dz_x = self._dead_zone / (self._img_w / 2.0)
        eye_dz_y = self._dead_zone / (self._img_h / 2.0)
        if abs(norm_x) < eye_dz_x and abs(norm_y) < eye_dz_y:
            return
        half_h = (self._eye_lr_range[1] - self._eye_lr_range[0]) / 2.0
        half_v = (self._eye_ud_range[1] - self._eye_ud_range[0]) / 2.0
        ctr_lr = (self._eye_lr_range[0] + self._eye_lr_range[1]) / 2.0
        ctr_ud = (self._eye_ud_range[0] + self._eye_ud_range[1]) / 2.0
        self._eye_lr = _clamp(
            ctr_lr + self._eye_pan_dir  * norm_x * half_h * self._gain_eye,
            *self._eye_lr_range)
        self._eye_ud = _clamp(
            ctr_ud + self._eye_tilt_dir * norm_y * half_v * self._gain_eye,
            *self._eye_ud_range)

    # ── Rest ──────────────────────────────────────────────────────────────

    def _return_to_rest(self):
        self._rothead = self.get_parameter('rest_rothead').value
        self._neck    = self.get_parameter('rest_neck').value
        self._eye_lr  = self.get_parameter('rest_eye_lr').value
        self._eye_ud  = self.get_parameter('rest_eye_ud').value
        with self._lock:
            self._left_bbox        = None
            self._right_bbox       = None
            self._target_track_id  = None
            self._target_person_id = None
            self._at_rest          = True
            self._runaway_streak   = 0
            self._streak_ref_mag   = None
        self._publish_head()
        # At rest we publish the left — EYE_SYNC mirrors the right
        self._publish_eyes('left')

    # ── Publishers ────────────────────────────────────────────────────────

    def _publish_head(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = ['rothead', 'neck']
        msg.position = [
            _deg_to_rad(self._rothead, center=90.0),
            _deg_to_rad(self._neck,    center=90.0),
        ]
        self._head_pub.publish(msg)

    def _publish_eyes(self, side: str):
        """Publishes only one eye — EYE_SYNC in arduino_comm_node mirrors the other."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        if side == 'left':
            msg.name = ['eye_lr_L', 'eye_ud_L']
        else:
            msg.name = ['eye_lr_R', 'eye_ud_R']
        msg.position = [
            _deg_to_rad(self._eye_lr, center=90.0),
            _deg_to_rad(self._eye_ud, center=90.0),
        ]
        self._face_pub.publish(msg)




def main():
    rclpy.init()
    node = VisionHeadTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
