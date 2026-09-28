"""
servo_urdf_map.py — conversion between servo angles and URDF joint angles.

Pure Python (no rclpy), shared by urdf_bridge_node, tests and test/mock_rosbridge_robot.py.

Three angle conventions meet here:

  servo_deg  — what the firmware writes to the servo, 0..180 (SET_SERVOS byte).
  servo_rad  — the /joint_cmd, /joint_states convention of inmoov_control:
               servo_rad = (servo_deg - 90) * pi / 180, names are servo names
               (thumb_R, bicep_L, eyelid_R_Upper, ...).
  urdf_rad   — joint angle of inmoov_i2.urdf (inmoov_description), names are
               URDF joint names (i02_rightHand_ThumbMid_joint, ...).

Each mapped joint has a calibration table of points [servo_deg, urdf_rad]
(at least 2, strictly monotonic in both columns). Between points the mapping is
linear; outside it is extrapolated with the slope of the end segment, and the
result is clamped to the servo range / URDF limits. Two points = the classic
offset/direction/scale; more points = a non-linear linkage (fingers, face).

The table lives in config/servo_urdf_map.yaml (shipped) and can be overridden by
the user file written by the calibration (~/.config/inmoov/servo_urdf_map.yaml,
or $INMOOV_SERVO_URDF_MAP).

Author: Artur Fedjukevits
Assisted by: Claude (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import math
import os
from bisect import bisect_right

import yaml

USER_MAP_PATH = os.path.expanduser(
    os.environ.get('INMOOV_SERVO_URDF_MAP', '~/.config/inmoov/servo_urdf_map.yaml'))

SERVO_CENTER_DEG = 90.0


def servo_deg_to_rad(deg: float) -> float:
    return (deg - SERVO_CENTER_DEG) * math.pi / 180.0


def servo_rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi + SERVO_CENTER_DEG


class JointMap:
    """One servo <-> one URDF joint."""

    def __init__(self, servo, urdf, servo_range, servo_rest, points,
                 urdf_limits=None, verified=False, note='', board=''):
        self.servo = servo
        self.board = board
        self.urdf = urdf
        self.servo_min, self.servo_max = float(servo_range[0]), float(servo_range[1])
        self.servo_rest = float(servo_rest)
        self.urdf_limits = (tuple(float(v) for v in urdf_limits)
                            if urdf_limits else None)
        self.verified = bool(verified)
        self.note = note
        self.set_points(points)

    # ------------------------------------------------------------------ table
    def set_points(self, points):
        pts = sorted((float(s), float(u)) for s, u in points)
        if len(pts) < 2:
            raise ValueError(f'{self.servo}: need at least 2 calibration points')
        ds = [b[0] - a[0] for a, b in zip(pts, pts[1:])]
        du = [b[1] - a[1] for a, b in zip(pts, pts[1:])]
        if any(d <= 0 for d in ds):
            raise ValueError(f'{self.servo}: servo_deg of the points must be distinct')
        if not (all(d > 0 for d in du) or all(d < 0 for d in du)):
            raise ValueError(f'{self.servo}: urdf_rad must be strictly monotonic '
                             f'(got {pts})')
        self.points = pts
        self._s = [p[0] for p in pts]
        self._u = [p[1] for p in pts]
        self.direction = 1 if du[0] > 0 else -1
        # the same table sorted by urdf for the inverse
        inv = sorted((u, s) for s, u in pts)
        self._iu = [p[0] for p in inv]
        self._is = [p[1] for p in inv]

    @staticmethod
    def _interp(x, xs, ys):
        i = bisect_right(xs, x) - 1
        i = max(0, min(len(xs) - 2, i))
        x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def slope(self, servo_deg: float) -> float:
        """d(urdf_rad)/d(servo_deg) at this servo angle."""
        i = bisect_right(self._s, servo_deg) - 1
        i = max(0, min(len(self._s) - 2, i))
        return (self._u[i + 1] - self._u[i]) / (self._s[i + 1] - self._s[i])

    # ------------------------------------------------------------ conversion
    def clamp_servo(self, deg: float) -> float:
        return min(self.servo_max, max(self.servo_min, deg))

    def clamp_urdf(self, rad: float) -> float:
        if self.urdf_limits is None:
            return rad
        return min(self.urdf_limits[1], max(self.urdf_limits[0], rad))

    def servo_to_urdf(self, servo_deg: float, clamp: bool = True) -> float:
        if clamp:
            servo_deg = self.clamp_servo(servo_deg)
        rad = self._interp(servo_deg, self._s, self._u)
        return self.clamp_urdf(rad) if clamp else rad

    def urdf_to_servo(self, urdf_rad: float, clamp: bool = True) -> float:
        if clamp:
            urdf_rad = self.clamp_urdf(urdf_rad)
        deg = self._interp(urdf_rad, self._iu, self._is)
        return self.clamp_servo(deg) if clamp else deg

    def to_dict(self) -> dict:
        d = {
            'servo': self.servo,
            'urdf': self.urdf,
            'board': self.board,
            'servo_range': [self.servo_min, self.servo_max],
            'servo_rest': self.servo_rest,
            'points': [[round(s, 3), round(u, 6)] for s, u in self.points],
            'verified': self.verified,
        }
        if self.urdf_limits:
            d['urdf_limits'] = [round(v, 6) for v in self.urdf_limits]
        if self.note:
            d['note'] = self.note
        return d


class ServoUrdfMap:

    def __init__(self, joints, unmapped_servos=(), source=''):
        self.joints = list(joints)
        self.unmapped_servos = list(unmapped_servos)
        self.source = source
        self.by_servo = {j.servo: j for j in self.joints}
        self.by_urdf = {j.urdf: j for j in self.joints}
        if len(self.by_servo) != len(self.joints) or len(self.by_urdf) != len(self.joints):
            raise ValueError('servo_urdf_map: duplicate servo or urdf joint name')

    # ------------------------------------------------------------------- I/O
    @classmethod
    def from_dict(cls, data: dict, source: str = '') -> 'ServoUrdfMap':
        joints = [JointMap(j['servo'], j['urdf'], j['servo_range'], j['servo_rest'],
                           j['points'], j.get('urdf_limits'), j.get('verified', False),
                           j.get('note', ''), j.get('board', ''))
                  for j in data['joints']]
        return cls(joints, data.get('unmapped_servos', []), source)

    @classmethod
    def load(cls, path: str) -> 'ServoUrdfMap':
        with open(path, encoding='utf-8') as f:
            return cls.from_dict(yaml.safe_load(f), path)

    @classmethod
    def load_default(cls, shipped_path: str) -> 'ServoUrdfMap':
        """User calibration if it exists, otherwise the shipped table.

        Joints missing from the user file are taken from the shipped one, so a
        partial user file (only calibrated joints) is fine.
        """
        base = cls.load(shipped_path)
        if not os.path.exists(USER_MAP_PATH):
            return base
        user = cls.load(USER_MAP_PATH)
        merged = [user.by_servo.get(j.servo, j) for j in base.joints]
        return cls(merged, base.unmapped_servos, f'{USER_MAP_PATH} + {shipped_path}')

    def to_dict(self) -> dict:
        return {
            'version': 1,
            'unmapped_servos': self.unmapped_servos,
            'joints': [j.to_dict() for j in self.joints],
        }

    def save(self, path: str = USER_MAP_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write('# servo_urdf_map.yaml — written by urdf_bridge calibration\n')
            f.write('# points: [servo_deg, urdf_rad]\n')
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)
        os.replace(tmp, path)

    # ------------------------------------------------------------ conversion
    def servo_rad_to_urdf(self, servo_name: str, servo_rad: float):
        """/joint_states (servo name, servo rad) -> (urdf name, urdf rad) or None."""
        j = self.by_servo.get(servo_name)
        if j is None:
            return None
        return j.urdf, j.servo_to_urdf(servo_rad_to_deg(servo_rad))

    def urdf_to_servo_rad(self, urdf_name: str, urdf_rad: float):
        """URDF command -> (servo name, servo rad for /joint_cmd) or None."""
        j = self.by_urdf.get(urdf_name)
        if j is None:
            return None
        return j.servo, servo_deg_to_rad(j.urdf_to_servo(urdf_rad))

    def urdf_velocity_to_servo(self, urdf_name: str, urdf_rad: float, urdf_vel: float) -> float:
        """URDF rad/s -> servo rad/s (magnitude) at the given position."""
        j = self.by_urdf[urdf_name]
        k = abs(j.slope(j.urdf_to_servo(urdf_rad)))   # urdf_rad per servo_deg
        if k == 0:
            return 0.0
        return abs(urdf_vel) / k * math.pi / 180.0
