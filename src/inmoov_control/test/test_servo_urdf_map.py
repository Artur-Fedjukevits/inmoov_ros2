"""Tests for servo_urdf_map (no ROS, no hardware).

  python3 -m pytest src/inmoov_control/test/test_servo_urdf_map.py -v
"""
import math
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inmoov_control.servo_urdf_map import (JointMap, ServoUrdfMap,  # noqa: E402
                                           servo_deg_to_rad, servo_rad_to_deg)

HERE = os.path.dirname(__file__)
PKG = os.path.dirname(HERE)
MAP = os.path.join(PKG, 'config', 'servo_urdf_map.yaml')
URDF = os.path.join(PKG, '..', 'inmoov_description', 'description', 'inmoov_i2.urdf.xacro')


@pytest.fixture(scope='module')
def m():
    return ServoUrdfMap.load(MAP)


def _servo_tables():
    """(name, rest, min, max) from the comments of the Arduino node tables."""
    out = {}
    for f in ('arduino_right_node.py', 'arduino_left_node.py'):
        src = open(os.path.join(PKG, 'inmoov_control', f), encoding='utf-8').read()
        for n, rest, mn, mx in re.findall(
                r"\('(\w+)',\s*90,\s*(\d+)\),\s*#.*?rest=\s*\d+,\s*min=\s*(\d+),\s*max=\s*(\d+)", src):
            out[n] = (int(rest), int(mn), min(int(mx), 180))
    return out


def _urdf_revolute_non_mimic():
    src = open(URDF, encoding='utf-8').read()
    out = {}
    for name, body in re.findall(r'<joint name="([^"]+)" type="revolute">(.*?)</joint>', src, re.S):
        if '<mimic' in body:
            continue
        lo, hi = re.search(r'lower="([-\d.e]+)"\s+upper="([-\d.e]+)"', body).groups()
        out[name] = (float(lo), float(hi))
    return out


def test_all_41_urdf_joints_mapped(m):
    urdf = _urdf_revolute_non_mimic()
    assert len(urdf) == 41
    assert set(m.by_urdf) == set(urdf)


def test_every_servo_mapped_or_listed(m):
    servos = _servo_tables()
    assert set(m.by_servo) | set(m.unmapped_servos) == set(servos)
    assert not set(m.by_servo) & set(m.unmapped_servos)


def test_ranges_match_firmware_and_urdf(m):
    servos = _servo_tables()
    urdf = _urdf_revolute_non_mimic()
    for j in m.joints:
        rest, mn, mx = servos[j.servo]
        assert (j.servo_min, j.servo_max, j.servo_rest) == (mn, mx, rest), j.servo
        assert j.urdf_limits == pytest.approx(urdf[j.urdf], abs=1e-6), j.urdf


def test_roundtrip_inside_servo_range(m):
    for j in m.joints:
        for k in range(21):
            deg = j.servo_min + (j.servo_max - j.servo_min) * k / 20
            rad = j.servo_to_urdf(deg, clamp=False)
            assert j.urdf_to_servo(rad, clamp=False) == pytest.approx(deg, abs=1e-6), j.servo


def test_clamping(m):
    for j in m.joints:
        lo, hi = j.urdf_limits
        for rad in (lo - 1.0, hi + 1.0):
            deg = j.urdf_to_servo(rad)
            assert j.servo_min <= deg <= j.servo_max
        for deg in (-50, 250):
            assert lo - 1e-9 <= j.servo_to_urdf(deg) <= hi + 1e-9


def test_servo_rad_convention():
    assert servo_deg_to_rad(90) == 0.0
    assert servo_rad_to_deg(servo_deg_to_rad(37.0)) == pytest.approx(37.0)


def test_convert_via_joint_state_convention(m):
    name, rad = m.servo_rad_to_urdf('rothead', servo_deg_to_rad(90))   # rest -> URDF 0
    assert name == 'i01_head_rothead_joint' and rad == pytest.approx(0.0)
    sname, srad = m.urdf_to_servo_rad('i01_head_rothead_joint', math.radians(20))
    assert sname == 'rothead' and servo_rad_to_deg(srad) == pytest.approx(110.0)
    assert m.servo_rad_to_urdf('lowstom', 0.0) is None
    assert m.urdf_to_servo_rad('i02_leftHand_indexMid_joint', 0.1) is None   # mimic


def test_nonlinear_and_inverted_table():
    j = JointMap('x', 'x_joint', [0, 180], 90, [[0, 1.0], [90, 0.0], [180, -0.2]])
    assert j.direction == -1
    assert j.servo_to_urdf(45) == pytest.approx(0.5)
    assert j.urdf_to_servo(-0.1) == pytest.approx(135)
    assert j.urdf_to_servo(0.5) == pytest.approx(45)


def test_bad_tables_rejected():
    with pytest.raises(ValueError):
        JointMap('x', 'x', [0, 180], 90, [[10, 0.0]])
    with pytest.raises(ValueError):
        JointMap('x', 'x', [0, 180], 90, [[0, 0.0], [90, 1.0], [180, 0.5]])   # not monotonic
    with pytest.raises(ValueError):
        JointMap('x', 'x', [0, 180], 90, [[0, 0.0], [0, 1.0]])


def test_velocity(m):
    # direct drive: 1 servo degree = 1 joint degree -> same rad/s
    v = m.urdf_velocity_to_servo('i01_head_rothead_joint', 0.0, 0.5)
    assert v == pytest.approx(0.5)


def test_save_load_roundtrip(m, tmp_path):
    p = tmp_path / 'map.yaml'
    m.save(str(p))
    m2 = ServoUrdfMap.load(str(p))
    for a, b in zip(m.joints, m2.joints):
        assert a.to_dict() == b.to_dict()
