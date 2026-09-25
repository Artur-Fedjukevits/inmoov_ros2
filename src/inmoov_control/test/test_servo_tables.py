"""
test_servo_tables.py — the Python joint tables must match the firmware (no hardware needed).

arduino_{left,right}_node.py send one byte per servo in BODY_JOINTS + FACE_JOINTS
order and start from their rest values; the firmware accepts SET_SERVOS only with
exactly SERVO_TOTAL_COUNT bytes and treats every byte as a literal angle. So:
  1. same servo count,
  2. same order (Python name ↔ ServoIndex enum name),
  3. same rest angle as the firmware servo table.
min/max live only in the firmware (Python keeps them in comments), so they are
not compared.

Run:
  cd /home/artur/ros2_ws
  python3 -m pytest src/inmoov_control/test/test_servo_tables.py -v
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inmoov_control.arduino_left_node import ArduinoLeftNode    # noqa: E402
from inmoov_control.arduino_right_node import ArduinoRightNode  # noqa: E402

_ARDUINO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Arduino'))


def _strip_comments(src: str) -> str:
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
    return re.sub(r'//[^\n]*', '', src)


def _enum_names(header: str) -> list[str]:
    body = re.search(r'enum\s+ServoIndex\s*\{(.*?)\}', _strip_comments(header), re.S).group(1)
    names = [n.split('=')[0].strip() for n in body.split(',')]
    names = [n for n in names if n]
    assert names[-1] == 'SERVO_TOTAL_COUNT'
    return names[:-1]


def _table_rows(ino: str) -> list[list[str]]:
    table = re.search(
        r'SmoothServo\s+servos\[SERVO_TOTAL_COUNT\]\s*=\s*\{(.*?)\n\};',
        _strip_comments(ino), re.S).group(1)
    return [[f.strip() for f in row.split(',')] for row in re.findall(r'\{([^{}]*)\}', table)]


def _norm(name: str) -> str:
    return name.lower().replace('_', '')


def _same_joint(enum_name: str, py_name: str) -> bool:
    """IDX_EYE_LR ↔ eye_lr_L, IDX_THUMB ↔ thumb_R, IDX_EYELID_L_UPPER ↔ eyelid_L_Upper."""
    e, p = _norm(enum_name[len('IDX_'):]), _norm(py_name)
    return e == p or (p[-1] in 'lr' and e == p[:-1])


# rest is field 2 in the left struct ({current, target, rest, ...}) and field 3 in
# the right one ({Servo(), current, target, rest, ...}) — see InMoov{Left,Right}.h.
CASES = [
    ('InMoovLeft',  ArduinoLeftNode,  2),
    ('InMoovRight', ArduinoRightNode, 3),
]


def _load(board):
    base = os.path.join(_ARDUINO_DIR, board, board)
    with open(base + '.h') as f:
        header = f.read()
    with open(base + '.ino') as f:
        ino = f.read()
    return _enum_names(header), _table_rows(ino)


@pytest.mark.parametrize('board,node_cls,rest_field', CASES)
def test_servo_count(board, node_cls, rest_field):
    enum, rows = _load(board)
    joints = node_cls.BODY_JOINTS + node_cls.FACE_JOINTS
    assert len(rows) == len(enum), f'{board}: servos[] rows ≠ ServoIndex entries'
    assert len(joints) == len(enum), f'{board}: Python sends {len(joints)} bytes, firmware expects {len(enum)}'


@pytest.mark.parametrize('board,node_cls,rest_field', CASES)
def test_servo_order(board, node_cls, rest_field):
    enum, _ = _load(board)
    joints = node_cls.BODY_JOINTS + node_cls.FACE_JOINTS
    for i, ((py_name, _, _), enum_name) in enumerate(zip(joints, enum)):
        assert _same_joint(enum_name, py_name), f'{board}[{i}]: Python {py_name} ↔ firmware {enum_name}'


@pytest.mark.parametrize('board,node_cls,rest_field', CASES)
def test_servo_rest(board, node_cls, rest_field):
    _, rows = _load(board)
    joints = node_cls.BODY_JOINTS + node_cls.FACE_JOINTS
    for i, ((py_name, _, py_rest), row) in enumerate(zip(joints, rows)):
        fw_rest = int(row[rest_field])
        assert py_rest == fw_rest, f'{board}[{i}] {py_name}: Python rest {py_rest} ≠ firmware {fw_rest}'
