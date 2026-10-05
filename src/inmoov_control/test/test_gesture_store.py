"""Tests for gesture_store (no ROS, no hardware).

  python3 -m pytest src/inmoov_control/test/test_gesture_store.py -v
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inmoov_control import gesture_store as gs  # noqa: E402
from inmoov_control.servo_urdf_map import ServoUrdfMap  # noqa: E402

MAP = os.path.join(os.path.dirname(__file__), '..', 'config', 'servo_urdf_map.yaml')


@pytest.fixture(scope='module')
def m():
    return ServoUrdfMap.load(MAP)


def _msg(**over):
    g = {'format': 'inmoov_gesture', 'version': 1, 'kind': 'body', 'name': 'wave',
         'robot': 'inmoov_i2', 'units': 'urdf_rad', 'created': '2026-10-05T12:00:00Z',
         'positions': {'left_elbow_x_joint': 0.5}}
    g.update(over)
    return json.dumps(g)


def test_parse_ok():
    assert gs.parse(_msg(), 'body')['name'] == 'wave'


@pytest.mark.parametrize('bad', [
    _msg(format='x'), _msg(version=2), _msg(kind='face'), _msg(units='deg'),
    _msg(name='///'), _msg(name=5), _msg(positions={}),
    _msg(positions={'a': 'x'}), _msg(positions={'a': True}),
    '{', '[]',
])
def test_parse_rejects(bad):
    with pytest.raises(gs.GestureError):
        gs.parse(bad, 'body')


def test_parse_rejects_nan_and_huge():
    with pytest.raises(gs.GestureError):
        gs.parse(_msg().replace('0.5', 'NaN'), 'body')
    with pytest.raises(gs.GestureError):
        gs.parse(_msg(name='a' * gs.MAX_BYTES), 'body')


def test_safe_name_blocks_traversal():
    assert '/' not in gs.safe_name('../../etc/passwd')
    assert gs.safe_name('Привет мир') == 'Привет_мир'


def test_save_load_overwrite_list(tmp_path):
    g = gs.parse(_msg(name='Привет ../x'), 'body')
    p = gs.save(g, str(tmp_path))
    assert os.path.dirname(p) == str(tmp_path / 'body')
    assert gs.load('body', 'Привет ../x', str(tmp_path)) == g
    gs.save(dict(g, positions={'left_elbow_x_joint': 1.0}), str(tmp_path))
    assert gs.load('body', g['name'], str(tmp_path))['positions'] == {'left_elbow_x_joint': 1.0}
    assert gs.list_gestures('body', str(tmp_path)) == [g['name']]
    assert gs.list_gestures('face', str(tmp_path)) == []
    assert not [f for f in os.listdir(tmp_path / 'body') if f.endswith('.tmp')]


def test_servo_deg_recomputed(m):
    j = m.joints[0]
    g = gs.parse(_msg(positions={j.urdf: 0.0, 'mimic_joint': 0.1},
                      servo_deg={'client': 1.0}), 'body')
    assert gs.add_servo_deg(g, m) == ['mimic_joint']
    assert list(g['servo_deg']) == [j.servo]
    assert 'mimic_joint' in g['positions']
