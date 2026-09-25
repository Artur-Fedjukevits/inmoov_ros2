"""
test_camera_path.py — eye-camera frame path without cameras or models.

  face_capture: MJPEG passthrough publishes the camera's JPEG untouched with
  frame_id; a corrupt frame counts as a miss; the fallback republishes the other
  eye's frame with ITS stamp + frame_id; the flip path re-encodes.
  face_detection: stores JPEG bytes, decodes only in _detect.

Run:
  cd /home/artur/ros2_ws && source install/setup.bash
  python3 -m pytest src/inmoov_vision/test/test_camera_path.py -v
"""

import os
import sys
import threading
import types

import cv2
import numpy as np
import pytest
import rclpy
from builtin_interfaces.msg import Time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inmoov_vision.face_capture_node import FaceCaptureNode  # noqa: E402
from inmoov_vision.face_detection_node import FaceDetectionNode  # noqa: E402


@pytest.fixture(scope='module', autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def _jpeg(value=128) -> bytes:
    img = np.full((48, 64, 3), value, np.uint8)
    return cv2.imencode('.jpg', img)[1].tobytes()


class FakeCap:
    def __init__(self, frames):
        self.frames = list(frames)   # items: bytes (raw MJPEG) / ndarray / None (fail)

    def isOpened(self):
        return True

    def read(self):
        f = self.frames.pop(0)
        if f is None:
            return False, None
        if isinstance(f, bytes):
            return True, np.frombuffer(f, np.uint8).reshape(1, -1)
        return True, f

    def release(self):
        pass


class Pub:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)


@pytest.fixture
def cap_node():
    n = FaceCaptureNode()
    n._pubs = {'left': Pub(), 'right': Pub()}
    n._flips = {'left': False, 'right': False}
    n._passthrough = {'left': True, 'right': True}
    n._reopen_after = 100
    n._quality = 85
    yield n
    n.destroy_node()


def test_passthrough_publishes_camera_jpeg(cap_node):
    j = _jpeg()
    cap_node._caps = {'left': FakeCap([j])}
    cap_node._grab('left', Time(sec=5))
    m = cap_node._pubs['left'].msgs[0]
    assert bytes(m.data) == j and m.header.frame_id == 'eye_left' and m.header.stamp.sec == 5


def test_corrupt_frame_falls_back_to_other_eye_with_its_stamp(cap_node):
    cap_node._caps = {'left': FakeCap([b'garbage-not-jpeg']), 'right': FakeCap([_jpeg(50)])}
    cap_node._grab('right', Time(sec=7))          # right is fine
    cap_node._grab('left', Time(sec=8))           # left frame corrupt → mirror of right
    assert cap_node._fails['left'] == 1
    m = cap_node._pubs['left'].msgs[0]
    assert m.header.frame_id == 'eye_right'       # consumers see the real source
    assert m.header.stamp.sec == 7                # NOT re-stamped as fresh
    assert bytes(m.data) == bytes(cap_node._pubs['right'].msgs[0].data)


def test_flip_path_reencodes(cap_node):
    img = np.zeros((48, 64, 3), np.uint8)
    img[:, :10] = 255                             # white stripe on the left
    cap_node._passthrough['left'] = False
    cap_node._flips['left'] = True
    cap_node._caps = {'left': FakeCap([img])}
    cap_node._grab('left', Time(sec=1))
    out = cv2.imdecode(np.frombuffer(bytes(cap_node._pubs['left'].msgs[0].data), np.uint8),
                       cv2.IMREAD_COLOR)
    assert out[:, -5:].mean() > 200 and out[:, :5].mean() < 50   # stripe moved right


def test_face_detection_decodes_lazily():
    got = {}
    pub = Pub()
    stub = types.SimpleNamespace(
        _app=types.SimpleNamespace(get=lambda frame: got.setdefault('shape', frame.shape) and []),
        _pub=pub, _generation=3, _no_face_since=0.0, _busy=True,
        get_logger=lambda: types.SimpleNamespace(debug=lambda *a: None, error=lambda *a: None))
    FaceDetectionNode._detect(stub, _jpeg(), 12.5, 3)
    assert got['shape'] == (48, 64, 3) and len(pub.msgs) == 1 and stub._busy is False

    pub.msgs.clear()
    FaceDetectionNode._detect(stub, b'not a jpeg', 13.0, 3)   # undecodable → nothing
    assert pub.msgs == [] and stub._busy is False

    FaceDetectionNode._detect(stub, _jpeg(), 14.0, 2)          # stale generation
    assert pub.msgs == []


def test_face_detection_callback_stores_bytes_only():
    stub = types.SimpleNamespace(_lc_active=True, _frame_lock=threading.Lock(),
                                 _latest_frame=None, _latest_stamp=0.0, _last_frame_t=0.0)
    msg = types.SimpleNamespace(data=b'\xff\xd8jpeg',
                                header=types.SimpleNamespace(stamp=Time(sec=3, nanosec=500000000)))
    FaceDetectionNode._frame_callback(stub, msg)
    assert stub._latest_frame == b'\xff\xd8jpeg' and stub._latest_stamp == 3.5
