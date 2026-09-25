"""
test_diagnostics.py — /diagnostics levels of the hardware nodes (no hardware needed).

Each node's _diagnose() runs on a stub with the relevant state:
  audio_source (microphone), arduino link, eye cameras, OAK pipeline.
Lives in inmoov_voice/test but covers all four — the logic is the same pattern.

Run:
  cd ~/ros2_ws && source install/setup.bash
  python3 -m pytest src/inmoov_voice/test/test_diagnostics.py -v
"""

import time
import types

from diagnostic_msgs.msg import DiagnosticStatus as DS

from inmoov_voice.audio_source_node import AudioSourceNode
from inmoov_control.arduino_comm_node import ArduinoCommNode
from inmoov_control.joint_arbiter import JointArbiter
from inmoov_vision.face_capture_node import FaceCaptureNode
from inmoov_vision.oak_node import OakNode


class Stat:
    def __init__(self):
        self.level, self.message, self.values = None, '', {}

    def summary(self, level, message):
        self.level, self.message = level, message

    def add(self, key, value):
        self.values[key] = value


def _run(fn, stub, *args):
    st = Stat()
    fn(stub, st, *args)
    return st


# ── microphone ──────────────────────────────────────────────────────────────

def _mic(**kw):
    s = types.SimpleNamespace(_stream_running=True, _stream=object(), _last_ok=time.time(),
                              _zero_streak=0, _watchdog_sec=5.0,
                              _ZERO_WARN_AFTER=AudioSourceNode._ZERO_WARN_AFTER)
    s.__dict__.update(kw)
    return s


def test_microphone_levels():
    assert _run(AudioSourceNode._diagnose, _mic()).level == DS.OK
    assert _run(AudioSourceNode._diagnose, _mic(_zero_streak=500)).level == DS.WARN
    assert _run(AudioSourceNode._diagnose, _mic(_last_ok=time.time() - 10)).level == DS.ERROR
    assert _run(AudioSourceNode._diagnose, _mic(_stream_running=False,
                                                _last_ok=0.0)).message == 'inactive'


# ── arduino link ────────────────────────────────────────────────────────────

def _board(**kw):
    s = types.SimpleNamespace(_ser=types.SimpleNamespace(is_open=True), _running=True,
                              _failsafe=False, _sleeping=False, _last_rx_t=time.monotonic(),
                              _serial_port='/dev/x', _arbiter=JointArbiter())
    s.__dict__.update(kw)
    return s


def test_arduino_link_levels():
    assert _run(ArduinoCommNode._diagnose, _board()).level == DS.OK
    assert _run(ArduinoCommNode._diagnose, _board(_failsafe=True)).level == DS.WARN
    assert _run(ArduinoCommNode._diagnose, _board(_ser=None)).level == DS.ERROR
    assert _run(ArduinoCommNode._diagnose, _board(_running=False, _ser=None)).message == 'inactive'


# ── eye cameras ─────────────────────────────────────────────────────────────

def _cams(left_age, right_age, active=True):
    now = time.monotonic()
    return types.SimpleNamespace(
        _active=active, _devs={'left': 'L', 'right': 'R'},
        _passthrough={'left': True, 'right': True}, _fails={'left': 0, 'right': 0},
        _n_frames={'left': 15, 'right': 15},
        _last_ok_t={'left': now - left_age, 'right': now - right_age})


def test_eye_camera_levels():
    assert _run(FaceCaptureNode._diagnose, _cams(0.1, 0.1), 'left').level == DS.OK
    st = _run(FaceCaptureNode._diagnose, _cams(10, 0.1), 'left')
    assert st.level == DS.WARN and 'mirroring the right' in st.message
    assert _run(FaceCaptureNode._diagnose, _cams(10, 10), 'left').level == DS.ERROR
    assert _run(FaceCaptureNode._diagnose, _cams(10, 10, active=False), 'left').message == 'inactive'


# ── OAK ─────────────────────────────────────────────────────────────────────

def test_oak_levels():
    def oak(age, running=True):
        return types.SimpleNamespace(_running=running, _model_name='yolov6-nano', _restarts=0,
                                     _n_packets=15, _last_packet_t=time.monotonic() - age)
    assert _run(OakNode._diagnose, oak(0.1)).level == DS.OK          # empty scene still OK
    assert _run(OakNode._diagnose, oak(5)).level == DS.ERROR
    assert _run(OakNode._diagnose, oak(5, running=False)).message == 'inactive'
