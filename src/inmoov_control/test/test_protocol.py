"""
test_protocol.py — Unit tests for binary serial protocol (no hardware needed).

Tests:
  1. Frame building: correct SOF, CMD, LEN, CRC
  2. FrameParser: parses valid frames
  3. FrameParser: rejects frames with bad CRC
  4. FrameParser: handles fragmented / multi-frame byte streams
  5. Servo degree clamping in build_set_servos
  6. Round-trip: build → parse → same data

Run:
  cd /home/artur/ros2_ws
  python3 -m pytest src/inmoov_control/test/test_protocol.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inmoov_control.protocol import (
    build_frame, build_set_servos, build_sleep, FrameParser,
    CMD_SET_SERVOS, CMD_ULTRASONIC, CMD_PIR, CMD_SLEEP, CMD_HALL, CMD_STATUS, crc8
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_all(data: bytes) -> list[tuple[int, bytes]]:
    """Feed bytes into a fresh parser, return all completed frames."""
    p = FrameParser()
    p.push(data)
    return list(p.frames)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Frame structure
# ─────────────────────────────────────────────────────────────────────────────

def test_frame_starts_with_sof():
    frame = build_frame(CMD_SET_SERVOS, b'\x5A')
    assert frame[:2] == b'\xAA\x55'

def test_frame_cmd_and_len():
    data = bytes([10, 20, 30])
    frame = build_frame(CMD_SET_SERVOS, data)
    assert frame[2] == CMD_SET_SERVOS
    assert frame[3] == 3

def test_frame_total_length():
    data = bytes(14)   # Right Arduino — 14 servos
    frame = build_set_servos(data)
    # SOF(2) + CMD(1) + LEN(1) + DATA(14) + CRC(1) = 19
    assert len(frame) == 19

def test_frame_crc_valid():
    data = bytes([90] * 14)
    frame = build_set_servos(data)
    payload = frame[2:-1]   # CMD + LEN + DATA
    expected_crc = crc8(payload)
    assert frame[-1] == expected_crc

def test_empty_frame():
    frame = build_frame(CMD_PIR, b'')
    assert frame[3] == 0      # LEN=0
    assert len(frame) == 5    # SOF(2)+CMD+LEN+CRC


# ─────────────────────────────────────────────────────────────────────────────
# 2. Clamping in build_set_servos
# ─────────────────────────────────────────────────────────────────────────────

def test_clamping_above_180():
    frame = build_set_servos([200, 255, 181])
    frames = parse_all(frame)
    assert len(frames) == 1
    _, data = frames[0]
    assert all(b == 180 for b in data)

def test_clamping_below_0():
    frame = build_set_servos([-5, -1, 0])
    frames = parse_all(frame)
    _, data = frames[0]
    assert data[0] == 0
    assert data[1] == 0
    assert data[2] == 0

def test_valid_range_preserved():
    values = [0, 45, 90, 135, 180]
    frame = build_set_servos(values)
    frames = parse_all(frame)
    _, data = frames[0]
    assert list(data) == values


# ─────────────────────────────────────────────────────────────────────────────
# 3. FrameParser — valid frame
# ─────────────────────────────────────────────────────────────────────────────

def test_parser_single_frame():
    frame = build_set_servos([90] * 14)
    frames = parse_all(frame)
    assert len(frames) == 1
    cmd, data = frames[0]
    assert cmd == CMD_SET_SERVOS
    assert len(data) == 14
    assert all(b == 90 for b in data)

def test_parser_ultrasonic_frame():
    dist_cm = 123
    data = bytes([dist_cm >> 8, dist_cm & 0xFF])
    frame = build_frame(CMD_ULTRASONIC, data)
    frames = parse_all(frame)
    assert len(frames) == 1
    cmd, payload = frames[0]
    assert cmd == CMD_ULTRASONIC
    recovered = (payload[0] << 8) | payload[1]
    assert recovered == dist_cm

def test_parser_pir_frame():
    frame = build_frame(CMD_PIR, bytes([1]))
    frames = parse_all(frame)
    cmd, data = frames[0]
    assert cmd == CMD_PIR
    assert data[0] == 1

def test_build_sleep_true():
    frame = build_sleep(True)
    frames = parse_all(frame)
    assert len(frames) == 1
    cmd, data = frames[0]
    assert cmd == CMD_SLEEP
    assert data == bytes([1])

def test_build_sleep_false():
    frame = build_sleep(False)
    frames = parse_all(frame)
    cmd, data = frames[0]
    assert cmd == CMD_SLEEP
    assert data == bytes([0])

def test_parser_hall_frame():
    """5 fingers x uint16 big-endian, order [thumb, index, middle, ring, pinky]."""
    values = [544, 700, 512, 800, 650]
    data = b''.join(bytes([v >> 8, v & 0xFF]) for v in values)
    frame = build_frame(CMD_HALL, data)
    frames = parse_all(frame)
    assert len(frames) == 1
    cmd, payload = frames[0]
    assert cmd == CMD_HALL
    recovered = [(payload[i] << 8) | payload[i + 1] for i in range(0, 10, 2)]
    assert recovered == values


# ─────────────────────────────────────────────────────────────────────────────
# 4. FrameParser — error resistance
# ─────────────────────────────────────────────────────────────────────────────

def test_parser_bad_crc_rejected():
    frame = bytearray(build_set_servos([90] * 14))
    frame[-1] ^= 0xFF   # corrupt CRC
    frames = parse_all(bytes(frame))
    assert len(frames) == 0

def test_parser_garbage_before_frame():
    garbage = bytes([0x00, 0xAA, 0x11, 0xFF, 0x55, 0xAA])
    frame = build_set_servos([45] * 14)
    frames = parse_all(garbage + frame)
    assert len(frames) == 1
    _, data = frames[0]
    assert all(b == 45 for b in data)

def test_parser_two_frames():
    f1 = build_set_servos([10] * 14)
    f2 = build_set_servos([170] * 14)
    frames = parse_all(f1 + f2)
    assert len(frames) == 2
    assert all(b == 10  for b in frames[0][1])
    assert all(b == 170 for b in frames[1][1])

def test_parser_partial_then_complete():
    """Simulates TCP-like fragmentation: frame split across two pushes."""
    frame = build_set_servos([90] * 14)
    p = FrameParser()
    p.push(frame[:8])
    assert len(p.frames) == 0    # not yet complete
    p.push(frame[8:])
    assert len(p.frames) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. Round-trip: 28 servos (Left Arduino packet size)
# ─────────────────────────────────────────────────────────────────────────────

def test_roundtrip_left_arduino():
    """Full 28-byte packet as sent by arduino_left_node."""
    # Body (18) + Face PCA (10)
    body  = list(range(0, 18 * 10, 10))[:18]   # 0,10,20,...,170
    face  = [90] * 10
    values = body + face
    assert len(values) == 28

    frame = build_set_servos(values)
    frames = parse_all(frame)
    assert len(frames) == 1
    _, data = frames[0]
    assert list(data) == values

def test_roundtrip_right_arduino():
    """Full 14-byte packet as sent by arduino_right_node."""
    values = [0, 0, 0, 0, 0, 0,   # fingers
              45, 90, 30, 10,      # bicep, rotate, shoulder, omoplate
              80,                   # rollneck
              90, 100, 90]         # eye_lr, eye_ud, upperLip
    assert len(values) == 14

    frame = build_set_servos(values)
    frames = parse_all(frame)
    _, data = frames[0]
    assert list(data) == values


# ─────────────────────────────────────────────────────────────────────────────
# 6. Firmware failsafe status + literal 0°
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_status_frame():
    """CMD_STATUS from firmware: 1 = entered failsafe, 0 = left it."""
    for flag in (1, 0):
        frames = parse_all(build_frame(CMD_STATUS, bytes([flag])))
        assert frames == [(CMD_STATUS, bytes([flag]))]

def test_zero_degrees_is_sent_literally():
    """0 is a real angle for the firmware now (no rest sentinel) — must reach it as 0."""
    frame = build_set_servos([0, 90, 0])
    _, data = parse_all(frame)[0]
    assert list(data) == [0, 90, 0]
