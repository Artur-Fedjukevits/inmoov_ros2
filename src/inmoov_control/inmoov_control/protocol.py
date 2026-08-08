"""
protocol.py — Binary serial protocol for InMoov Arduino communication.

Frame format (both directions):
  [0xAA][0x55][CMD][LEN][DATA...][CRC8]

  SOF  = 0xAA 0x55  (2 bytes, start-of-frame marker)
  CMD  = 1 byte command
  LEN  = 1 byte, number of DATA bytes
  DATA = LEN bytes payload
  CRC8 = XOR of CMD + LEN + all DATA bytes

ROS → Arduino commands:
  CMD_SET_SERVOS   = 0x01   DATA: servo values, 1 byte each (0–180 degrees)
  CMD_SET_SPEEDS   = 0x02   DATA: step sizes, 1 byte each (0=keep current, 1–255 degrees/tick)
                              speed_deg_per_sec ≈ step * 1000 / SMOOTH_INTERVAL_MS (default 60ms)
                              e.g. step=10 → ~167°/s, step=2 → ~33°/s
  CMD_DIAG_REQ     = 0x20   DATA: none — request I2C scan + PCA9685 check

Arduino → ROS events:
  CMD_ULTRASONIC   = 0x10   DATA: uint16 big-endian, distance in cm
  CMD_PIR          = 0x11   DATA: 1 byte, 0 = no motion, 1 = motion detected
  CMD_DIAG_RESP    = 0x21   DATA: [n_devices, addr0, addr1, ..., pca_mode1]
                              n_devices: number of I2C devices found
                              addrN:     I2C address of each device (7-bit)
                              pca_mode1: PCA9685 MODE1 register (0xFF = not found)
  CMD_ACK          = 0xFF   DATA: 1 byte, echoed CMD (acknowledgement)
"""

import struct

SOF = b'\xAA\x55'

CMD_SET_SERVOS  = 0x01
CMD_SET_SPEEDS  = 0x02
CMD_DIAG_REQ    = 0x20
CMD_DIAG_RESP   = 0x21
CMD_ULTRASONIC  = 0x10
CMD_PIR         = 0x11
CMD_ACK         = 0xFF

HEADER_LEN = 4   # SOF(2) + CMD(1) + LEN(1)
FOOTER_LEN = 1   # CRC8


def crc8(data: bytes) -> int:
    """XOR-based CRC8."""
    crc = 0
    for b in data:
        crc ^= b
    return crc


def build_frame(cmd: int, data: bytes) -> bytes:
    """Pack a frame ready to send over serial."""
    payload = bytes([cmd, len(data)]) + data
    return SOF + payload + bytes([crc8(payload)])


def build_set_servos(servo_degrees: list[int]) -> bytes:
    """Build a SET_SERVOS frame. Values clamped to 0-180."""
    data = bytes(max(0, min(180, int(v))) for v in servo_degrees)
    return build_frame(CMD_SET_SERVOS, data)


# SMOOTH_INTERVAL_MS from firmware (both Left and Right Arduinos)
_SMOOTH_INTERVAL_S = 0.060


def deg_per_sec_to_step(deg_per_sec: float) -> int:
    """Convert speed in degrees/second to firmware step size (degrees/tick)."""
    step = round(abs(deg_per_sec) * _SMOOTH_INTERVAL_S)
    return max(1, min(255, step))


def build_set_speeds(speeds: list[int]) -> bytes:
    """Build a SET_SPEEDS frame. 0 = keep current speed, 1–255 = step size."""
    data = bytes(max(0, min(255, int(v))) for v in speeds)
    return build_frame(CMD_SET_SPEEDS, data)


class FrameParser:
    """
    Stateful streaming parser. Feed bytes with push(); get completed frames
    from the frames list.

    Usage:
        parser = FrameParser()
        parser.push(serial.read(64))
        for cmd, data in parser.frames:
            handle(cmd, data)
        parser.frames.clear()
    """

    _ST_WAIT_SOF1 = 0
    _ST_WAIT_SOF2 = 1
    _ST_CMD       = 2
    _ST_LEN       = 3
    _ST_DATA      = 4
    _ST_CRC       = 5

    def __init__(self):
        self.frames: list[tuple[int, bytes]] = []
        self._state = self._ST_WAIT_SOF1
        self._cmd   = 0
        self._len   = 0
        self._buf   = bytearray()

    def push(self, raw: bytes) -> None:
        for b in raw:
            self._feed(b)

    def _feed(self, b: int) -> None:
        s = self._state
        if s == self._ST_WAIT_SOF1:
            if b == 0xAA:
                self._state = self._ST_WAIT_SOF2
        elif s == self._ST_WAIT_SOF2:
            if b == 0x55:
                self._state = self._ST_CMD
            elif b == 0xAA:
                self._state = self._ST_WAIT_SOF2  # 0xAA may start new frame
            else:
                self._state = self._ST_WAIT_SOF1
        elif s == self._ST_CMD:
            self._cmd   = b
            self._state = self._ST_LEN
        elif s == self._ST_LEN:
            self._len   = b
            self._buf   = bytearray()
            self._state = self._ST_DATA if b > 0 else self._ST_CRC
        elif s == self._ST_DATA:
            self._buf.append(b)
            if len(self._buf) == self._len:
                self._state = self._ST_CRC
        elif s == self._ST_CRC:
            payload = bytes([self._cmd, self._len]) + bytes(self._buf)
            if crc8(payload) == b:
                self.frames.append((self._cmd, bytes(self._buf)))
            self._state = self._ST_WAIT_SOF1
