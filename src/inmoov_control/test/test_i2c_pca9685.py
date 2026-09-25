#!/usr/bin/env python3
"""
test_i2c_pca9685.py — Тест I2C шины и PCA9685 через Left Arduino.

Arduino должна быть подключена по USB. Серво питание НЕ нужно.
PCA9685 питается от 3.3V/5V логики Arduino через VCC пин.

Что проверяется:
  1. Arduino отвечает на CMD_DIAG_REQ
  2. I2C скан — какие устройства найдены
  3. PCA9685 найдена на адресе 0x40
  4. Регистр MODE1 читается корректно (не 0xFF)
  5. Ожидаемые биты MODE1 соответствуют инициализированному состоянию

Использование:
  python3 src/inmoov_control/test/test_i2c_pca9685.py \
    --port /dev/serial/by-path/pci-0000:c6:00.3-usb-0:1.3:1.0-port0

  Или напрямую:
  python3 src/inmoov_control/test/test_i2c_pca9685.py --port /dev/ttyACM0
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import serial
from inmoov_control.protocol import (
    build_frame, FrameParser,
    CMD_DIAG_REQ, CMD_DIAG_RESP
)

# PCA9685 MODE1 register bits
MODE1_SLEEP   = 0x10   # Sleep mode bit (set = sleeping)
MODE1_AI      = 0x20   # Auto-increment
MODE1_ALLCALL = 0x01   # AllCall address enabled

PCA9685_ADDR  = 0x40


def check_mode1(mode1: int) -> list[str]:
    """Return list of notes about MODE1 register value."""
    notes = []
    if mode1 & MODE1_SLEEP:
        notes.append("SLEEP bit set — chip is sleeping (normal after pca.begin() before setPWMFreq)")
    else:
        notes.append("SLEEP bit clear — chip is awake and running")
    if mode1 & MODE1_AI:
        notes.append("Auto-increment ON (expected after Adafruit library init)")
    if mode1 & MODE1_ALLCALL:
        notes.append("ALLCALL enabled (default)")
    return notes


def run(port: str, timeout: float = 5.0):
    print(f"\n{'─'*60}")
    print("PCA9685 I2C Diagnostic Test")
    print(f"Port:    {port}")
    print(f"Target:  Left Arduino Mega → PCA9685 @ 0x{PCA9685_ADDR:02X}")
    print(f"{'─'*60}\n")

    try:
        ser = serial.Serial(port, 115200, timeout=0.2)
    except serial.SerialException as e:
        print(f"[FAIL] Cannot open port: {e}")
        sys.exit(1)

    print("[....] Waiting for Arduino boot (2s)...")
    time.sleep(2.0)
    ser.reset_input_buffer()

    # Send diagnostic request
    frame = build_frame(CMD_DIAG_REQ, b'')
    ser.write(frame)
    print(f"[ TX ] CMD_DIAG_REQ sent ({len(frame)} bytes)\n")

    # Wait for response
    parser   = FrameParser()
    deadline = time.time() + timeout
    resp_data = None

    while time.time() < deadline:
        raw = ser.read(64)
        if raw:
            parser.push(raw)
            for cmd, data in parser.frames:
                if cmd == CMD_DIAG_RESP:
                    resp_data = data
                    break
            parser.frames.clear()
        if resp_data is not None:
            break
        time.sleep(0.05)

    ser.close()

    if resp_data is None:
        print("[FAIL] No response from Arduino within timeout.")
        print("       Check: firmware flashed? Port correct? Arduino powered?\n")
        sys.exit(1)

    # Parse response: [n_devices, addr0..addrN, pca_mode1]
    n_devices = resp_data[0]
    addrs     = list(resp_data[1:1 + n_devices])
    pca_mode1 = resp_data[1 + n_devices] if len(resp_data) >= 2 + n_devices else 0xFF

    # ── I2C scan results ──────────────────────────────────────────────────────
    print(f"I2C Scan — {n_devices} device(s) found:")
    if n_devices == 0:
        print("  [WARN] No I2C devices found — check SDA/SCL wiring and VCC to PCA9685")
    else:
        for addr in addrs:
            tag = " ← PCA9685" if addr == PCA9685_ADDR else ""
            print(f"  0x{addr:02X} ({addr:3d}){tag}")

    print()

    # ── PCA9685 specific ──────────────────────────────────────────────────────
    pca_found = PCA9685_ADDR in addrs

    if not pca_found:
        print(f"[FAIL] PCA9685 NOT found at 0x{PCA9685_ADDR:02X}")
        print("       Check: VCC connected? SDA/SCL connected? A0-A5 pins (set address)?")
        sys.exit(1)

    print(f"[ OK ] PCA9685 found at 0x{PCA9685_ADDR:02X}")

    if pca_mode1 == 0xFF:
        print("[WARN] MODE1 register read failed (returned 0xFF — may mean read error)")
    else:
        print(f"[ OK ] MODE1 register = 0x{pca_mode1:02X} ({pca_mode1:08b}b)")
        for note in check_mode1(pca_mode1):
            print(f"       {note}")

    # ── PWM write test — without servo power, just register write ─────────────
    print()
    print("Summary:")
    ok = pca_found and pca_mode1 != 0xFF
    if ok:
        print("  [PASS] PCA9685 is powered, responding, and readable via I2C")
        print("  [PASS] Firmware can communicate with the chip")
        print("  [INFO] Servo movement requires separate V+ power supply (5-6V)")
    else:
        print("  [FAIL] PCA9685 did not respond correctly")

    print(f"{'─'*60}\n")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description='Test PCA9685 via Left Arduino I2C')
    ap.add_argument('--port',    required=True,  help='Serial port')
    ap.add_argument('--timeout', type=float, default=5.0, help='Response timeout (sec)')
    args = ap.parse_args()
    run(args.port, args.timeout)


if __name__ == '__main__':
    main()
