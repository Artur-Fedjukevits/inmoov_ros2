#!/usr/bin/env python3
"""
serial_loopback_test.py — Тест Arduino по USB без питания серво.

Arduino работает от USB (5V), серво не подключены к питанию → безопасно.
Скрипт отправляет SET_SERVOS пакеты и слушает ответы (ultrasonic, PIR).

Использование:
  python3 src/inmoov_control/test/serial_loopback_test.py --port /dev/ttyACM0
  python3 src/inmoov_control/test/serial_loopback_test.py --port /dev/serial/by-path/...

  --port  PORT    Серийный порт Arduino
  --count N       Количество пакетов (по умолчанию 10)
  --delay S       Задержка между пакетами в сек (по умолчанию 0.5)
  --left          Режим Left Arduino (28 байт), по умолчанию Right (14 байт)
"""

import argparse
import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import serial  # noqa: E402
from inmoov_control.protocol import (  # noqa: E402
    build_set_servos, FrameParser,
    CMD_ULTRASONIC, CMD_PIR
)


# ─────────────────────────────────────────────────────────────────────────────
# Test packets
# ─────────────────────────────────────────────────────────────────────────────

# Right Arduino: 14 servos — все в rest (=0 для пальцев, остальные центр)
RIGHT_REST = [0, 0, 0, 0, 0, 0,     # fingers (thumb→pinky)
              0, 90, 30, 10,          # bicep, rotate, shoulder, omoplate
              80,                      # rollneck
              90, 100, 90]            # eye_lr_R, eye_ud_R, upperLip

# Left Arduino: 28 servos
LEFT_REST  = [0, 0, 0, 0, 0, 0,     # fingers L
              0, 90, 30, 25,          # bicep, rotate, shoulder, omoplate L
              40, 90,                  # neck, rothead
              90, 100,                 # eye_lr_L, eye_ud_L
              10,                      # jaw
              83, 90, 90,             # topstom, midstom, lowstom
              85, 85, 85, 85,         # eyelids (PCA)
              90, 80,                  # eyebrows
              100, 87,                 # cheeks
              90, 85]                  # foreheads

SEQUENCES = {
    'right': [
        ("Rest position",           RIGHT_REST),
        ("Fingers open",            [180]*6 + RIGHT_REST[6:]),
        ("Fingers closed",          [0]*6   + RIGHT_REST[6:]),
        ("Shoulder up",             [0, 0, 0, 0, 0, 0, 0, 90, 90, 10, 80, 90, 100, 90]),
        ("Head roll left",          [0, 0, 0, 0, 0, 0, 0, 90, 30, 10, 50, 90, 100, 90]),
        ("Head roll right",         [0, 0, 0, 0, 0, 0, 0, 90, 30, 10, 115, 90, 100, 90]),
        ("Rest position",           RIGHT_REST),
    ],
    'left': [
        ("Rest position",           LEFT_REST),
        ("Fingers L open",          [180]*6 + LEFT_REST[6:]),
        ("Fingers L closed",        [0]*6   + LEFT_REST[6:]),
        ("Neck tilt",               LEFT_REST[:10] + [70] + LEFT_REST[11:]),
        ("Rothead left",            LEFT_REST[:11] + [50] + LEFT_REST[12:]),
        ("Jaw open",                LEFT_REST[:14] + [25] + LEFT_REST[15:]),
        ("Jaw close",               LEFT_REST[:14] + [10] + LEFT_REST[15:]),
        ("Rest position",           LEFT_REST),
    ],
}


def format_rx(cmd: int, data: bytes) -> str:
    if cmd == CMD_ULTRASONIC and len(data) >= 2:
        cm = (data[0] << 8) | data[1]
        return f"ULTRASONIC: {cm} cm"
    elif cmd == CMD_PIR and len(data) >= 1:
        return f"PIR: {'motion' if data[0] else 'clear'}"
    else:
        return f"CMD=0x{cmd:02X} data={data.hex()}"


def run_test(port: str, mode: str, count: int, delay: float):
    print(f"\n{'─'*60}")
    print("InMoov Arduino Serial Test")
    print(f"Port:  {port}")
    print(f"Mode:  {mode} Arduino")
    print(f"{'─'*60}\n")

    try:
        ser = serial.Serial(port, 115200, timeout=0.1)
    except serial.SerialException as e:
        print(f"[ERROR] Cannot open port: {e}")
        sys.exit(1)

    time.sleep(2.0)   # Arduino resets on serial connect, wait for bootloader
    ser.reset_input_buffer()
    print("[OK] Port opened. Waiting for Arduino ready...\n")

    parser   = FrameParser()
    sequence = SEQUENCES[mode]
    tx_ok    = 0
    rx_count = 0

    try:
        step_idx = 0
        for i in range(count):
            name, values = sequence[step_idx % len(sequence)]
            step_idx += 1

            frame = build_set_servos(values)
            ser.write(frame)
            tx_ok += 1

            print(f"[TX #{i+1:3d}] {name:30s}  {len(frame)} bytes")

            # Read any incoming sensor data
            time.sleep(delay)
            raw = ser.read(128)
            if raw:
                parser.push(raw)
                for cmd, data in parser.frames:
                    print(f"         [RX] {format_rx(cmd, data)}")
                    rx_count += 1
                parser.frames.clear()

    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
    finally:
        # Send rest position before closing
        rest = sequence[0][1]
        ser.write(build_set_servos(rest))
        time.sleep(0.1)
        ser.close()

    print(f"\n{'─'*60}")
    print(f"TX sent: {tx_ok}   RX received: {rx_count}")
    if rx_count == 0:
        print("[WARN] No sensor responses — check Arduino is running and")
        print("       ultrasonic sensor is connected (or it's OK if not).")
    else:
        print("[OK] Communication working.")
    print(f"{'─'*60}\n")


def run_listen(port: str, duration: float):
    """Listen-only mode: print all sensor frames for N seconds (no servo commands)."""
    print(f"\n{'─'*60}")
    print("InMoov Arduino Sensor Listen")
    print(f"Port:     {port}")
    print(f"Duration: {duration} s  (Ctrl-C to stop)")
    print(f"{'─'*60}\n")

    try:
        ser = serial.Serial(port, 115200, timeout=0.1)
    except serial.SerialException as e:
        print(f"[ERROR] Cannot open port: {e}")
        sys.exit(1)

    time.sleep(2.0)
    print("[OK] Arduino ready. Listening...\n")

    parser   = FrameParser()
    deadline = time.time() + duration
    rx_count = 0

    try:
        while time.time() < deadline:
            raw = ser.read(128)
            if raw:
                parser.push(raw)
                for cmd, data in parser.frames:
                    ts = time.strftime('%H:%M:%S')
                    print(f"[{ts}] {format_rx(cmd, data)}")
                    rx_count += 1
                parser.frames.clear()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
    finally:
        ser.close()

    print(f"\n{'─'*60}")
    print(f"Received {rx_count} frame(s) in {duration:.0f} s")
    if rx_count == 0:
        print("[WARN] Nothing received — check sensor wiring and firmware.")
    print(f"{'─'*60}\n")


def main():
    ap = argparse.ArgumentParser(description='InMoov Arduino serial test (no servo power needed)')
    ap.add_argument('--port',   required=True,  help='Serial port (e.g. /dev/ttyACM0)')
    ap.add_argument('--count',  type=int,   default=14,  help='Number of packets to send')
    ap.add_argument('--delay',  type=float, default=0.5, help='Delay between packets (sec)')
    ap.add_argument('--left',   action='store_true', help='Test Left Arduino (28 bytes)')
    ap.add_argument('--listen', type=float, metavar='SECONDS',
                    help='Listen-only mode: print sensor data for N seconds (no servo commands)')
    args = ap.parse_args()

    if args.listen:
        run_listen(args.port, args.listen)
    else:
        mode = 'left' if args.left else 'right'
        run_test(args.port, mode, args.count, args.delay)


if __name__ == '__main__':
    main()
