"""
arduino_left_node.py — Left Arduino Mega (subsys2).

UART packet (body + face, total 28 bytes) — ORDER MUST MATCH InMoovLeft.ino:

  Body (15 bytes, 0-14) — ACT imitation learning joints:
    Byte 0:  thumb_L     Byte 4:  pinky_L    Byte 8:  shoulder_L
    Byte 1:  index_L     Byte 5:  wrist_L    Byte 9:  omoplate_L
    Byte 2:  majeure_L   Byte 6:  bicep_L    Byte 10: neck
    Byte 3:  ring_L      Byte 7:  rotate_L   Byte 11: rothead
                                              Byte 12: topstom
                                              Byte 13: midstom
                                              Byte 14: lowstom

  Face GPIO (3 bytes, 15-17):
    Byte 15: eye_lr_L    Byte 16: eye_ud_L   Byte 17: jaw

  Face PCA9685 (10 bytes, 18-27):
    Byte 18-27: eyelid_L_Upper .. forhead_R

  Hall finger sensors (5)  → /hall_left_raw (Int16MultiArray, raw analogRead)
                              order: [thumb, index, middle, ring, pinky]
                              (MIDDLE/PINKY physical pins swapped vs. the
                              right arm — see InMoovLeft.ino)

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor
from .arduino_comm_node import ArduinoCommNode

DEFAULT_PORT = '/dev/serial/by-path/pci-0000:c6:00.3-usb-0:1.3:1.0-port0'


class ArduinoLeftNode(ArduinoCommNode):

    # ORDER MUST MATCH InMoovLeft.ino ServoIndex enum / packet byte offsets.
    BODY_JOINTS = [
        ('thumb_L',     90,  50),   # [0]  pin 2,  rest=50,  min=0,   max=145
        ('index_L',     90,   0),   # [1]  pin 3,  rest=0,   min=0,   max=150
        ('majeure_L',   90,   0),   # [2]  pin 4,  rest=0,   min=0,   max=150
        ('ring_L',      90,   0),   # [3]  pin 5,  rest=0,   min=0,   max=140
        ('pinky_L',     90,   0),   # [4]  pin 6,  rest=0,   min=0,   max=150
        ('wrist_L',     90, 150),   # [5]  pin 7,  rest=150, min=0,   max=300 (firmware; Servo.write caps at 180)
        ('bicep_L',     90,   0),   # [6]  pin 8,  rest=0,   min=0,   max=90
        ('rotate_L',    90,  90),   # [7]  pin 9,  rest=90,  min=40,  max=180
        ('shoulder_L',  90,  20),   # [8]  pin 10, rest=20,  min=0,   max=180
        ('omoplate_L',  90,  25),   # [9]  pin 11, rest=25,  min=25,  max=90
        ('neck',        90,  40),   # [10] pin 12, rest=40,  min=0,   max=100
        ('rothead',     90,  90),   # [11] pin 13, rest=90,  min=30,  max=140
        ('topstom',     90,  83),   # [12] pin 28, rest=83,  min=60,  max=110
        ('midstom',     90,  90),   # [13] pin 27, rest=90,  min=60,  max=120
        ('lowstom',     90,  90),   # [14] pin 29, rest=90,  min=0,   max=180
    ]

    # Face joints: GPIO eyes/jaw (bytes 15-17) + PCA9685 (bytes 18-27)
    FACE_JOINTS = [
        ('eye_lr_L',        90,  90),   # [15] pin 22, rest=90,  min=80,  max=100
        ('eye_ud_L',        90, 100),   # [16] pin 24, rest=100, min=80,  max=110
        ('jaw',             90,  10),   # [17] pin 26, rest=10,  min=10,  max=90
        ('eyelid_L_Upper',  90,  85),   # [18] ch 6,  rest=85, min=70,  max=95
        ('eyelid_L_Lower',  90,  85),   # [19] ch 7,  rest=85, min=75,  max=95
        ('eyelid_R_Upper',  90,  85),   # [20] ch 8,  rest=85, min=65,  max=100
        ('eyelid_R_Lower',  90,  85),   # [21] ch 9,  rest=85, min=70,  max=95
        ('eyebrow_L',       90,  90),   # [22] ch 10, rest=90, min=60,  max=110
        ('eyebrow_R',       90,  80),   # [23] ch 11, rest=80, min=70,  max=105
        ('cheek_L',         90, 100),   # [24] ch 14, rest=100,min=75,  max=115
        ('cheek_R',         90,  87),   # [25] ch 15, rest=87, min=68,  max=105
        ('forhead_L',       90,  90),   # [26] ch 12, rest=90, min=90,  max=110
        ('forhead_R',       90,  85),   # [27] ch 13, rest=85, min=85,  max=105
    ]

    # Firmware table step (deg per 60 ms tick) — default speed, same order
    DEFAULT_STEPS = [2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2,   # body
                     1, 1, 10,                                      # eye_lr, eye_ud, jaw
                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2]                  # PCA face

    HAS_ULTRASONIC   = True
    HAS_PIR          = False
    HAS_HALL         = True
    ULTRASONIC_TOPIC = 'ultrasonic_left_distance'
    HALL_TOPIC       = 'hall_left_raw'

    def __init__(self, serial_port: str = DEFAULT_PORT):
        super().__init__('arduino_left_node', serial_port)


def main(args=None):
    rclpy.init(args=args)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=DEFAULT_PORT)
    known, _ = parser.parse_known_args()

    node = ArduinoLeftNode(serial_port=known.port)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
