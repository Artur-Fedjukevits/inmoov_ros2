"""
arduino_right_node.py — Right Arduino Mega (subsys1).

Hardware (matching setup_xicro_subsystem1.yaml):
  Body servos:
    omoplate_R, shoulder_R, rotate_R, bicep_R  — JX-PDI HV2060MG
    wrist_R, thumb_R, index_R, middle_R, ring_R, pinky_R  — PDI 6225MG
    rollneck  — JX PDI 6221MG

  Face servos (on this Arduino but treated as face):
    eye_lr_R, eye_ud_R  — small eye servos
    upperLip            — lip servo

  Sensors:
    ultrasonic (right side)  → /ultrasonic_right_distance
    PIR                      → /pir_state
    Hall finger sensors (5)  → /hall_right_raw (Int16MultiArray, raw analogRead)
                                order: [thumb, index, middle, ring, pinky]
                                pins A0-A4 on the Arduino, sent as CMD_HALL frame

  /robot_sleep (latched Bool) is forwarded to the Arduino as CMD_SLEEP —
  while asleep, ultrasonic/PIR/Hall telemetry stops.

UART packet order (body + face, total 14 servos) — ORDER MUST MATCH InMoovRight.ino:
  Byte 0:  thumb_R
  Byte 1:  index_R
  Byte 2:  middle_R
  Byte 3:  ring_R
  Byte 4:  pinky_R
  Byte 5:  wrist_R
  Byte 6:  bicep_R
  Byte 7:  rotate_R
  Byte 8:  shoulder_R
  Byte 9:  omoplate_R
  Byte 10: rollneck
  Byte 11: eye_lr_R     (face)
  Byte 12: eye_ud_R     (face)
  Byte 13: upperLip     (face)

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import sys
import rclpy
from rclpy.executors import MultiThreadedExecutor
from .arduino_comm_node import ArduinoCommNode

DEFAULT_PORT = '/dev/serial/by-path/pci-0000:c6:00.3-usb-0:5:1.0-port0'


class ArduinoRightNode(ArduinoCommNode):

    # (joint_name_in_JointState, center_deg)
    # ORDER MUST MATCH InMoovRight.ino ServoIndex enum / packet byte offsets.
    # (joint_name, center_deg, rest_deg)
    # ORDER MUST MATCH InMoovRight.ino ServoIndex enum / packet byte offsets.
    BODY_JOINTS = [
        ('thumb_R',     90,  60),   # [0]  pin 2,  rest=60,  min=0,   max=140
        ('index_R',     90,  40),   # [1]  pin 3,  rest=40,  min=0,   max=160
        ('middle_R',    90,  40),   # [2]  pin 4,  rest=40,  min=0,   max=160
        ('ring_R',      90,  30),   # [3]  pin 5,  rest=30,  min=0,   max=150
        ('pinky_R',     90,  40),   # [4]  pin 6,  rest=40,  min=0,   max=170
        ('wrist_R',     90,  90),   # [5]  pin 7,  rest=90,  min=0,   max=180
        ('bicep_R',     90,   0),   # [6]  pin 8,  rest=0,   min=0,   max=80
        ('rotate_R',    90,  90),   # [7]  pin 9,  rest=90,  min=40,  max=180
        ('shoulder_R',  90,  30),   # [8]  pin 10, rest=30,  min=0,   max=180
        ('omoplate_R',  90,  10),   # [9]  pin 11, rest=10,  min=10,  max=80
        ('rollneck',    90,  80),   # [10] pin 13, rest=80,  min=50,  max=115
    ]

    FACE_JOINTS = [
        ('eye_lr_R',    90,  90),   # [11] pin 22, rest=90,  min=80,  max=100
        ('eye_ud_R',    90, 100),   # [12] pin 24, rest=100, min=85,  max=115
        ('upperLip',    90,  90),   # [13] pin 26, rest=90,  min=90,  max=105
    ]

    HAS_ULTRASONIC   = True
    HAS_PIR          = True
    HAS_HALL         = True
    ULTRASONIC_TOPIC = 'ultrasonic_right_distance'
    PIR_TOPIC        = 'pir_state'
    HALL_TOPIC       = 'hall_right_raw'

    def __init__(self, serial_port: str = DEFAULT_PORT):
        super().__init__('arduino_right_node', serial_port)


def main(args=None):
    rclpy.init(args=args)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=DEFAULT_PORT)
    known, _ = parser.parse_known_args()

    node = ArduinoRightNode(serial_port=known.port)
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
