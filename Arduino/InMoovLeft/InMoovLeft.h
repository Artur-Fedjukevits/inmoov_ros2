// InMoovLeft.h — servo/state definitions for the left Arduino Mega 2560.
// Author: Artur Fedjukevits | Assisted by: Claude Code (Anthropic)
// License: GNU General Public License v3.0 (see repository root LICENSE)
#pragma once
#include <Arduino.h>
#include <Servo.h>
#include <Adafruit_PWMServoDriver.h>

enum ServoDriver { DRIVER_GPIO, DRIVER_PCA };

struct SmoothServo {
  int16_t  current;
  int16_t  target;
  int16_t  rest_angle;
  int16_t  min_angle;
  int16_t  max_angle;
  uint8_t  step;
  uint32_t last_update;

  ServoDriver driver;
  Servo    servo_obj;   // DRIVER_GPIO
  uint8_t  pin;         // DRIVER_GPIO
  uint8_t  channel;     // DRIVER_PCA
  bool     inverted;    // DRIVER_PCA: physical = 180 - angle
};

enum ServoIndex {
  // Body joints (0-14) — used for ACT imitation learning
  IDX_THUMB_L = 0,
  IDX_INDEX_L,
  IDX_MAJEURE_L,
  IDX_RING_L,
  IDX_PINKY_L,
  IDX_WRIST_L,
  IDX_BICEP_L,
  IDX_ROTATE_L,
  IDX_SHOULDER_L,
  IDX_OMOPLATE_L,
  IDX_NECK,
  IDX_ROTHEAD,
  IDX_TOPSTOM,
  IDX_MIDSTOM,
  IDX_LOWSTOM,
  // Face GPIO joints (15-17)
  IDX_EYE_LR,
  IDX_EYE_UD,
  IDX_JAW,
  // Face PCA9685 joints (18-27)
  IDX_EYELID_L_UPPER,
  IDX_EYELID_L_LOWER,
  IDX_EYELID_R_UPPER,
  IDX_EYELID_R_LOWER,
  IDX_EYEBROW_L,
  IDX_EYEBROW_R,
  IDX_CHEEK_L,
  IDX_CHEEK_R,
  IDX_FORHEAD_L,
  IDX_FORHEAD_R,
  SERVO_TOTAL_COUNT   // = 28
};
