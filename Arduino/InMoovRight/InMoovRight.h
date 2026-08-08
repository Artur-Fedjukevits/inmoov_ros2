#pragma once
#include <Arduino.h>
#include <Servo.h>

struct SmoothServo {
  Servo    servo_obj;
  int16_t  current;
  int16_t  target;
  int16_t  rest_angle;
  int16_t  min_angle;
  int16_t  max_angle;
  uint8_t  step;
  uint32_t last_update;
  uint8_t  pin;
};

enum ServoIndex {
  IDX_THUMB = 0,
  IDX_INDEX,
  IDX_MIDDLE,
  IDX_RING,
  IDX_PINKY,
  IDX_WRIST,
  IDX_BICEP,
  IDX_ROTATE,
  IDX_SHOULDER,
  IDX_OMOPLATE,
  IDX_ROLLNECK,
  IDX_EYE_LR_R,
  IDX_EYE_UD_R,
  IDX_UPPERLIP,
  SERVO_TOTAL_COUNT   // = 14
};
