/**
 * InMoovRight.ino — Right Arduino Mega 2560
 *
 * Author: Artur Fedjukevits
 * Assisted by: Claude Code (Anthropic)
 * License: GNU General Public License v3.0 (see repository root LICENSE)
 *
 * Replaces xicro. Same servo logic, min/max/rest/step/pins preserved.
 * Protocol: binary batch frame from ROS2 over USB Serial.
 *
 * Frame format: [0xAA][0x55][CMD][LEN][DATA...][CRC8]
 *   CRC8 = XOR of CMD, LEN, and all DATA bytes
 *
 * ROS2 → Arduino:
 *   CMD=0x01  SET_SERVOS: DATA = SERVO_TOTAL_COUNT bytes, degrees 0-180
 *             Byte order matches ServoIndex enum below.
 *   CMD=0x03  SLEEP:      DATA = 1 byte, 0=awake, 1=sleeping.
 *             While sleeping, ULTRASONIC/PIR/HALL telemetry stops.
 *
 * Arduino → ROS2:
 *   CMD=0x10  ULTRASONIC: DATA = uint16 big-endian, distance in cm
 *   CMD=0x11  PIR:        DATA = 1 byte, 0/1
 *   CMD=0x12  HALL:       DATA = 5 x uint16 big-endian, raw analogRead (0-1023)
 *             order: [thumb, index, middle, ring, pinky] — pins A0-A4
 *
 * Packet order (14 bytes):
 *   [0]  thumb_R    [1]  index_R    [2]  middle_R   [3]  ring_R
 *   [4]  pinky_R    [5]  wrist_R    [6]  bicep_R    [7]  rotate_R
 *   [8]  shoulder_R [9]  omoplate_R [10] rollneck
 *   [11] eye_lr_R   [12] eye_ud_R   [13] upperLip
 */

#include "InMoovRight.h"

// ─────────────────────────────────────────────────────────────────────────────
// Constants (same as Xicro_subsys_right_ID_1.ino)
// ─────────────────────────────────────────────────────────────────────────────
#define SMOOTH_INTERVAL_MS      60
#define ULTRASONIC_INTERVAL_MS  250
#define ULTRASONIC_TIMEOUT_US   25000   // ~4m
#define PIR_INTERVAL_MS         100
#define HALL_INTERVAL_MS        100

#define ULTRASONIC_TRIG_PIN     64
#define ULTRASONIC_ECHO_PIN     63
#define PIR_PIN                 23

#define THUMB_HALL_PIN          A0
#define INDEX_HALL_PIN          A1
#define MIDDLE_HALL_PIN         A2
#define RING_HALL_PIN           A3
#define PINKY_HALL_PIN          A4

// ─────────────────────────────────────────────────────────────────────────────
// Protocol
// ─────────────────────────────────────────────────────────────────────────────
#define CMD_SET_SERVOS  0x01
#define CMD_SET_SPEEDS  0x02
#define CMD_SLEEP       0x03
#define CMD_ULTRASONIC  0x10
#define CMD_PIR         0x11
#define CMD_HALL        0x12

bool sleeping = false;   // set via CMD_SLEEP; gates ultrasonic/PIR/Hall telemetry

void sendFrame(uint8_t cmd, uint8_t* data, uint8_t len) {
  uint8_t crc = cmd ^ len;
  for (uint8_t i = 0; i < len; i++) crc ^= data[i];
  Serial.write(0xAA);
  Serial.write(0x55);
  Serial.write(cmd);
  Serial.write(len);
  if (len > 0) Serial.write(data, len);
  Serial.write(crc);
}

// ─────────────────────────────────────────────────────────────────────────────
// Servo table (identical parameters to Xicro_subsys_right_ID_1.ino)
// ─────────────────────────────────────────────────────────────────────────────
//                              rest  min   max  step  pin
SmoothServo servos[SERVO_TOTAL_COUNT] = {
  /* THUMB_R    pin 2  */ {Servo(), 0, 0,  60,   0, 140,  2, 0,  2},
  /* INDEX_R    pin 3  */ {Servo(), 0, 0,  40,   0, 160,  2, 0,  3},
  /* MIDDLE_R   pin 4  */ {Servo(), 0, 0,  40,   0, 160,  2, 0,  4},
  /* RING_R     pin 5  */ {Servo(), 0, 0,  30,   0, 150,  2, 0,  5},
  /* PINKY_R    pin 6  */ {Servo(), 0, 0,  40,   0, 170,  2, 0,  6},
  /* WRIST_R    pin 7  */ {Servo(), 0, 0,  90,   0, 180,  2, 0,  7},
  /* BICEP_R    pin 8  */ {Servo(), 0, 0,   0,   0,  80,  1, 0,  8},
  /* ROTATE_R   pin 9  */ {Servo(), 0, 0,  90,  40, 180,  1, 0,  9},
  /* SHOULDER_R pin 10 */ {Servo(), 0, 0,  30,   0, 180,  1, 0, 10},
  /* OMOPLATE_R pin 11 */ {Servo(), 0, 0,  10,  10,  80,  1, 0, 11},
  /* ROLLNECK   pin 13 */ {Servo(), 0, 0,  80,  50, 115,  2, 0, 13},
  /* EYE_LR_R   pin 22 */ {Servo(), 0, 0,  90,  80, 100,  2, 0, 22},
  /* EYE_UD_R   pin 24 */ {Servo(), 0, 0, 100,  85, 115,  2, 0, 24},
  /* UPPERLIP   pin 26 */ {Servo(), 0, 0,  90,  90, 105,  2, 0, 26},
};

// ─────────────────────────────────────────────────────────────────────────────
// Parser state machine
// ─────────────────────────────────────────────────────────────────────────────
enum ParserState { PS_SOF1, PS_SOF2, PS_CMD, PS_LEN, PS_DATA, PS_CRC };
ParserState pState = PS_SOF1;
uint8_t  pCmd = 0, pLen = 0, pIdx = 0;
uint8_t  pBuf[64];

// Servo target storage filled by processFrame(), applied in loop()
int16_t  incoming[SERVO_TOTAL_COUNT];
bool     packet_received = false;

void processFrame(uint8_t cmd, uint8_t* data, uint8_t len) {
  if (cmd == CMD_SLEEP) {
    if (len >= 1) sleeping = (data[0] != 0);
    return;
  }

  if (cmd == CMD_SET_SPEEDS) {
    uint8_t n = min((uint8_t)SERVO_TOTAL_COUNT, len);
    for (uint8_t i = 0; i < n; i++) {
      if (data[i] > 0) servos[i].step = data[i];
    }
    return;
  }

  if (cmd != CMD_SET_SERVOS) return;
  uint8_t n = min((uint8_t)SERVO_TOTAL_COUNT, len);
  for (uint8_t i = 0; i < n; i++) {
    incoming[i] = (int16_t)data[i];
  }
  packet_received = true;
}

void feedByte(uint8_t b) {
  switch (pState) {
    case PS_SOF1:
      pState = (b == 0xAA) ? PS_SOF2 : PS_SOF1; break;
    case PS_SOF2:
      if      (b == 0x55) pState = PS_CMD;
      else if (b == 0xAA) pState = PS_SOF2;  // 0xAA may start new frame
      else                pState = PS_SOF1;
      break;
    case PS_CMD:
      pCmd   = b; pState = PS_LEN;  break;
    case PS_LEN:
      pLen   = b; pIdx = 0;
      pState = (pLen > 0) ? PS_DATA : PS_CRC;  break;
    case PS_DATA:
      if (pIdx < sizeof(pBuf)) pBuf[pIdx++] = b;
      if (pIdx == pLen) pState = PS_CRC;  break;
    case PS_CRC: {
      uint8_t crc = pCmd ^ pLen;
      for (uint8_t i = 0; i < pLen; i++) crc ^= pBuf[i];
      if (b == crc) processFrame(pCmd, pBuf, pLen);
      pState = PS_SOF1;  break;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sensor state
// ─────────────────────────────────────────────────────────────────────────────
unsigned long ultrasonic_last    = 0;
bool          pir_last_state     = false;
unsigned long pir_last_ms        = 0;
unsigned long pir_heartbeat_ms   = 0;
unsigned long hall_last          = 0;
#define PIR_HEARTBEAT_MS  5000   // re-send current state every 5 s

int readUltrasonicCM() {
  digitalWrite(ULTRASONIC_TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(ULTRASONIC_TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRASONIC_TRIG_PIN, LOW);
  unsigned long dur = pulseIn(ULTRASONIC_ECHO_PIN, HIGH, ULTRASONIC_TIMEOUT_US);
  return (dur == 0) ? -1 : (int)(dur / 58);
}

// ─────────────────────────────────────────────────────────────────────────────
// Setup
// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  unsigned long now = millis();
  for (int i = 0; i < SERVO_TOTAL_COUNT; i++) {
    SmoothServo &s = servos[i];
    int16_t rest  = constrain(s.rest_angle, s.min_angle, s.max_angle);
    s.current     = rest;
    s.target      = rest;
    s.last_update = now;
    s.servo_obj.attach(s.pin);
    s.servo_obj.write(rest);
    incoming[i]   = 0;
  }

  pinMode(ULTRASONIC_TRIG_PIN, OUTPUT);
  pinMode(ULTRASONIC_ECHO_PIN, INPUT);
  pinMode(PIR_PIN, INPUT);

  pinMode(THUMB_HALL_PIN, INPUT);
  pinMode(INDEX_HALL_PIN, INPUT);
  pinMode(MIDDLE_HALL_PIN, INPUT);
  pinMode(RING_HALL_PIN, INPUT);
  pinMode(PINKY_HALL_PIN, INPUT);
}

// ─────────────────────────────────────────────────────────────────────────────
// Loop
// ─────────────────────────────────────────────────────────────────────────────
void loop() {
  // 1. Parse incoming serial bytes
  while (Serial.available()) {
    feedByte((uint8_t)Serial.read());
  }

  // 2. Apply new targets (only after first packet received)
  if (packet_received) {
    for (int i = 0; i < SERVO_TOTAL_COUNT; i++) {
      SmoothServo &s  = servos[i];
      int16_t     val = incoming[i];
      // val=0 → return to rest (same logic as xicro version)
      int16_t angle = (val == 0) ? s.rest_angle
                                 : constrain(val, s.min_angle, s.max_angle);
      s.target = angle;
    }
  }

  // 3. Smooth movement + write
  unsigned long now = millis();
  for (int i = 0; i < SERVO_TOTAL_COUNT; i++) {
    SmoothServo &s = servos[i];
    if (now - s.last_update < SMOOTH_INTERVAL_MS) continue;
    s.last_update = now;

    if (s.current < s.target) {
      s.current += s.step;
      if (s.current > s.target) s.current = s.target;
    } else if (s.current > s.target) {
      s.current -= s.step;
      if (s.current < s.target) s.current = s.target;
    } else {
      continue;  // position reached
    }
    s.servo_obj.write(constrain(s.current, s.min_angle, s.max_angle));
  }

  // 4. Ultrasonic — auto at ULTRASONIC_INTERVAL_MS (paused while sleeping)
  if (!sleeping && (now - ultrasonic_last >= ULTRASONIC_INTERVAL_MS)) {
    ultrasonic_last = now;
    int dist = readUltrasonicCM();
    if (dist > 0) {
      uint8_t buf[2] = { (uint8_t)(dist >> 8), (uint8_t)(dist & 0xFF) };
      sendFrame(CMD_ULTRASONIC, buf, 2);
    }
  }

  // 5. PIR — publish on state change; also heartbeat every 5 s (paused while sleeping)
  if (!sleeping && (now - pir_last_ms >= PIR_INTERVAL_MS)) {
    pir_last_ms = now;
    bool pir_state = (bool)digitalRead(PIR_PIN);
    bool changed   = (pir_state != pir_last_state);
    bool heartbeat = (now - pir_heartbeat_ms >= PIR_HEARTBEAT_MS);
    if (changed || heartbeat) {
      if (changed) pir_last_state = pir_state;
      if (heartbeat) pir_heartbeat_ms = now;
      uint8_t buf[1] = { (uint8_t)pir_last_state };
      sendFrame(CMD_PIR, buf, 1);
    }
  }

  // 6. Hall finger sensors — auto at HALL_INTERVAL_MS (paused while sleeping)
  if (!sleeping && (now - hall_last >= HALL_INTERVAL_MS)) {
    hall_last = now;
    uint16_t h[5] = {
      (uint16_t)analogRead(THUMB_HALL_PIN),
      (uint16_t)analogRead(INDEX_HALL_PIN),
      (uint16_t)analogRead(MIDDLE_HALL_PIN),
      (uint16_t)analogRead(RING_HALL_PIN),
      (uint16_t)analogRead(PINKY_HALL_PIN),
    };
    uint8_t buf[10];
    for (uint8_t i = 0; i < 5; i++) {
      buf[i * 2]     = (uint8_t)(h[i] >> 8);
      buf[i * 2 + 1] = (uint8_t)(h[i] & 0xFF);
    }
    sendFrame(CMD_HALL, buf, 10);
  }
}
