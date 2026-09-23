/**
 * InMoovLeft.ino — Left Arduino Mega 2560
 *
 * Author: Artur Fedjukevits
 * Assisted by: Claude Code (Anthropic)
 * License: GNU General Public License v3.0 (see repository root LICENSE)
 *
 * Replaces xicro. Same servo logic, min/max/rest/step/pins/channels preserved.
 * Protocol: binary batch frame from ROS2 over USB Serial.
 *
 * Frame format: [0xAA][0x55][CMD][LEN][DATA...][CRC8]
 *   CRC8 = XOR of CMD, LEN, and all DATA bytes
 *
 * ROS2 → Arduino:
 *   CMD=0x01  SET_SERVOS: DATA = SERVO_TOTAL_COUNT bytes, degrees 0-180
 *   CMD=0x03  SLEEP:      DATA = 1 byte, 0=awake, 1=sleeping.
 *             While sleeping, ULTRASONIC/HALL telemetry stops.
 *
 * Arduino → ROS2:
 *   CMD=0x10  ULTRASONIC: DATA = uint16 big-endian, distance in cm
 *   CMD=0x12  HALL:       DATA = 5 x uint16 big-endian, raw analogRead (0-1023)
 *             order: [thumb, index, middle, ring, pinky] — same semantic order
 *             as the right arm, but MIDDLE/PINKY physical pins are swapped
 *             (A4/A2) to match this arm's hall sensor wiring.
 *
 * Packet order (28 bytes):
 *   Body GPIO (15) — ACT imitation learning joints:
 *   [0]  thumb_L    [1]  index_L    [2]  majeure_L  [3]  ring_L
 *   [4]  pinky_L    [5]  wrist_L    [6]  bicep_L    [7]  rotate_L
 *   [8]  shoulder_L [9]  omoplate_L [10] neck       [11] rothead
 *   [12] topstom    [13] midstom    [14] lowstom
 *
 *   Face GPIO (3):
 *   [15] eye_lr_L   [16] eye_ud_L   [17] jaw
 *
 *   Face PCA9685 (10):
 *   [18] eyelid_L_Upper (ch 6)    [19] eyelid_L_Lower (ch 7)
 *   [20] eyelid_R_Upper (ch 8)    [21] eyelid_R_Lower (ch 9)
 *   [22] eyebrow_L (ch 10)        [23] eyebrow_R (ch 11)
 *   [24] cheek_L (ch 14)          [25] cheek_R (ch 15)
 *   [26] forhead_L (ch 12)        [27] forhead_R (ch 13)
 */

#include "InMoovLeft.h"
#include <Wire.h>

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
#define SMOOTH_INTERVAL_MS      60
#define ULTRASONIC_INTERVAL_MS  250
#define ULTRASONIC_TIMEOUT_US   25000   // ~4m
#define HALL_INTERVAL_MS        100

#define ULTRASONIC_TRIG_PIN     64
#define ULTRASONIC_ECHO_PIN     63

// Hall finger sensors — same semantic order as right arm [thumb, index,
// middle, ring, pinky], but MIDDLE/PINKY pins swapped for this arm's wiring.
#define THUMB_HALL_PIN          A0
#define INDEX_HALL_PIN          A1
#define PINKY_HALL_PIN          A2
#define RING_HALL_PIN           A3
#define MIDDLE_HALL_PIN         A4

// PCA9685
#define PCA9685_ADDR  0x40
#define SERVO_FREQ    50
#define SERVOMIN      110
#define SERVOMAX      510

// ─────────────────────────────────────────────────────────────────────────────
// Protocol
// ─────────────────────────────────────────────────────────────────────────────
#define CMD_SET_SERVOS  0x01
#define CMD_SET_SPEEDS  0x02
#define CMD_SLEEP       0x03
#define CMD_DIAG_REQ    0x20
#define CMD_DIAG_RESP   0x21
#define CMD_ULTRASONIC  0x10
#define CMD_HALL        0x12

bool sleeping = false;   // set via CMD_SLEEP; gates ultrasonic/hall telemetry

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
// Servo table 
// ─────────────────────────────────────────────────────────────────────────────
//                               rest  min   max  step          drv             pin  ch   inv
SmoothServo servos[SERVO_TOTAL_COUNT] = {
  /* THUMB_L     pin 2  */ {0,0,   50,   0, 145,  2, 0, DRIVER_GPIO, Servo(),  2,  0, false},
  /* INDEX_L     pin 3  */ {0,0,    0,   0, 150,  2, 0, DRIVER_GPIO, Servo(),  3,  0, false},
  /* MAJEURE_L   pin 4  */ {0,0,    0,   0, 150,  2, 0, DRIVER_GPIO, Servo(),  4,  0, false},
  /* RING_L      pin 5  */ {0,0,    0,   0, 140,  2, 0, DRIVER_GPIO, Servo(),  5,  0, false},
  /* PINKY_L     pin 6  */ {0,0,    0,   0, 150,  2, 0, DRIVER_GPIO, Servo(),  6,  0, false},
  /* WRIST_L     pin 7  */ {0,0,  150,   0, 300,  2, 0, DRIVER_GPIO, Servo(),  7,  0, false},
  /* BICEP_L     pin 8  */ {0,0,    0,   0,  90,  1, 0, DRIVER_GPIO, Servo(),  8,  0, false},
  /* ROTATE_L    pin 9  */ {0,0,   90,  40, 180,  2, 0, DRIVER_GPIO, Servo(),  9,  0, false},
  /* SHOULDER_L  pin 10 */ {0,0,   20,   0, 180,  2, 0, DRIVER_GPIO, Servo(), 10,  0, false},
  /* OMOPLATE_L  pin 11 */ {0,0,   25,  25,  90,  2, 0, DRIVER_GPIO, Servo(), 11,  0, false},
  /* NECK        pin 12 */ {0,0,   40,   0, 100,  2, 0, DRIVER_GPIO, Servo(), 12,  0, false},
  /* ROTHEAD     pin 13 */ {0,0,   90,  30, 140,  1, 0, DRIVER_GPIO, Servo(), 13,  0, false},
  /* TOPSTOM     pin 28 */ {0,0,   83,  60, 110,  2, 0, DRIVER_GPIO, Servo(), 28,  0, false},
  /* MIDSTOM     pin 27 */ {0,0,   90,  60, 120,  2, 0, DRIVER_GPIO, Servo(), 27,  0, false},
  /* LOWSTOM     pin 29 */ {0,0,   90,   0, 180,  2, 0, DRIVER_GPIO, Servo(), 29,  0, false},
  /* EYE_LR      pin 22 */ {0,0,   90,  80, 100,  1, 0, DRIVER_GPIO, Servo(), 22,  0, false},
  /* EYE_UD      pin 24 */ {0,0,  100,  80, 110,  1, 0, DRIVER_GPIO, Servo(), 24,  0, false},
  /* JAW         pin 26 */ {0,0,   10,  10,  90, 10, 0, DRIVER_GPIO, Servo(), 26,  0, false},
  // PCA9685 — same channels and inversion as original
  /* EYELID_L_U  ch 6  INV */ {0,0,  85,  70,  95,  2, 0, DRIVER_PCA, Servo(), 0,  6,  true},
  /* EYELID_L_L  ch 7      */ {0,0,  85,  75,  95,  2, 0, DRIVER_PCA, Servo(), 0,  7, false},
  /* EYELID_R_U  ch 8      */ {0,0,  85,  65, 100,  2, 0, DRIVER_PCA, Servo(), 0,  8, false},
  /* EYELID_R_L  ch 9  INV */ {0,0,  85,  70,  95,  2, 0, DRIVER_PCA, Servo(), 0,  9,  true},
  /* EYEBROW_L   ch 10 INV */ {0,0,  90,  60, 110,  2, 0, DRIVER_PCA, Servo(), 0, 10,  true},
  /* EYEBROW_R   ch 11     */ {0,0,  80,  70, 105,  2, 0, DRIVER_PCA, Servo(), 0, 11, false},
  /* CHEEK_L     ch 14 INV */ {0,0, 100,  75, 115,  2, 0, DRIVER_PCA, Servo(), 0, 14,  true},
  /* CHEEK_R     ch 15     */ {0,0,  87,  68, 105,  2, 0, DRIVER_PCA, Servo(), 0, 15, false},
  /* FORHEAD_L   ch 12 INV */ {0,0,  90,  90, 110,  2, 0, DRIVER_PCA, Servo(), 0, 12,  true},
  /* FORHEAD_R   ch 13     */ {0,0,  85,  85, 105,  2, 0, DRIVER_PCA, Servo(), 0, 13, false},
};

Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(PCA9685_ADDR);

// ─────────────────────────────────────────────────────────────────────────────
// Helpers 
// ─────────────────────────────────────────────────────────────────────────────
inline uint16_t angleToPulse(int angle) {
  return (uint16_t)map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

void writeServo(SmoothServo &s, int angle) {
  angle = constrain(angle, s.min_angle, s.max_angle);
  if (s.driver == DRIVER_GPIO) {
    s.servo_obj.write(angle);
  } else {
    int a = s.inverted ? (180 - angle) : angle;
    pca9685.setPWM(s.channel, 0, angleToPulse(a));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parser state machine
// ─────────────────────────────────────────────────────────────────────────────
enum ParserState { PS_SOF1, PS_SOF2, PS_CMD, PS_LEN, PS_DATA, PS_CRC };
ParserState pState = PS_SOF1;
uint8_t  pCmd = 0, pLen = 0, pIdx = 0;
uint8_t  pBuf[64];

int16_t  incoming[SERVO_TOTAL_COUNT];
bool     packet_received = false;

// Scan I2C bus, read PCA9685 MODE1, respond with CMD_DIAG_RESP
void runDiag() {
  uint8_t buf[34];   // max 32 devices + n_devices + pca_mode1
  uint8_t n = 0;

  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      buf[1 + n++] = addr;
    }
  }
  buf[0] = n;

  // Read PCA9685 MODE1 register (reg 0x00)
  uint8_t mode1 = 0xFF;
  Wire.beginTransmission(PCA9685_ADDR);
  Wire.write(0x00);
  if (Wire.endTransmission(false) == 0) {
    Wire.requestFrom((uint8_t)PCA9685_ADDR, (uint8_t)1);
    if (Wire.available()) mode1 = Wire.read();
  }
  buf[1 + n] = mode1;

  sendFrame(CMD_DIAG_RESP, buf, 2 + n);
}

void processFrame(uint8_t cmd, uint8_t* data, uint8_t len) {
  if (cmd == CMD_DIAG_REQ) { runDiag(); return; }

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
unsigned long ultrasonic_last = 0;
unsigned long hall_last       = 0;

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

  Wire.begin();
  Wire.setClock(400000L);
  delay(100);
  pca9685.begin();
  pca9685.setPWMFreq(SERVO_FREQ);
  delay(200);

  unsigned long now = millis();
  for (int i = 0; i < SERVO_TOTAL_COUNT; i++) {
    SmoothServo &s = servos[i];
    int16_t rest  = constrain(s.rest_angle, s.min_angle, s.max_angle);
    s.current     = rest;
    s.target      = rest;
    s.last_update = now;
    incoming[i]   = 0;

    if (s.driver == DRIVER_GPIO) {
      s.servo_obj.attach(s.pin);
    }
    writeServo(s, rest);
  }

  pinMode(ULTRASONIC_TRIG_PIN, OUTPUT);
  pinMode(ULTRASONIC_ECHO_PIN, INPUT);

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
      // val=0 → return to rest 
      int16_t angle = (val == 0) ? s.rest_angle
                                 : constrain(val, s.min_angle, s.max_angle);
      // EYE_UD GPIO inversion (angle = min + max - angle)
      if (i == IDX_EYE_UD) angle = (s.min_angle + s.max_angle) - angle;
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
    writeServo(s, s.current);
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

  // 5. Hall finger sensors — auto at HALL_INTERVAL_MS (paused while sleeping)
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
