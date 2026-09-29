# inmoov_control

Motion-control layer for the InMoov humanoid robot. It talks to the two
Arduino Mega 2560 boards (right and left half of the robot) over USB serial
using a small **batch binary protocol**: one frame carries the target angle of
*every* servo on a board, sent at 50 Hz. This replaces the older
[xicro](https://github.com/ROBOTIS-GIT/xicro)-style setup, where every servo
and every sensor had its own ROS topic, with:

- one arbitrated command topic, `/joint_cmd` (`inmoov_msgs/JointCommand`: a
  `JointState` plus source / priority / lease), resolved **per joint** by the
  Arduino nodes (see *Joint arbitration* below); the older `/joint_command`
  and `/face_command` (`JointState`) still work as the lowest-priority source;
- two aggregated state topics, `/joint_states` and `/face_joint_states`,
  re-published by `joint_state_publisher` (note: these reflect the *commanded*
  state, there is no position feedback from the servos);
- individual sensor topics (ultrasonic, PIR, Hall finger sensors) decoded from
  the frames the firmware sends back.

## Joint arbitration

Several nodes move the same servos (head tracker, behavior-tree commands and
scans, TTS lip sync, face expressions, blinking, calibration tools). Each
command on `/joint_cmd` carries a `source`, a `priority` and a `lease_sec`;
[`joint_arbiter.py`](inmoov_control/joint_arbiter.py) decides per joint:

- applied if the joint is free, its lease expired, it's the same source, or
  the priority is strictly higher than the owner's (equal priority: the
  owner keeps it until its lease ends);
- `lease_sec > 0` makes the sender the owner for that long; `0` is a plain
  write that holds nothing; `release=true` drops the sender's ownership;
- a lease on one eye (`eye_lr_*` / `eye_ud_*`) covers the mirrored eye.

| Priority | Source | Joints | Lease |
|---|---|---|---|
| 90 | `calibration` (servo_calibration_gui, face_expression_calibrator) | any | 2 s, refreshed while moving |
| 80 | `remote` — reserved for an external control app (e.g. Android via rosbridge) | any | — |
| 70 | `bt_command` (look_direction / robot_control head) | head, torso | whole override, then released |
| 60 | `bt_scan` (PIR / sound scan, face search, aim at human) | head / torso | scan phase, released at the end |
| 50 | `tts_jaw` | jaw | 0.5 s per audio chunk |
| 40 | `head_tracker` | head, eyes | 0.5 s per tick |
| 30 | `expression` | expression joints | 1 s while animating; static holds none |
| 20 | `blink` | eyelids | none |
| 0 | `legacy` (`/joint_command`, `/face_command`) | any | none |

A command without `velocity` (or 0) moves the joint at its **default speed**
(the firmware table step, `DEFAULT_STEPS` in `arduino_{left,right}_node.py`,
checked against the sketches by `test/test_servo_tables.py`) — speed is not
inherited from the previous sender. Accepted commands are echoed on
`/joint_commanded` (what nodes tracking the current pose should use); live
owners and the rejected count are on `/arduino_{left,right}/joint_owners`.

The package also contains the face expression library / node
(`face_expressions_node`) and a Tk-based expression calibrator.

Everything here is Python (`ament_python`); the firmware lives outside this
package in [`Arduino/`](../../Arduino/) at the workspace root
(`InMoovLeft/InMoovLeft.ino`, `InMoovRight/InMoovRight.ino`).

All nodes are ROS2 **managed-lifecycle nodes** (`rclpy.lifecycle.LifecycleNode`)
and stay in `Unconfigured` until something drives them through
`configure` → `activate` (normally `lifecycle_manager` in
[`inmoov_bringup`](../inmoov_bringup/README.md)).

## Nodes

Executables registered in [`setup.py`](setup.py):

| Executable | Module |
|---|---|
| `arduino_right_node` | `inmoov_control.arduino_right_node:main` |
| `arduino_left_node` | `inmoov_control.arduino_left_node:main` |
| `joint_state_publisher` | `inmoov_control.joint_state_publisher:main` |
| `face_expressions_node` | `inmoov_control.face_expressions_node:main` |
| `face_expression_calibrator` | `inmoov_control.face_expression_calibrator:main` |
| `urdf_bridge_node` | `inmoov_control.urdf_bridge_node:main` |

### `arduino_right_node` / `arduino_left_node`

Sources: [`inmoov_control/arduino_comm_node.py`](inmoov_control/arduino_comm_node.py)
(shared base class `ArduinoCommNode`),
[`inmoov_control/arduino_right_node.py`](inmoov_control/arduino_right_node.py),
[`inmoov_control/arduino_left_node.py`](inmoov_control/arduino_left_node.py).

Each node owns one serial port. The subclass only declares which joints live
on its board (`BODY_JOINTS`, `FACE_JOINTS`, each entry
`(joint_name, center_deg, rest_deg)` in **packet order**) and which sensors the
board has; the base class does all the I/O.

| | `arduino_right_node` | `arduino_left_node` |
|---|---|---|
| Node name (in code) | `arduino_right_node` | `arduino_left_node` |
| Default port | `/dev/serial/by-path/pci-0000:c6:00.3-usb-0:5:1.0-port0` | `/dev/serial/by-path/pci-0000:c6:00.3-usb-0:1.3:1.0-port0` |
| Firmware | `InMoovRight.ino` | `InMoovLeft.ino` |
| Body joints (packet bytes) | 11 (`0–10`) | 15 (`0–14`) |
| Face joints (packet bytes) | 3 (`11–13`) | 13 (`15–27`) |
| Total servos per SET_SERVOS frame | 14 | 28 |
| Ultrasonic topic | `ultrasonic_right_distance` | `ultrasonic_left_distance` |
| PIR topic | `pir_state` | – |
| Hall topic | `hall_right_raw` | `hall_left_raw` |

The bringup launch file overrides the node name (`arduino_right`,
`arduino_left`), so the lifecycle services are
`/arduino_right/change_state` etc.

**Parameters**

There are **no ROS parameters**. The serial port is a command-line argument
(parsed with `argparse.parse_known_args`, so ROS arguments can follow):

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `--port` | string | see table above | Serial device of the Arduino. |

The baud rate is fixed at `115200` (constructor default of `ArduinoCommNode`,
matching `Serial.begin(115200)` in both sketches). Read timeout is 0.1 s.

**Topics**

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/joint_cmd` | `inmoov_msgs/JointCommand` | subscribe (depth 20) | Arbitrated per joint (see *Joint arbitration*). `cmd.name[i]` selects the joint, `cmd.position[i]` is radians relative to the joint centre (90° for every joint): `deg = rad*180/π + 90`, rounded and clamped to 0–180. Names not belonging to this board (or its mirrored eye) are ignored, so both nodes can receive the same message. Optional `cmd.velocity[i]` (rad/s) sets the per-servo speed, otherwise the default speed. |
| `/joint_command`, `/face_command` | `sensor_msgs/JointState` | subscribe (depth 10) | Legacy: same encoding, handled as source `legacy`, priority 0, no lease. |
| `/joint_commanded` | `sensor_msgs/JointState` | publish (lifecycle) | The commands this board accepted after arbitration. |
| `~/joint_owners` | `std_msgs/String` (JSON) | publish (lifecycle, 1 Hz) | `{owners: {joint: {source, priority, remaining_sec}}, rejected_total}`. |
| `~/failsafe` | `std_msgs/Bool` | publish (lifecycle, latched) | Firmware host-loss failsafe entered / left. |
| `/robot_sleep` | `std_msgs/Bool` | subscribe (`TRANSIENT_LOCAL`, depth 1) | Latched. On change it is forwarded as a `CMD_SLEEP` frame; while asleep the firmware stops ultrasonic / PIR / Hall telemetry. |
| `ultrasonic_{left,right}_distance` | `std_msgs/Int16` | publish (lifecycle publisher, depth 10) | Distance in cm. Published by the topic name relative to the node namespace (i.e. `/ultrasonic_left_distance` with the default empty namespace). |
| `pir_state` | `std_msgs/Bool` | publish (right only) | PIR motion state. |
| `hall_{left,right}_raw` | `std_msgs/Int16MultiArray` | publish | 5 raw `analogRead` values (0–1023), order `[thumb, index, middle, ring, pinky]`. |

**Behavior**

- `on_configure`: creates subscriptions and lifecycle publishers only; no
  serial I/O yet.
- `on_activate`: checks that the port exists on the filesystem
  (`os.path.exists`); if not, or if opening fails, the transition returns
  `FAILURE` so the lifecycle manager can retry. On success it waits 2 s for the
  Arduino bootloader (opening the port resets the board), starts a 50 Hz TX
  timer (`0.02 s`) and an RX thread that reads 64-byte chunks and feeds them
  into `FrameParser`.
- TX (every 20 ms): sends, in order, a `CMD_SLEEP` frame and a
  `CMD_SET_SPEEDS` frame (when the sleep state / a speed changed, after every
  (re)connect and every 2 s as a resync — both are idempotent in the firmware,
  so a board that reset on its own gets its sleep flag and speeds back), then
  always a `CMD_SET_SERVOS` frame with all body + face angles. The initial
  targets are the `rest_deg` values from the joint tables, so activating the
  node immediately drives the servos to those rests.
- Speeds: a non-zero `velocity[i]` is converted with
  `step = clamp(round(|deg/s| * 0.060), 1, 255)` (degrees per firmware smoothing
  tick, `SMOOTH_INTERVAL_MS = 60` — see
  [`protocol.py`](inmoov_control/protocol.py)) and sent for **all** servos of
  the board in one frame (`0` = keep the firmware's current step). A zero or
  absent velocity leaves the speed unchanged.
- Eye mirroring (`EYE_SYNC`): a command for `eye_lr_L` is also applied to
  `eye_lr_R` and vice-versa; the same for `eye_ud_L` / `eye_ud_R`. Because both
  boards receive both topics, each board applies only the joint it owns.
- `on_deactivate` / `on_shutdown`: send one `CMD_SET_SERVOS` frame with the rest
  positions (then wait 150 ms), stop the RX thread (2 s join), destroy the TX
  timer and close the port.
- Link loss: a serial read/write error (board unplugged, USB reset) closes the
  port and the RX thread reopens it in the background with a growing back-off
  (2 s → 30 s); the node stays `active` and logs `Serial link lost` /
  `Serial link restored`.
- RX: `CMD_ULTRASONIC` (uint16 big-endian, cm), `CMD_PIR` (1 byte),
  `CMD_HALL` (5 × uint16 big-endian) are decoded and published.
  `CMD_ACK` / `CMD_DIAG_RESP` are defined in the protocol but not handled by
  the nodes.

**Joint tables.** The exact order of `BODY_JOINTS` / `FACE_JOINTS` is the
packet layout and **must match the sketches** — see [Protocol](#protocol) for
the full lists.

**Known issues / TODOs (from code and cross-checks)**

- The Python joint tables mirror the firmware tables (see
  [Protocol](#protocol)); the firmware is the authority for limits. A value of
  `0` in `SET_SERVOS` means "go to the firmware rest angle".

### `joint_state_publisher`

Source: [`inmoov_control/joint_state_publisher.py`](inmoov_control/joint_state_publisher.py).

Lifecycle node that keeps the last *commanded* position of every joint and
re-publishes them at 50 Hz as two aggregated `JointState` messages with a proper
`header.stamp`. Intended for rviz2, rosbag data collection and policy
inference nodes.

It tracks the joints of both Arduino classes (joint lists are imported from
`ArduinoRightNode` / `ArduinoLeftNode`): 11 + 15 = 26 body joints and
3 + 13 = 16 face joints. All positions start at `0.0` rad.

**Parameters:** none.

**Topics**

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/joint_commanded` | `sensor_msgs/JointState` | subscribe (depth 20) | Accepted commands from both boards; positions (rad) stored per joint name (eyes mirrored); unknown names ignored. |
| `/joint_states` | `sensor_msgs/JointState` | publish (lifecycle, depth 10, 50 Hz) | `frame_id = base_link`; names/positions of all body joints. |
| `/face_joint_states` | `sensor_msgs/JointState` | publish (lifecycle, depth 10, 50 Hz) | `frame_id = head_link`; names/positions of all face joints. |

**Known issues**

- The state is *commanded*, not measured (no encoders / feedback, no echo
  channel in the protocol). Joints that have never been commanded are reported
  at their rest position (`rest_deg` of the joint tables).
- Because the values are re-published regardless of whether the servo really
  moved (e.g. clamped by firmware limits), the reported pose can differ from the
  real one.

### `face_expressions_node`

Source: [`inmoov_control/face_expressions_node.py`](inmoov_control/face_expressions_node.py).

Face expression library (`FaceExpressions`) plus a thin lifecycle node around
it. Expressions were ported from MRL InMoov2 (`gestures/faceExpressions.py`,
`EyebrowMovements.py`, `EyelidMovements.py`, `CheekMovements.py`,
`EyeMovements.py`). Each expression is a `{joint_name: degrees}` dict; joints
not listed keep their current position. Face commands are published on
`/joint_cmd` (source `expression`, priority 30 — the head tracker keeps the
eyes and TTS keeps the jaw while they hold them).

Available expressions: `neutral`, `angry`, `wink`, `disgust`, `fear`, `happy`,
`smile`, `sad`, `sigh`, `sorry`, `suspicious`, `thinking`, `unamused`,
`surprise`, `sleeping`. Aliases accepted by `/face_expression` (map in
`FaceExpressions.EXPRESSION_MAP`): `anger`→`angry`, `surprised`→`surprise`,
`contempt`→`happy`, `anxiety`→`angry`, `disappointment`→`sad`, `frown`→`sad`,
`gasp`→`surprise`, `excited`→`surprise`, `chuckle`→`smile`, `grin`→`smile`,
`helplessness`→`sorry`.

Two ways to play an expression:

- `/face_expression` — *one-shot*. The animated gestures (`wink`, `happy`,
  `smile`, `sigh`, `sorry`, `thinking`, `surprise`) contain `time.sleep()` and
  return the face to rest by themselves; the others (`neutral`, `angry`,
  `disgust`, `fear`, `sad`, `suspicious`, `unamused`, `sleeping`) just set the
  pose and stay until the next command.
- `/face_expression_hold` — *static hold*. Applies the pose exactly and never
  reverts by itself (used by `tts_node` to keep an expression for the whole
  spoken sentence). `hold('neutral')` explicitly sends the rest pose. Only the
  canonical expression names are accepted here (no aliases).

Both callbacks run the work on a single-worker `ThreadPoolExecutor`, so
animated gestures and holds are serialised (no servo races) and the `sleep`s
never block the ROS executor.

**Parameters:** none.

**Topics**

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/face_expression` | `std_msgs/String` | subscribe (depth 10) | Name (stripped, lower-cased); unknown names log a warning. |
| `/face_expression_hold` | `std_msgs/String` | subscribe (depth 10) | Name from `EXPRESSIONS_DATA` only. |
| `/joint_cmd` | `inmoov_msgs/JointCommand` | publish (regular publisher, depth 10) | Consumed by `arduino_left_node` / `arduino_right_node`. |

Example:

```bash
ros2 topic pub --once /face_expression std_msgs/msg/String "data: happy"
ros2 topic pub --once /face_expression_hold std_msgs/msg/String "data: neutral"
```

**Calibration file.** At import time the module loads the user calibration
file (`$INMOOV_FACE_CALIBRATION`, default
`~/.config/inmoov/face_expressions_calibration.json`, written by the
calibrator GUI and surviving rebuilds) or, if it does not exist, the
`face_expressions_calibration.json` shipped with the package (installed via
`package_data`), and *replaces* the default entry of every expression it contains (an expression in
the JSON overrides the whole default dict, it is not merged joint by joint).
The repository ships a JSON with all 15 expressions. The node logs
`calib: <path>` or `calib: defaults` on configure. To make a user calibration
the shipped default, copy it over `inmoov_control/face_expressions_calibration.json`.

**Known issues**

- Actions are not gated on the lifecycle state: the subscriptions and the
  publisher are plain (not lifecycle) entities created in `on_configure`, so
  expressions execute as soon as the node is *configured*, even before
  `activate`.
- Mouth/jaw lip-sync is not done here (`tts_node` publishes the jaw on
  `/joint_cmd` at a higher priority); this node only sets the jaw for
  `happy`/`smile`/`surprise`.

### `face_expression_calibrator`

Source: [`inmoov_control/face_expression_calibrator.py`](inmoov_control/face_expression_calibrator.py).

Interactive Tk GUI (not a lifecycle node; requires a display and `tkinter`) to
tune the face expressions on the real robot. Left: list of expressions. Right:
one slider per face servo (16), grouped (eyelids, eyebrows, cheeks, forehead,
eyes, mouth), with a checkbox saying whether the servo is part of the current
expression. Moving a slider publishes that servo on `/joint_cmd` (source
`calibration`, priority 90 — overrides the running stack) in real time
("auto send" toggle); *Save* writes only the checked servos of the current
expression into the user calibration file (see above); *Reset* restores the
built-in (MRL) defaults for the expression. The GUI labels are in Russian.

Run (ROS2 sourced, the face expression nodes and Arduino nodes must be running
so the servos react):

```bash
cd ~/ros2_ws && source install/setup.bash
ros2 run inmoov_control face_expression_calibrator
```

**Parameters:** none. **Topic:** publishes `/joint_cmd` (`inmoov_msgs/JointCommand`, priority 90).
Servo limits/rests are imported from `face_expressions_node` (`_MN`, `_MX`,
`FACE_REST`), so they are defined in one place.

### `urdf_bridge_node` (servo ↔ URDF, Android app)

Source: [`inmoov_control/urdf_bridge_node.py`](inmoov_control/urdf_bridge_node.py),
conversion library [`inmoov_control/servo_urdf_map.py`](inmoov_control/servo_urdf_map.py)
(pure Python), table [`config/servo_urdf_map.yaml`](config/servo_urdf_map.yaml).

Everything above speaks **servo** joints (`bicep_L`, radians around 90°). The robot
model `inmoov_i2.urdf` (inmoov_description), RViz, MoveIt and the Android app speak
**URDF** joints in URDF radians. This lifecycle node (launch name `urdf_bridge`,
tier 2) converts both ways:

| Topic / service | Type | Direction | Notes |
|---|---|---|---|
| `/joint_states`, `/face_joint_states` | `sensor_msgs/JointState` | subscribe | servo names (commanded pose) |
| `/urdf_joint_states` | `sensor_msgs/JointState` | publish, 25 Hz | 41 URDF joints (no mimic joints), clamped to URDF limits |
| `/urdf_joint_cmd` | `inmoov_msgs/JointCommand` | subscribe | URDF names/radians; source, lease, release are passed through, priority is capped at `PRIORITY_REMOTE` (80); right-eye joints are ignored (see below) |
| `/joint_cmd` | `inmoov_msgs/JointCommand` | publish | the converted command (servo names, servo rad, velocity converted) |
| `/urdf_bridge/get_map` | `std_srvs/Trigger` | service | `message` = JSON of the table + current servo degrees |
| `/urdf_bridge/set_calibration` | `std_msgs/String` | subscribe | JSON `{"servo": "bicep_L", "points": [[deg, rad], ...]}` or `{"servo": ..., "reset": true}` |
| `/urdf_bridge/status` | `std_msgs/String` | publish, latched | JSON: map source, number of calibrated joints, last calibration result (`persisted`: saved to disk or only applied until restart) |

**Table.** One entry per servo: `urdf` joint, firmware `servo_range`/`servo_rest`,
`urdf_limits`, and `points: [[servo_deg, urdf_rad], ...]` — piecewise linear,
≥ 2 points, strictly monotonic (2 points = offset/direction/scale; more = a
non-linear linkage). 41 servos are mapped; `lowstom` is not in the URDF. The
shipped table is an **uncalibrated guess** (`verified: false`): arm/neck/torso
1:1 around the firmware rest, fingers/face stretched over the URDF range, all
directions +1. Calibrated joints are written to
`~/.config/inmoov/servo_urdf_map.yaml` (`$INMOOV_SERVO_URDF_MAP`), which overrides
the shipped table joint by joint. With the `map_file` parameter set, that file
is loaded as is (no merge) and calibrations are saved back into it, whole.

**Eyes.** `arduino_comm_node` mirrors the eyes (EYE_SYNC), so a command for one
eye moves both. `/urdf_joint_cmd` drives only the left (leading) eye:
`i02_head_right_eye_{horizontal,vertical}_joint` are ignored, otherwise two eyes
in one command would race. `/urdf_joint_states` still reports both.

**Calibration** is done from the Android app (tab *Робот → Калибровка*): the servo
slider drives the servo directly on `/joint_cmd` (source `calibration`, priority
90), the model slider is adjusted until the 3D model matches the real robot, each
match is a point; *Сохранить* publishes `/urdf_bridge/set_calibration`. A table
sent with `"verified": false` is applied but not saved (`persisted: false`).

**rosbridge.** `inmoov.launch.py` also starts `rosbridge_websocket` (plain node,
args `rosbridge:=true`, `rosbridge_port:=9090`; needs `ros-jazzy-rosbridge-server`).
There is no authentication, so the launch whitelists only what the app uses:
publish `/urdf_joint_cmd`, `/joint_cmd`, `/urdf_bridge/set_calibration`;
subscribe `/urdf_joint_states`, `/urdf_bridge/status` and, for the app's
diagnostics, `/joint_states`, `/face_joint_states`, `/joint_commanded`,
`/arduino_*/failsafe`, `/arduino_*/joint_owners`; services
`/urdf_bridge/get_map` and `/*/get_state` (read-only lifecycle state —
`change_state` stays refused); no actions. Anything else is refused (silently
for the client, a `No match found` warning in the log) — extend the globs
in `inmoov.launch.py` when the app needs more. `/joint_cmd` is open for the
calibration (priority 90), so still keep the robot on a trusted network. For RViz on the robot:
`ros2 run robot_state_publisher robot_state_publisher --ros-args -r joint_states:=/urdf_joint_states -p robot_description:=...`
(the plain `/joint_states` carries servo names, which robot_state_publisher does not know).

## Protocol

Source of truth: [`inmoov_control/protocol.py`](inmoov_control/protocol.py),
cross-checked against `InMoovLeft.ino` / `InMoovRight.ino`. Unit tests:
[`test/test_protocol.py`](test/test_protocol.py).

### Frame format (both directions)

```
[0xAA][0x55][CMD][LEN][DATA... (LEN bytes)][CRC8]
```

- `SOF` = `0xAA 0x55`.
- `CRC8` is the **XOR** of `CMD`, `LEN` and all `DATA` bytes (not a
  polynomial CRC; the SOF bytes are not included).
- The parser is a byte-wise state machine on both sides (`FrameParser` in
  Python; `feedByte()` in the sketches). A second `0xAA` while waiting for
  `0x55` re-arms the start of frame; frames with a bad CRC are dropped
  silently. The firmware receive buffer is 64 bytes.
- Multi-byte values are **big-endian**.

### Commands ROS → Arduino

| CMD | Name | DATA |
|---|---|---|
| `0x01` | `CMD_SET_SERVOS` | One byte per servo, degrees `0–180`, in the packet order below. Value `0` = go to the servo's firmware `rest_angle`; otherwise clamped to `[min, max]`. The Python builder clamps to `0–180`. |
| `0x02` | `CMD_SET_SPEEDS` | One byte per servo: step in degrees per smoothing tick (`SMOOTH_INTERVAL_MS = 60`). `0` = keep the current step, `1–255` = new step. `deg/s ≈ step * 1000 / 60` (step 10 ≈ 167 °/s, step 2 ≈ 33 °/s). |
| `0x03` | `CMD_SLEEP` | 1 byte: `0` awake, `1` sleeping. While sleeping the firmware stops ultrasonic / PIR / Hall telemetry. |
| `0x20` | `CMD_DIAG_REQ` | none — request an I2C scan + PCA9685 check. **Left firmware only**; no Python builder in `protocol.py` (used by `test/test_i2c_pca9685.py` via `build_frame`). |

### Events Arduino → ROS

| CMD | Name | DATA | Firmware rate |
|---|---|---|---|
| `0x10` | `CMD_ULTRASONIC` | `uint16` distance in cm | every 250 ms; nothing is sent when the echo times out (25 ms, ~4 m) |
| `0x11` | `CMD_PIR` | 1 byte, 0/1 (right board only) | on change (checked every 100 ms) and a heartbeat every 5 s |
| `0x12` | `CMD_HALL` | 5 × `uint16` raw `analogRead` (0–1023), `[thumb, index, middle, ring, pinky]` | every 100 ms |
| `0x21` | `CMD_DIAG_RESP` | `[n_devices, addr0 … addrN-1, pca_mode1]` (`pca_mode1 = 0xFF` = PCA9685 not found) | reply to `CMD_DIAG_REQ` (left firmware) |
| `0xFF` | `CMD_ACK` | 1 byte, echoed CMD | defined in `protocol.py`, **not sent** by either sketch |

Left Hall wiring note: the semantic order is the same as on the right arm, but
the physical `MIDDLE` / `PINKY` analog pins are swapped on the left board
(`PINKY=A2`, `MIDDLE=A4`); the firmware already reorders them.

### Servo order — Right Arduino (`InMoovRight.ino`, 14 servos)

Columns `rest / min / max` are from the **firmware** servo table
(authoritative); "Python rest" is `rest_deg` in `BODY_JOINTS`/`FACE_JOINTS`.

| Byte | Joint | Pin | Firmware rest / min / max | Python rest |
|---|---|---|---|---|
| 0 | `thumb_R` | 2 | 60 / 0 / 140 | 60 |
| 1 | `index_R` | 3 | 40 / 0 / 160 | 40 |
| 2 | `middle_R` | 4 | 40 / 0 / 160 | 40 |
| 3 | `ring_R` | 5 | 30 / 0 / 150 | 30 |
| 4 | `pinky_R` | 6 | 40 / 0 / 170 | 40 |
| 5 | `wrist_R` | 7 | 90 / 0 / 180 | 90 |
| 6 | `bicep_R` | 8 | 0 / 0 / 80 | 0 |
| 7 | `rotate_R` | 9 | 90 / 40 / 180 | 90 |
| 8 | `shoulder_R` | 10 | 30 / 0 / 180 | 30 |
| 9 | `omoplate_R` | 11 | 10 / 10 / 80 | 10 |
| 10 | `rollneck` | 13 | 80 / 50 / 115 | 80 |
| 11 | `eye_lr_R` (face) | 22 | 90 / 80 / 100 | 90 |
| 12 | `eye_ud_R` (face) | 24 | 100 / 85 / 115 | 100 |
| 13 | `upperLip` (face) | 26 | 90 / 90 / 105 | 90 |

Sensors on this board: ultrasonic (TRIG 64 / ECHO 63), PIR (pin 23), Hall
fingers on A0–A4 (`thumb, index, middle, ring, pinky`).

### Servo order — Left Arduino (`InMoovLeft.ino`, 28 servos)

| Byte | Joint | Driver / pin | Firmware rest / min / max | Python rest |
|---|---|---|---|---|
| 0 | `thumb_L` | GPIO 2 | 50 / 0 / 145 | 50 |
| 1 | `index_L` | GPIO 3 | 0 / 0 / 150 | 0 |
| 2 | `majeure_L` | GPIO 4 | 0 / 0 / 150 | 0 |
| 3 | `ring_L` | GPIO 5 | 0 / 0 / 140 | 0 |
| 4 | `pinky_L` | GPIO 6 | 0 / 0 / 150 | 0 |
| 5 | `wrist_L` | GPIO 7 | 150 / 0 / 300 | 150 |
| 6 | `bicep_L` | GPIO 8 | 0 / 0 / 90 | 0 |
| 7 | `rotate_L` | GPIO 9 | 90 / 40 / 180 | 90 |
| 8 | `shoulder_L` | GPIO 10 | 20 / 0 / 180 | 20 |
| 9 | `omoplate_L` | GPIO 11 | 25 / 25 / 90 | 25 |
| 10 | `neck` | GPIO 12 | 40 / 0 / 100 | 40 |
| 11 | `rothead` | GPIO 13 | 90 / 30 / 140 | 90 |
| 12 | `topstom` | GPIO 28 | 83 / 60 / 110 | 83 |
| 13 | `midstom` | GPIO 27 | 90 / 60 / 120 | 90 |
| 14 | `lowstom` | GPIO 29 | 90 / 0 / 180 | 90 |
| 15 | `eye_lr_L` (face) | GPIO 22 | 90 / 80 / 100 | 90 |
| 16 | `eye_ud_L` (face) | GPIO 24 (inverted in firmware) | 100 / 80 / 110 | 100 |
| 17 | `jaw` (face) | GPIO 26 | 10 / 10 / 90 | 10 |
| 18 | `eyelid_L_Upper` | PCA ch 6 (INV) | 85 / 70 / 95 | 85 |
| 19 | `eyelid_L_Lower` | PCA ch 7 | 85 / 75 / 95 | 85 |
| 20 | `eyelid_R_Upper` | PCA ch 8 | 85 / 65 / 100 | 85 |
| 21 | `eyelid_R_Lower` | PCA ch 9 (INV) | 85 / 70 / 95 | 85 |
| 22 | `eyebrow_L` | PCA ch 10 (INV) | 90 / 60 / 110 | 90 |
| 23 | `eyebrow_R` | PCA ch 11 | 80 / 70 / 105 | 80 |
| 24 | `cheek_L` | PCA ch 14 (INV) | 100 / 75 / 115 | 100 |
| 25 | `cheek_R` | PCA ch 15 | 87 / 68 / 105 | 87 |
| 26 | `forhead_L` | PCA ch 12 (INV) | 90 / 90 / 110 | 90 |
| 27 | `forhead_R` | PCA ch 13 | 85 / 85 / 105 | 85 |

PCA9685 at I2C address `0x40`, 50 Hz, pulse range `110–510` counts,
I2C clock 400 kHz; `INV` channels are written as `180 - angle`. The left board
has only an ultrasonic sensor (TRIG 64 / ECHO 63) and Hall fingers (A0–A4,
middle/pinky swapped); no PIR.

The Python `rest_deg` values and trailing min/max comments in
`arduino_{left,right}_node.py` (and the slider table in
`test/servo_calibration_gui.py`) mirror the firmware tables above — the
firmware is the source of truth, so change it there first and then sync the
Python side. Note `wrist_L` max is 300 in the firmware, but a SET_SERVOS byte
is 0–180 and `Servo.write` caps at 180, so the effective max is 180.

## Launch

The package has no launch file of its own. `arduino_right`, `arduino_left`,
`joint_state_publisher` and `face_expressions` are started by
[`inmoov_bringup/launch/inmoov.launch.py`](../inmoov_bringup/launch/inmoov.launch.py)
as `LifecycleNode`s (`respawn=True`) and configured/activated by
`lifecycle_manager`; the serial ports are the `port_right` / `port_left`
launch arguments there. To run a node by hand:

```bash
ros2 run inmoov_control arduino_left_node --port /dev/ttyACM1
ros2 lifecycle set /arduino_left_node configure && ros2 lifecycle set /arduino_left_node activate
```

Testing without the robot / servo power (the Arduino only needs USB power):

```bash
python3 -m pytest src/inmoov_control/test/test_protocol.py -v      # no hardware
python3 -m pytest src/inmoov_control/test/test_joint_arbiter.py -v # no hardware
python3 -m pytest src/inmoov_control/test/test_servo_urdf_map.py -v # no ROS, no hardware
python3 src/inmoov_control/test/serial_loopback_test.py --port /dev/ttyACM0 [--left]
python3 src/inmoov_control/test/test_i2c_pca9685.py --port <left-board-port>
python3 src/inmoov_control/test/servo_calibration_gui.py            # publishes /joint_cmd (priority 90)
```

Helper scripts in [`test/`](test/): `serial_loopback_test.py` (sends
`SET_SERVOS` frames and prints replies; `--left` = 28-byte packet, default
right = 14), `test_i2c_pca9685.py` (I2C scan / PCA9685 `MODE1` check through
`CMD_DIAG_REQ`), `servo_calibration_gui.py` (Tk slider GUI for all joints).
Some of these scripts still have Russian docstrings/GUI text.

## Requirements / Setup

- ROS2 Jazzy (uses `rclpy` lifecycle nodes), build type `ament_python`.
- Runtime dependencies (from [`package.xml`](package.xml)): `rclpy`,
  `std_msgs`, `sensor_msgs`, `python3-serial` (pyserial). The calibrator
  additionally needs `python3-tk`.
- Firmware (Arduino IDE / `arduino-cli`, board **Arduino Mega 2560**, both
  sketches use `Serial.begin(115200)`):
  - `Arduino/InMoovRight/InMoovRight.ino` (+ `InMoovRight.h`) → **right** board
    (`arduino_right_node`). Uses the standard `Servo` library.
  - `Arduino/InMoovLeft/InMoovLeft.ino` (+ `InMoovLeft.h`) → **left** board
    (`arduino_left_node`). Uses `Servo`, `Wire` and `Adafruit_PWMServoDriver`
    (with `Adafruit_BusIO`); copies of the two Adafruit libraries are in
    `Arduino/libraries/`.
  Flashing the wrong sketch onto a board makes the packet length not match
  (14 vs 28 bytes); the firmware silently uses `min(SERVO_TOTAL_COUNT, len)`
  bytes, so servos would be driven with the wrong joints.
- Serial ports: the defaults are `by-path` names that depend on the physical
  USB port / PCI address of the author's computer (`pci-0000:c6:00.3-usb-0:5:...`
  right, `...-usb-0:1.3:...` left). On another machine, find the ports with
  `ls -l /dev/serial/by-path/` (or `/dev/ttyACM*`) and pass `port_right` /
  `port_left` (or `--port`). `by-path` names are used because they do not
  depend on enumeration order (`ttyACM0/1` may swap between boots).
- Permissions: the user needs access to the serial devices, typically
  `sudo usermod -aG dialout $USER` (log out/in afterwards).
- Note that opening a Mega's serial port toggles DTR and resets the board;
  the node waits 2 s after opening for the bootloader.
- Build: `colcon build --packages-select inmoov_control` (no
  `--symlink-install` is used in this workspace).

## Known issues to verify

- Serial reconnect and the 2 s sleep/speed resync were tested with an emulated
  board (pty); verify on the robot by unplugging a board while it is active.
- `joint_state_publisher` reports commanded, not measured, state.
- The firmware defines `CMD_ACK` in `protocol.py`, but neither sketch sends it;
  the nodes therefore have no delivery confirmation for commands.

## License

GNU General Public License v3.0 (GPL-3.0-only) — see the repository root
[`LICENSE`](../../LICENSE).

Author: Artur Fedjukevits. Assisted by: Claude Code (Anthropic).
