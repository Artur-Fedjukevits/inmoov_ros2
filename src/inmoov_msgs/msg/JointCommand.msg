# A servo command with arbitration metadata — published on /joint_cmd and
# arbitrated per joint by inmoov_control's arduino_comm_node (JointArbiter).
#
# A non-finite position is dropped before arbitration (it takes no lease).
# A command for a joint is applied if the joint has no owner, the owner's lease
# expired, the owner is this same source, or this command's priority is higher.
# With equal priority the current owner keeps the joint until its lease ends.
# A lease on eye_lr_* / eye_ud_* covers the mirrored eye too (EYE_SYNC).
#
# Priorities in use (higher wins):
#   90 calibration   servo_calibration_gui, face_expression_calibrator
#   80 remote        reserved: external control app (e.g. Android via rosbridge)
#   70 bt_command    behavior tree: look_direction / robot_control head
#   60 bt_scan       behavior tree: PIR / sound scan, face search, aim at human
#   50 tts_jaw       tts_node lip sync
#   40 head_tracker  vision_head_tracker_node (head + eyes)
#   30 expression    face_expressions_node
#   20 blink         behavior tree idle blink
#    0 legacy        plain JointState on /joint_command and /face_command

uint8 PRIORITY_CALIBRATION  = 90
uint8 PRIORITY_REMOTE       = 80
uint8 PRIORITY_BT_COMMAND   = 70
uint8 PRIORITY_BT_SCAN      = 60
uint8 PRIORITY_TTS_JAW      = 50
uint8 PRIORITY_HEAD_TRACKER = 40
uint8 PRIORITY_EXPRESSION   = 30
uint8 PRIORITY_BLINK        = 20
uint8 PRIORITY_LEGACY       = 0

string source        # who sends it, e.g. 'head_tracker' — ownership is per source
uint8 priority
float32 lease_sec    # keep the joints this long after this command; 0 = write only, don't hold
                     # (capped at 30 s; NaN/negative = 0)
bool release         # drop this source's ownership of cmd.name (positions ignored, may be empty)

# Positions in radians around each joint's center: rad = (deg - 90) * pi / 180.
# velocity (rad/s) is optional; empty or 0 = the joint's default (firmware table) speed.
sensor_msgs/JointState cmd
