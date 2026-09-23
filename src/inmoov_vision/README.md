# inmoov_vision

Vision pipeline for the InMoov humanoid robot. It covers everything the robot
"sees": USB cameras in the eyes, face detection / tracking / recognition,
facial emotion, a photo gallery for face enrolment, head/eye tracking of the
interlocutor, and body / object perception with an OAK-D Lite depth camera.

All nodes are ROS2 **managed lifecycle nodes** (`rclpy.lifecycle.LifecycleNode`)
and are started and supervised by the `lifecycle_manager` of
`inmoov_bringup` (see that package's README and [Launch](#launch)).

## Dual-eye topology

The robot has **two USB cameras, one in each eye**, both opened by a single
node (`face_capture_node`). Face detection and tracking run as **two instances
of the same executables**, one per eye, parameterized by `camera_side`
(`left` | `right`):

- `face_detection_node_left` / `face_detection_node_right` →
  `/face/detections/{left,right}`
- `face_tracker_node_left` / `face_tracker_node_right` →
  `/face/tracks/{left,right}`

Downstream nodes read the **left eye as primary** and the **right eye as
fallback**:

- `face_detection_node_right` is configured (`fallback_for`
  `/face/detections/left`, `primary_timeout_sec` 1.5) as an **on-demand
  standby**: it only starts running insightface when the left detections topic
  has been silent for more than 1.5 s (saves a lot of CPU).
- `face_recognition_node` and `emotion_recognition_node` treat the left eye as
  available if it produced any track message in the last 2.0 s (`_STALE_SEC`)
  and switch to the right eye otherwise. `identity_manager_node`
  (`inmoov_cognition`, not part of this package) has its own
  `track_eye_fallback_sec` (default 2.0 s).
- `vision_head_tracker_node` **leads with the left eye**: it drives the head
  (`rothead` + `neck`) and publishes `eye_lr_L` / `eye_ud_L`; `EYE_SYNC` in
  `arduino_comm_node` (`inmoov_control`) mirrors the right eye. Only if the
  left camera has sent *no* track message for 5 s (`_FALLBACK_SEC`) does it
  take the right eye as the lead and publish `eye_lr_R` / `eye_ud_R` instead
  (both sets are never published at once, because `EYE_SYNC` would race).
- `face_capture_node` itself has a frame-level fallback: if one camera cannot
  be read, it republishes the other camera's last frame on the failed eye's
  topic.

The OAK-D Lite in the torso runs body/object detection independently of the
face pipeline. Its body signal (`/human_detected`, from
`human_detection_node`) is consumed by `identity_manager_node` in
`inmoov_cognition` as a **veto / gate** on the face pipeline: if the OAK-D is
active but has not seen a body for `no_human_timeout_sec` (default 20 s in the
node, 30 s in `inmoov.launch.py`), the identity manager treats face detections
as false positives and goes idle (see
`inmoov_cognition/identity_manager_node.py`, "OakD veto"). The veto logic lives
in `inmoov_cognition`; this package only produces the signal.

## Pipeline

```
 USB camera (left eye)   USB camera (right eye)              OAK-D Lite (torso)
          \                     /                                   |
        face_capture_node  (only node that opens the cameras)     oak_node
          |  /camera/eye_{left,right}/compressed                    |  /objects/detections
          |                                                         |  /objects/nearest
          |                                              +----------+-----------+
          v                                              |                      |
  face_detection_node_left ---- /face/detections/left    v                      v
  face_detection_node_right --- /face/detections/right  human_detection_node   scene_manager_node
  (right = on-demand standby,      |                     (+ ultrasonic L/R)     |
   wakes if left silent > 1.5 s)   v                     |                      v
  face_tracker_node_left  ------ /face/tracks/left       v               /scene/objects
  face_tracker_node_right ------ /face/tracks/right   /human_detected   (llm_node,
          |                                           /human_angle_deg   behavior_manager)
          |                                           (identity_manager, behavior_manager)
          +--> face_recognition_node  --> /face/identity   (<-> /memory/query)
          +--> emotion_recognition_node -> /face/emotion
          +--> face_gallery_node -------> photos on disk + /memory/query gallery_* ops
          +--> vision_head_tracker_node -> /joint_command, /face_command,
                                           /head_tracker/face_locked

 Gates (published by inmoov_cognition):
   /face_detection/enable  -> face_detection_node_*, face_tracker_node_*
   /head_tracker/enable    -> vision_head_tracker_node
   /robot_sleep (latched)  -> recognition, emotion, gallery, human_detection, scene_manager
   /social_context         -> vision_head_tracker_node, face_gallery_node
```

## Nodes

Executables are registered in [`setup.py`](setup.py) `console_scripts`. Default
values below are the **code defaults** (`_dp(...)` calls); where
[`inmoov_bringup/launch/inmoov.launch.py`](../inmoov_bringup/launch/inmoov.launch.py)
overrides them, the launch value (the one used on the robot) is given too. All topic names are fully qualified except the ones noted
as relative (they resolve to the same absolute names when no namespace is set,
which is how `inmoov_bringup` launches them).

### `face_capture_node`

Source: [`inmoov_vision/face_capture_node.py`](inmoov_vision/face_capture_node.py).

The only node that opens the USB cameras in the eyes (OpenCV `VideoCapture`
with the V4L2 backend, buffer size 1). At `fps` it grabs one frame from each
camera, JPEG-encodes it and publishes compressed images (no raw `Image`
topics). If a camera is not open or a read fails, it counts failures; after
`reopen_after_fails` consecutive failures it re-opens the device (at most once
per 2 s) and, meanwhile, publishes the *other* camera's last frame on the
failed eye's topic. `on_activate` fails (node goes degraded) only if **neither**
camera could be opened.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `cam_left` | string | `/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1.1:1.0-video-index0` | V4L2 device of the left eye (must be a **string**, not an integer index). |
| `cam_right` | string | `/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1.2:1.0-video-index0` | V4L2 device of the right eye. |
| `fps` | int | `15` | Capture/publish rate (timer period `1/fps`); also requested from the camera. |
| `width` | int | `640` | Capture width. |
| `height` | int | `480` | Capture height. |
| `jpeg_quality` | int | `85` | JPEG quality 1-100. |
| `flip_h_left` | bool | `False` | Flip the left frame horizontally. |
| `flip_h_right` | bool | `False` | Flip the right frame horizontally. |
| `reopen_after_fails` | int | `10` | Consecutive failed grabs before a re-open attempt. |

| Topic | Type | Dir | Notes |
|---|---|---|---|
| `camera/eye_left/compressed` | `sensor_msgs/CompressedImage` | pub | Relative name (`/camera/eye_left/compressed` with no namespace); QoS depth 5, `format='jpeg'`. |
| `camera/eye_right/compressed` | `sensor_msgs/CompressedImage` | pub | As above. |

### `face_detection_node` (instances `face_detection_node_left` / `_right`)

Source: [`inmoov_vision/face_detection_node.py`](inmoov_vision/face_detection_node.py).

Runs **insightface** (`FaceAnalysis`, model pack `buffalo_l`, ONNX Runtime
`CPUExecutionProvider`) on the frames of the eye selected by `camera_side`.
Frames are only buffered in the subscription callback; detection is triggered
by a timer at `detection_hz` and runs in a background thread (a new run is
skipped while the previous one is still busy). Detection is only performed
while `/face_detection/enable` is `True`. The ONNX Runtime thread pools are
capped by `intra_op_threads` / `inter_op_threads` (without a cap each of the
~5 buffalo_l models spawns a pool over all logical cores; the code comment
records ~350-370 % CPU per process without the limit).

**On-demand fallback** (`fallback_for`): when non-empty (the other eye's
detections topic), the node treats any message on that topic as the primary
camera's heartbeat (a message with no faces still counts) and skips detection
while the primary is alive; it starts detecting once the primary has been
silent for more than `primary_timeout_sec`, and logs when the primary
recovers. A grace period is restarted on every activation and on every
`enable=True`.

A watchdog (5 s timer) logs warnings for: no frames for >3 s, detection not
running for >3 s (not in standby), no face for >10 s. It also warns if the
frame *stamp* does not advance for 3 detections in a row (a "frozen" USB
camera that keeps delivering the same image).

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `camera_side` | string | `left` | `left` or `right`; selects input `/camera/eye_{side}/compressed` and output `/face/detections/{side}`. |
| `detection_hz` | double | `1.5` | Detection trigger rate. Launch: `5.0` (`detection_hz` arg). |
| `det_size` | int | `640` | insightface detector input size (square). Launch: `640`. |
| `det_thresh` | double | `0.4` | Detection score threshold. Launch: `0.5` (`det_thresh` arg). |
| `model_name` | string | `buffalo_l` | insightface model pack name. |
| `intra_op_threads` | int | `2` | ONNX Runtime intra-op threads. |
| `inter_op_threads` | int | `1` | ONNX Runtime inter-op threads. |
| `fallback_for` | string | `''` | Topic of the primary eye's detections; empty = always detect (primary role). Launch: right instance sets `/face/detections/left`. |
| `primary_timeout_sec` | double | `1.5` | Silence on `fallback_for` after which the standby node starts detecting. |

| Topic | Type | Dir | Notes |
|---|---|---|---|
| `/camera/eye_{side}/compressed` | `sensor_msgs/CompressedImage` | sub | QoS depth 5. |
| `/face_detection/enable` | `std_msgs/Bool` | sub | Latched (transient local, reliable, depth 1); published by `behavior_manager_node`. |
| `{fallback_for}` (e.g. `/face/detections/left`) | `std_msgs/String` | sub | Only when `fallback_for` is set. |
| `/face/detections/{side}` | `std_msgs/String` (JSON) | pub | `{"stamp": <frame stamp, s>, "faces": [{"bbox": [x1,y1,x2,y2] (int px), "det_score": float, "embedding": [512 floats, L2-normalized], "kps": [[x,y] x5]}]}`. Published even when `faces` is empty. |

### `face_tracker_node` (instances `face_tracker_node_left` / `_right`)

Source: [`inmoov_vision/face_tracker_node.py`](inmoov_vision/face_tracker_node.py).

A simple IOU tracker that gives persistent integer `track_id`s to faces across
detection messages. For every incoming detection message it (1) greedily
matches each existing track (in track order) to the unmatched detection with
the highest IOU, accepting it if IOU >= `iou_threshold`; (2) turns unmatched
detections into new tracks while fewer than `max_tracks` exist; (3) increments
`lost` on unmatched tracks and deletes a track after more than
`max_lost_frames` misses. Lost tracks are kept internally for re-matching but
**not published** (otherwise the head tracker would follow a stale bbox). One
`/face/tracks/{side}` message is published per detection message (empty
`tracks` list included, so it doubles as a heartbeat for downstream fallback
logic). When `/face_detection/enable` goes `False`, all tracks are dropped and
ids restart at 1.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `camera_side` | string | `left` | Selects input `/face/detections/{side}` and output `/face/tracks/{side}`. |
| `iou_threshold` | double | `0.35` | Minimum IOU for matching. Launch: `0.20`. |
| `max_lost_frames` | int | `30` | Detection frames a track may be unmatched before deletion. Launch: `20`. |
| `max_tracks` | int | `8` | Maximum simultaneous tracks. Launch: `4`. |

| Topic | Type | Dir | Notes |
|---|---|---|---|
| `/face/detections/{side}` | `std_msgs/String` (JSON) | sub | From `face_detection_node`. |
| `/face_detection/enable` | `std_msgs/Bool` | sub | Latched; `False` resets tracks. |
| `/face/tracks/{side}` | `std_msgs/String` (JSON) | pub | `{"stamp": s, "tracks": [{"track_id": int, "bbox": [...], "det_score": float, "embedding": [...], "kps": [...], "age": int}]}`. |

### `face_recognition_node`

Source: [`inmoov_vision/face_recognition_node.py`](inmoov_vision/face_recognition_node.py).

Identifies people from face tracks of both eyes, using the `/memory/query`
service (`inmoov_msgs/srv/MemoryQuery`, op `lookup_person`, served by
`inmoov_memory`).

- **left-primary** mode: the left track ids are canonical; embeddings from the
  right eye's largest face are added to the embedding buffer of the largest
  left track when their cosine similarity is >= `cross_eye_sim_thresh`
  (enrichment only for unlocked tracks).
- **right-only** mode: entered when no left track message was seen for 2.0 s
  (`_STALE_SEC`); the right eye becomes primary with its own `track_id`s.
  Every switch of side clears the cache.
- Per track it collects embeddings (only if `det_score >= min_det_score`) in a
  buffer (trimmed at `2 * embed_buf_size`); once it holds `embed_buf_size`
  embeddings and `recognition_cooldown_sec` has passed it averages them,
  re-normalizes, and calls `lookup_person` in a background thread. A known
  match locks the track (`lock_after_known`); otherwise the track is locked as
  unknown after `lock_after_attempts` attempts. Locked identities are
  re-published on every track message.
- Cache entries for tracks that disappeared are removed.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `recognition_cooldown_sec` | double | `2.0` | Min interval between service calls per track. Launch: `3.0`. |
| `lock_after_known` | bool | `True` | Lock a track once it is matched to a known person. |
| `lock_after_attempts` | int | `3` | Lookups after which an unknown track is locked as unknown. |
| `embed_buf_size` | int | `5` | Embeddings collected before the first lookup. |
| `min_det_score` | double | `0.65` | Minimum detection score for an embedding to be used. |
| `cross_eye_sim_thresh` | double | `0.45` | Min cosine similarity between right and left embeddings to merge them. |

| Topic / service | Type | Dir | Notes |
|---|---|---|---|
| `/face/tracks/left`, `/face/tracks/right` | `std_msgs/String` (JSON) | sub | From the trackers. |
| `/robot_sleep` | `std_msgs/Bool` | sub | Latched; while `True` all callbacks return early. |
| `/memory/query` | `inmoov_msgs/srv/MemoryQuery` | client | JSON `{"op": "lookup_person", "embedding": [...]}`; waits 2 s for the service, 5 s for the reply. |
| `/face/identity` | `std_msgs/String` (JSON) | pub | `{"track_id", "person_id", "name", "similarity", "is_known", "confidence", "best_candidate_id", "best_candidate_name", "locked"}`. |

### `emotion_recognition_node`

Source: [`inmoov_vision/emotion_recognition_node.py`](inmoov_vision/emotion_recognition_node.py).

Classifies the facial emotion of every tracked face with
**hsemotion-onnx** (`HSEmotionRecognizer`, model `enet_b0_8_best_vgaf`,
EfficientNet-B0 trained on AffectNet, 8 classes, CPU ONNX Runtime). Works from
the left eye; switches to the right eye only if the left eye has sent no
tracks message for 2.0 s (`_STALE_SEC`) - i.e. the camera pipeline is dead -
not merely when there are no faces in the left frame (in that case it does
nothing). It keeps the last ~20 frames of each eye (still JPEG-compressed)
and crops each track's bbox from the frame whose stamp matches the tracks
message's `stamp` (the frame the bbox was computed on; the tick is skipped if
that frame is not buffered),
skips faces smaller than `min_face_size` px, and publishes one message per
analyzed face. Class names are lower-cased (`Anger`→`angry`,
`Happiness`→`happy`, `Sadness`→`sad`, ...). Runs one analysis at a time.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `analysis_hz` | double | `2.0` | Analysis trigger rate (launch arg `analysis_hz`). |
| `min_face_size` | int | `48` | Minimum bbox width/height in px. |

| Topic | Type | Dir | Notes |
|---|---|---|---|
| `/camera/eye_left/compressed`, `/camera/eye_right/compressed` | `sensor_msgs/CompressedImage` | sub | Latest frame per eye. |
| `/face/tracks/left`, `/face/tracks/right` | `std_msgs/String` (JSON) | sub | Bboxes + freshness. |
| `/robot_sleep` | `std_msgs/Bool` | sub | Latched. |
| `/face/emotion` | `std_msgs/String` (JSON) | pub | `{"track_id", "emotion", "confidence", "source": "left"|"right", "all": {emotion: score}}`. |

### `face_gallery_node`

Source: [`inmoov_vision/face_gallery_node.py`](inmoov_vision/face_gallery_node.py).

Automatically saves face crops (20 % padding) from both eyes into a photo
gallery used for gallery-based recognition.

```
{gallery_dir}/
  persons/{person_id}_{name}/   left_YYYYmmdd_HHMMSS_xxxxxx.jpg, right_...jpg
  _pending/                     photos taken before an introduction completes
```

- State comes from `/social_context`. In `introducing`: one capture every
  `enroll_interval_sec` into `_pending/` (up to `max_pending_photos`). In
  `interacting` with a known `person_id`: one capture every
  `interact_interval_sec` into `persons/{id}_{name}/`, each photo registered
  via `/memory/query` op `gallery_add`.
- On the `introducing` → `interacting` transition (with a `person_id`) the
  pending photos are moved into the person's directory (up to the photo limit)
  and registered (`gallery_add`). Entering `introducing` clears `_pending/`.
- **Photo limit** (`max_photos_per_person`): when the directory is full only
  one left + one right photo per session are added (JPEG quality 100) and the
  oldest left / right photo is deleted from disk and from the DB
  (`gallery_remove`).
- **Embedding recompute**: at the end of a session (`interacting` → anything
  else) or on going to sleep, `gallery_rebuild_embedding` is called for that
  person.
- **Quality filters**: `det_score >= min_det_score`, bbox width >=
  `min_face_px`, and cosine similarity to the last 20 saved embeddings must be
  <= `max_diversity_sim` (no near-duplicates).
- While in a dialogue it follows only the interlocutor's `track_id`, obtained
  from a locked, known `/face/identity` matching the current `person_id`.
- A capture requires a **left** track (bbox + embedding); the right crop is
  added only if a right track above `min_det_score` is also available.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `gallery_dir` | string | `/home/artur/inmoov_faces` | Root photo directory (created if missing). **Override on other machines.** Launch: same value hard-coded. |
| `enroll_interval_sec` | double | `1.0` | Capture interval while introducing. |
| `interact_interval_sec` | double | `15.0` | Capture interval while interacting. |
| `min_det_score` | double | `0.80` | Minimum detection score. Launch: `0.75`. |
| `min_face_px` | int | `60` | Minimum bbox width. |
| `max_diversity_sim` | double | `0.90` | Reject a photo if similarity to a recent one is above this. |
| `max_pending_photos` | int | `40` | Cap on enrolment photos in `_pending/`. |
| `max_photos_per_person` | int | `30` | Photo limit per person (left + right combined). |

| Topic / service | Type | Dir | Notes |
|---|---|---|---|
| `/camera/eye_left/compressed`, `/camera/eye_right/compressed` | `sensor_msgs/CompressedImage` | sub | Source of crops. |
| `/face/tracks/left`, `/face/tracks/right` | `std_msgs/String` (JSON) | sub | Bbox, score, embedding. |
| `/social_context` | `std_msgs/String` (JSON) | sub | `state` (`idle`/`introducing`/`interacting`), `person_id`, `name`; from `identity_manager_node`. |
| `/face/identity` | `std_msgs/String` (JSON) | sub | Interlocutor track binding. |
| `/robot_sleep` | `std_msgs/Bool` | sub | Latched; going to sleep triggers an embedding rebuild. |
| `/memory/query` | `inmoov_msgs/srv/MemoryQuery` | client | Ops `gallery_add`, `gallery_remove`, `gallery_rebuild_embedding`. |

### `vision_head_tracker_node`

Source: [`inmoov_vision/vision_head_tracker_node.py`](inmoov_vision/vision_head_tracker_node.py).

Controls the robot's gaze: a P controller turns head and eyes toward the
tracked face. Runs a timer at `track_hz` while `/head_tracker/enable` is
`True`.

- **Leader / follower**: the left eye leads when it has a fresh bbox (< 2.0 s,
  `_STALE_SEC`). The right eye leads only if the left camera has sent no
  track message at all for >= 5.0 s (`_FALLBACK_SEC`) *and* the right bbox is
  fresh. Only one eye's joint names are published (`eye_lr_L`/`eye_ud_L` or
  `eye_lr_R`/`eye_ud_R`); `EYE_SYNC` in `arduino_comm_node` mirrors the other.
  Bboxes are EMA-smoothed (`bbox_ema_alpha`).
- **Target selection**: without a bound interlocutor, the largest face wins.
  In an `interacting` dialogue (`/social_context`) it follows only the
  `track_id` bound by a locked, known `/face/identity` for that `person_id`;
  if that face is not in frame it does **not** move.
- **Head**: exactly **one step per new detection** (`bbox_seq`), not per tick
  (fix for a live bug where one stale offset was applied several times between
  detections). Step = `head_pan_dir * norm_x * max_step_deg * gain_head`
  (clamped to +-`max_step_deg`), only outside the `head_dead_zone_px` dead
  zone. Joint limits (hard-coded): `rothead` 30-130 deg, `neck` 1-100 deg.
- **Runaway guard**: if 8 consecutive detections do not reduce the offset
  magnitude by at least 0.04 relative to a baseline set at the start of the
  series, head accumulation stops (with a `RUNAWAY` warning) until progress is
  seen or the track binding changes. The baseline (not step-to-step) is used
  because bbox noise between detections is easily +-0.05-0.15.
- **Eyes**: absolute target within +-half of the eye range (`eye_lr` 80-100,
  `eye_ud` 80-110 deg, hard-coded), scaled by `gain_eye`, outside `dead_zone_px`.
- **Rest**: when the tracker is disabled it returns head/eyes to `rest_*`
  once (and publishes `face_locked=False`); when no track has been seen for
  `return_timeout_sec` while enabled, it returns to rest and then keeps
  re-publishing the rest pose on every tick.
- **Diagnostics**: a `SUSPICIOUS track_id JUMP` warning when the target
  `track_id` changes and the new bbox is near the frame edge (|offset| > 0.5);
  a tracking log line every 3 s.
- Rothead/neck from any other publisher on `/joint_command` (e.g. a one-off
  head aim by `SoundScanBehaviour`) are picked up so the P controller starts
  from the real position.
- Joint values are published as radians relative to 90 deg
  (`(deg - 90) * pi / 180`).

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `image_width` | int | `640` | Frame width used to normalize bbox offsets. |
| `image_height` | int | `480` | Frame height. |
| `gain_head` | double | `0.5` | Head proportional gain. Launch: `0.3` (`gain_head` arg). |
| `gain_eye` | double | `0.2` | Eye gain. Launch: `0.6` (`gain_eye` arg). |
| `dead_zone_px` | int | `20` | Eye dead zone. Launch: `20`. |
| `head_dead_zone_px` | int | `80` | Head dead zone. Launch: `80`. |
| `return_timeout_sec` | double | `12.0` | Seconds without a track before returning to rest. Launch: `10.0`. |
| `track_hz` | double | `15.0` | Control-loop rate. Launch: `10.0`. |
| `min_det_score` | double | `0.50` | Tracks below this score are ignored. |
| `head_pan_dir`, `head_tilt_dir` | int | `1`, `-1` | Sign conventions for head axes. |
| `eye_pan_dir`, `eye_tilt_dir` | int | `-1`, `-1` | Sign conventions for eye axes. |
| `max_step_deg` | double | `2.0` | Max head step per detection. |
| `rest_rothead` | double | `90.0` | Rest angle of `rothead` (launch arg `rest_rothead`). |
| `rest_neck` | double | `40.0` | Rest angle of `neck` (launch arg `rest_neck`). |
| `rest_eye_lr` | double | `90.0` | Rest angle of eye left/right. |
| `rest_eye_ud` | double | `100.0` | Rest angle of eye up/down. |
| `bbox_ema_alpha` | double | `0.4` | EMA weight of the new bbox. |

| Topic | Type | Dir | Notes |
|---|---|---|---|
| `/face/tracks/left`, `/face/tracks/right` | `std_msgs/String` (JSON) | sub | Ignored while disabled. |
| `/head_tracker/enable` | `std_msgs/Bool` | sub | Latched; `False` returns to rest and publishes `face_locked=False`. |
| `/face/identity` | `std_msgs/String` (JSON) | sub | Binds the interlocutor's `track_id`. |
| `/social_context` | `std_msgs/String` (JSON) | sub | `state == 'interacting'` + `person_id` selects the target person. |
| `/joint_command` | `sensor_msgs/JointState` | sub + pub | Publishes `rothead`, `neck`; also subscribes to sync from external head commands. |
| `/face_command` | `sensor_msgs/JointState` | pub | `eye_lr_L`+`eye_ud_L` or `eye_lr_R`+`eye_ud_R`. |
| `/head_tracker/face_locked` | `std_msgs/Bool` | pub | True iff a fresh bbox exists *right now*; published every tick, not latched. Single source of truth for "face caught" used by the face-search retry behaviours in `behavior_manager_node`. |

### `oak_node`

Source: [`inmoov_vision/oak_node.py`](inmoov_vision/oak_node.py).

OAK-D Lite driver: YOLO object detection with stereo depth, using the
**depthai v3** API (follows the official Luxonis `spatial_detection` example).
A background thread builds a pipeline: RGB camera (`CAM_A`), mono cameras
`CAM_B` / `CAM_C` at `fps` feeding `StereoDepth` (extended disparity, 640x400),
and `SpatialDetectionNetwork` with the model given by `model_name` (resolved
through Luxonis HubAI via `dai.NNModelDescription`; default `yolov6-nano`).
Depth is limited to 100-8000 mm; the NN input queue is non-blocking. Labels
come from the model (`labelName`), falling back to a built-in COCO-80 list.
Messages are published only for frames that contain at least one detection.
If `depthai` cannot be imported, `on_activate` returns `FAILURE` (node
degraded). The OAK thread is a supervisor: if pipeline creation fails, the
pipeline stops or the device disconnects, it rebuilds the pipeline after a
growing back-off (5 s → 60 s; reset after a 60 s stable run) while the node is
active.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `model_name` | string | `yolov6-nano` | HubAI model name (launch arg `oak_model`). |
| `conf_threshold` | double | `0.5` | NN confidence threshold (launch arg `oak_conf_threshold`). |
| `fps` | int | `15` | Sensor FPS of all three cameras. |

| Topic | Type | Dir | Notes |
|---|---|---|---|
| `objects/detections` | `std_msgs/String` (JSON) | pub | Relative name. `{"objects": [{"label", "confidence", "x_mm", "y_mm", "z_mm", "bbox": [xmin, ymin, xmax, ymax] (normalized 0-1)}]}`. |
| `objects/nearest` | `std_msgs/String` (JSON) | pub | Relative name. The single object with the smallest `abs(z_mm)`, same fields. |

### `human_detection_node`

Source: [`inmoov_vision/human_detection_node.py`](inmoov_vision/human_detection_node.py).

Body presence: derives a `person` boolean from the OAK-D detections. A
detection counts if `label == 'person'`, `confidence >= min_confidence` and
`0 < z_mm <= max_distance_m * 1000`. `/human_detected` is published at
`publish_rate_hz` and stays `True` for `lost_timeout_sec` after the last
matching detection. When a person is detected `/human_angle_deg` is published
as `atan2(x_mm, z_mm)` (degrees) of the nearest person - positive means to the
**right** (DepthAI convention: +X to the right of the camera). This sign is
**not validated physically**; check by hand on first use. `/robot_sleep`
resets its state and publishes a single `False`.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `max_distance_m` | double | `4.0` | Maximum range for a person detection. |
| `min_confidence` | double | `0.45` | Minimum YOLO confidence. |
| `lost_timeout_sec` | double | `4.0` | Time without detection before `False`. |
| `publish_rate_hz` | double | `5.0` | Publish rate. |
| `use_ultrasonic` | bool | `True` | Subscribe to the ultrasonic sensors and use them for the distance estimate. |
| `ultrasonic_stale_s` | double | `1.0` | Max age of an ultrasonic reading. |

| Topic | Type | Dir | Notes |
|---|---|---|---|
| `/objects/detections` | `std_msgs/String` (JSON) | sub | From `oak_node`. |
| `/ultrasonic_left_distance`, `/ultrasonic_right_distance` | `std_msgs/Int16` (cm) | sub | From `arduino_left_node` / `arduino_right_node`; values <= 0 ignored. Only if `use_ultrasonic`. |
| `/robot_sleep` | `std_msgs/Bool` | sub | Latched. |
| `/human_detected` | `std_msgs/Bool` | pub | Consumed by `identity_manager_node` and `behavior_manager_node`. |
| `/human_angle_deg` | `std_msgs/Float32` | pub | Only while a person is detected; consumed by `behavior_manager_node` (aiming the head before face detection sees the person). |
| `/human_distance_m` | `std_msgs/Float32` | pub | Only while a person is detected: ultrasonic distance if fresh (nearer of the two sensors), else the YOLO z. No consumer yet; presence itself uses YOLO only. |

### `scene_manager_node`

Source: [`inmoov_vision/scene_manager_node.py`](inmoov_vision/scene_manager_node.py).

Turns the raw OAK-D detection stream into a stable summary of the scene
(people and objects around the robot, distance, side) for `llm_node` (system
prompt) and `behavior_manager_node` (Blackboard). Detections below
`min_confidence` or beyond `max_distance_m` are dropped. Per label it keeps a
sliding window (`smoothing_window_sec`) of per-frame counts and reports the
maximum over fresh frames (suppresses count jitter); if the window is empty
but the label has not yet expired, the last known count is kept. A label
disappears after `object_ttl_sec` without confirmation (a short TTL made the
summary flicker because YOLO occasionally drops frames). Distance is the
nearest instance's `z_mm`; direction is `left` if the bbox centre x < 0.35,
`right` if > 0.65, else `center`. Objects are sorted by distance and truncated
to `top_k_objects`.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `min_confidence` | double | `0.5` | Minimum detection confidence. |
| `max_distance_m` | double | `4.0` | Maximum range. |
| `object_ttl_sec` | double | `8.0` | Label lifetime without confirmation. |
| `smoothing_window_sec` | double | `1.0` | Count smoothing window. |
| `publish_rate_hz` | double | `1.0` | Summary publish rate. |
| `top_k_objects` | int | `8` | Max labels in the summary. |
| `location_name` | string | `''` | Static name of the robot's current location; re-read on every publish, so it can be changed live: `ros2 param set /scene_manager_node location_name "kitchen"` (launch arg `scene_location`). |

| Topic | Type | Dir | Notes |
|---|---|---|---|
| `/objects/detections` | `std_msgs/String` (JSON) | sub | From `oak_node`. |
| `/robot_sleep` | `std_msgs/Bool` | sub | Latched; clears the label table on sleep. |
| `/scene/objects` | `std_msgs/String` (JSON) | pub | `{"location": str, "person_count": int, "objects": [{"label", "count", "distance_m", "direction"}], "updated_at": epoch s}`. |

## Launch

The package has no launch file of its own. All nodes are started by
[`inmoov_bringup/launch/inmoov.launch.py`](../inmoov_bringup/launch/inmoov.launch.py)
as `LifecycleNode` actions (`respawn=True`) and brought up tier by tier by
`lifecycle_manager`; `vision:=false` skips all of them:

```bash
ros2 launch inmoov_bringup inmoov.launch.py
ros2 launch inmoov_bringup inmoov.launch.py cam_left:=/dev/v4l/by-path/... cam_right:=/dev/v4l/by-path/...
```

**How left/right are parameterized.** The same two executables
(`face_detection_node`, `face_tracker_node`) are launched twice under
different node names (`..._left`, `..._right`), and the `camera_side`
parameter selects the input/output topic pair. The right detection instance
additionally gets `fallback_for: /face/detections/left` and
`primary_timeout_sec: 1.5`, which makes it an on-demand standby.

Vision-related launch arguments of `inmoov.launch.py`:

| Launch argument | Default | Goes to |
|---|---|---|
| `cam_left` | `/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1.1:1.0-video-index0` | `face_capture_node.cam_left` |
| `cam_right` | `/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1.2:1.0-video-index0` | `face_capture_node.cam_right` |
| `fps` | `15` | `face_capture_node.fps` |
| `detection_hz` | `5.0` | both `face_detection_node.detection_hz` |
| `det_thresh` | `0.5` | both `face_detection_node.det_thresh` |
| `analysis_hz` | `2.0` | `emotion_recognition_node.analysis_hz` |
| `gain_head` | `0.3` | `vision_head_tracker_node.gain_head` |
| `gain_eye` | `0.6` | `vision_head_tracker_node.gain_eye` |
| `rest_rothead` | `90.0` | `vision_head_tracker_node.rest_rothead` |
| `rest_neck` | `40.0` | `vision_head_tracker_node.rest_neck` |
| `oak_model` | `yolov6-nano` | `oak_node.model_name` |
| `oak_conf_threshold` | `0.5` | `oak_node.conf_threshold` |
| `scene_location` | `''` | `scene_manager_node.location_name` |

Other values (tracker `iou_threshold` 0.20, `max_lost_frames` 20,
`max_tracks` 4; gallery `gallery_dir`, `min_det_score` 0.75; head-tracker
`track_hz` 10, `return_timeout_sec` 10, ...) are hard-coded in the launch file.

To run a single node by hand: `ros2 run inmoov_vision <node>` and then
`ros2 lifecycle set /<node> configure` / `activate`. Nothing publishes `/face_detection/enable` or
`/head_tracker/enable` from this package: they come from
`behavior_manager_node` in `inmoov_cognition`, and the detection and head
tracker nodes stay idle until they are `True`.

## Requirements / Setup

- ROS2 Jazzy, `ament_python` package. Build with
  `colcon build --packages-select inmoov_vision` (no `--symlink-install` is
  used in this workspace). Depends on `rclpy`, `std_msgs`, `sensor_msgs` and
  `inmoov_msgs` (for `MemoryQuery.srv`), per [`package.xml`](package.xml).
- **Python dependencies**: `numpy`, `opencv`, `onnxruntime` and `insightface`
  are declared in `package.xml` as rosdep keys
  (`rosdep install --from-paths src --ignore-src -y`); `depthai` (v3) and
  `hsemotion-onnx` have no rosdep keys. Everything is listed in
  [`requirements.txt`](requirements.txt)
  (`pip install -r src/inmoov_vision/requirements.txt`). Versions in use on
  the author's machine in parentheses:
  - `numpy` (2.4.4), `opencv-python` / `cv2` (4.13) - all camera/image nodes;
  - `onnxruntime` (1.24.3) and `insightface` (0.7.3) - `face_detection_node`;
  - `hsemotion-onnx` (0.3.1) - `emotion_recognition_node`;
  - `depthai` (3.6.1, **v3 API**) - `oak_node` (imported lazily; without it the
    node fails activation).
- Model weights (downloaded automatically on first use, so first start needs
  network access):
  - insightface `buffalo_l` → `~/.insightface/models/buffalo_l/`;
  - hsemotion `enet_b0_8_best_vgaf.onnx` → `~/.hsemotion/` (fetched from the
    HSE-asavchenko/face-emotion-recognition GitHub repo);
  - OAK-D model (`yolov6-nano` by default) → fetched from Luxonis HubAI by
    depthai into its model cache (`DEPTHAI_ZOO_CACHE_PATH` if set, otherwise
    depthai's default cache directory).
- Hardware: two USB (UVC) cameras in the eyes, an OAK-D Lite on USB, and the
  Arduino boards providing `/ultrasonic_{left,right}_distance` (optional; see
  `use_ultrasonic`). The user typically needs access to the video devices
  (`video` group).
- **Camera paths**: the defaults are `/dev/v4l/by-path/...` names that depend
  on the physical USB port / PCI address of the author's computer
  (`pci-0000:c6:00.3-usb-0:1.1:1.0-video-index0` left, `...-usb-0:1.2:...`
  right). `by-path` is used because the two cameras have identical serial
  numbers, so `by-id` cannot tell them apart, and `/dev/videoN` numbering is not
  stable. On another machine find them with `ls -l /dev/v4l/by-path/` and pass
  `cam_left:=... cam_right:=...`. Use the `-video-index0` (capture) node.
- **Gallery / database**: `face_gallery_node` writes photos under
  `gallery_dir` (default `/home/artur/inmoov_faces`, hard-coded also in the
  launch file - change both for another user). Embeddings and person records
  are kept by `inmoov_memory`'s `memory_node` in its SQLite database
  (`/home/artur/inmoov_memory.db` by default); this package never opens the
  database, it only calls the `/memory/query` service, so `memory_node` must be
  running for recognition and gallery registration.
- `face_recognition_node`, `face_gallery_node`, the head tracker and the
  emotion node depend on the `inmoov_cognition` nodes for `/social_context`,
  `/robot_sleep` and the enable gates.

## Known issues to verify

- **Right-eye data may never be produced in normal operation.** With the
  shipped launch configuration `face_detection_node_right` is a standby
  (`fallback_for` set) and only detects when the left eye is silent, so
  `/face/detections/right` and `/face/tracks/right` are normally silent. The
  right-eye enrichment in `face_recognition_node` and the `right_*.jpg`
  photos in `face_gallery_node` (which need a right track) are therefore only
  active during a left-eye failure. Verify whether this is intended.
- **`face_gallery_node` needs the left eye.** A capture is skipped if there is
  no left bbox/embedding, so it does not work during a left-eye failover even
  if the right eye is detecting.
- `/human_angle_deg` sign convention is not validated physically (noted in the
  code).
- Live-bug history in the code comments (2026-08-24 ... 2026-09-01): the head
  drifting away from the real face or running to the servo limit (runaway),
  hijack by another face at identity confirmation, and "frozen" USB frames.
  The mitigations (one step per detection, baseline-based runaway guard,
  frame-stamp watchdog, `SUSPICIOUS track_id JUMP` warning) are in place but
  were validated only in live tests; a preference for continuity over face
  area when choosing the target was tried and reverted.
- `face_capture_node`'s cross-camera fallback republishes the other eye's
  image under the failed eye's topic, so downstream nodes cannot tell that
  left and right are identical.
- `oak_node` only publishes when a frame has detections (silence means "nothing
  seen"). Its pipeline restart supervisor was tested with a failing fake
  `depthai` only — verify with a real disconnect of the OAK-D.
- Several defaults in the code differ from the values `inmoov_bringup`
  passes (see the tables above); the launch values are the ones used on the
  robot.
- No automated tests beyond the stock ament copyright/flake8/pep257 checks in
  `test/`.

## License

GNU General Public License v3.0 (GPL-3.0-only); see the `LICENSE` file at the
repository root.

Author: Artur Fedjukevits. Assisted by: Claude Code (Anthropic).
