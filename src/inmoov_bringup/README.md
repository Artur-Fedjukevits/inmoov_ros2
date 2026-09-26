# inmoov_bringup

Top-level bringup package for the InMoov robot. It does not implement any
robot behavior itself — it starts and supervises every other node in the
system through a single custom orchestrator, `lifecycle_manager`, built on
ROS2 managed lifecycle nodes (`rclpy_lifecycle` / `lifecycle_msgs`).

Nodes are grouped into ordered **tiers (0–6)**, configured in
[`config/lifecycle.yaml`](config/lifecycle.yaml): tiers activate sequentially
(tier *N+1* only starts once tier *N* is up), while nodes inside a tier
activate in parallel. Each node is marked `critical` or optional; a critical
node's activation failure aborts the whole tier (and the system), while an
optional node's failure just marks it `degraded` and lets the system continue.
A background **watchdog** polls `get_state` on every node once the system is
fully active, detects crashed/respawned processes, cascades deactivation up
through dependent tiers when a critical node dies, and cascades recovery back
down once it comes back (via `ros2 launch respawn=True`). The whole system's
health is exposed as a single JSON message on `/lifecycle/status`, and can be
driven externally (sleep/wake, tier restarts, full shutdown) via
`/lifecycle/command`.

## Nodes

### `lifecycle_manager`

Tier-based lifecycle orchestrator. Source:
[`inmoov_bringup/lifecycle_manager.py`](inmoov_bringup/lifecycle_manager.py).

This is a plain (non-lifecycle) `rclpy` node that manages the *other* nodes,
which are themselves ROS2 managed-lifecycle nodes (`LifecycleNode` in the
launch file). It talks to each managed node purely through the standard
`<node>/change_state` and `<node>/get_state` services — it does not depend on
any custom API from the nodes it manages.

**Behavior**

- **Serial execution**: every operation — the commands, SLEEP/WAKE from
  `/robot_sleep`, the watchdog's cascade and recovery — goes onto one queue
  and is executed by a single worker thread, one at a time, so operations
  never interleave. Each operation has a generation number: per-node
  activation threads stop retrying once their operation is superseded, and
  `SHUTDOWN` pre-empts the running operation. Queued SLEEP/WAKE apply the
  *latest* requested sleep state, so a burst of toggles settles correctly.
- **Startup**: after `autostart_delay_sec`, activates tiers 0→6 in order.
  Within a tier, all nodes are configured+activated in parallel threads, each
  with up to `retry_count` attempts (linearly increasing backoff:
  `retry_interval_sec * attempt`). If a `critical` node in a tier fails all
  retries, the tier — and the whole activation sequence — aborts and system
  state becomes `fault`. A non-critical failure just marks that node
  `degraded` (reason `activation`) and the tier continues. The whole tier
  shares one `tier_advance_timeout_sec` deadline; a node still activating
  after it is marked `degraded` (reason `timeout`) and the tier moves on.
  When that node's thread finishes, a `RECONCILE` operation brings it to the
  state wanted *now* (clears the mark, or deactivates it if the robot went to
  sleep / was deactivated meanwhile).
- **Watchdog**: starts `watchdog_startup_delay_sec` after the system reaches
  `active`, then polls `get_state` on every node every
  `watchdog_interval_sec` — detection only, actions are queued. Detects two death scenarios: the service call gets
  no response at all (process gone, no respawn yet), or the node answers but
  is back in `UNCONFIGURED` (process died and `ros2 launch respawn=True`
  already restarted it, but it hasn't been configured/activated again). A
  dead critical node triggers `_cascade_deactivate`, which `DEACTIVATE`s every
  node in every tier above it (they're assumed to depend on it) and marks
  them `degraded` with reason `cascade_<tier_idx>`. Recovery
  (`_recover_node`) re-runs the normal activate-with-retry logic against the
  respawned process; it is queued after the cascade, so on success for a
  critical node `_cascade_recover` re-activates every tier that was
  cascade-degraded because of this node. `max_respawn_count` caps how many times the watchdog
  will attempt to recover a single node before giving up permanently.
- **SLEEP / WAKE**: a fixed set of vision + VAD nodes
  (`_SLEEP_DEACTIVATE` in the source — `face_capture_node`,
  `face_detection_node_{left,right}`, `face_tracker_node_{left,right}`,
  `face_recognition_node`, `face_gallery_node`, `emotion_recognition_node`,
  `vision_head_tracker_node`, `human_detection_node`, `oak_node`) get
  `DEACTIVATE`d on `SLEEP` and re-activated on `WAKE`. `voice_detector_node`
  stays active so the command right after the wake word isn't lost while
  WAKE is still running. Triggered either by the `/lifecycle/command` topic or automatically
  by the latched `/robot_sleep` topic (published elsewhere in the system,
  e.g. by `inmoov_cognition`).
- **Shutdown**: stops the watchdog, then takes every node through
  `DEACTIVATE` → `CLEANUP` → `SHUTDOWN`, tier by tier from the highest tier
  down to tier 0.
- **State-aware transitions**: before a `DEACTIVATE` / `CLEANUP` / `SHUTDOWN`
  the manager reads the node's current state and sends only valid
  transitions — an invalid one (e.g. `DEACTIVATE` of an already `INACTIVE`
  node) makes rclpy raise inside the node's `change_state` service and kills
  the process. Watchdog and cascade recovery respect SLEEP: a SLEEP-set node
  recovered while the robot is asleep is configured but left `INACTIVE`.
- **YAML overrides**: `retry_*`, `transition_timeout_sec`,
  `tier_advance_timeout_sec`, `watchdog_*` and `max_respawn_count` in the
  `config_file` YAML override the node parameters.

**Parameters**

| Name | Type | Default | Meaning |
|---|---|---|---|
| `retry_count` | int | `3` | Activation attempts per node before giving up. |
| `retry_interval_sec` | double | `10.0` | Base backoff between retries; actual wait is `retry_interval_sec * attempt`. |
| `transition_timeout_sec` | double | `30.0` | Timeout waiting for a `change_state`/`get_state` service call. |
| `tier_advance_timeout_sec` | double | `90.0` | One deadline for the whole tier; nodes still activating after it are marked `degraded(timeout)` and reconciled when they finish. |
| `config_file` | string | `''` | Path to a YAML file (see `config/lifecycle.yaml`) defining `tiers` and overriding the parameters above. If empty, the manager waits for `configure_tiers()` to be called programmatically instead. |
| `autostart_delay_sec` | double | `5.0` | Delay after startup before automatically activating tier 0. If tiers aren't configured yet, or this is `<= 0`, autostart is skipped and the manager waits in `idle` for an `ACTIVATE` command. |
| `watchdog_interval_sec` | double | `5.0` | Polling interval for the watchdog loop. Set `<= 0` to disable the watchdog entirely. |
| `watchdog_startup_delay_sec` | double | `15.0` | Grace period after full system activation before the watchdog starts polling (avoids false positives while nodes are still settling). |
| `max_respawn_count` | int | `5` | Max watchdog-driven recovery attempts per node before it's left permanently `degraded`. |
| `disabled_nodes` | string | `''` | Comma-separated node names dropped from the tiers (not managed at all). The launch file fills it with the vision nodes when `vision:=false` and with `telegram_bridge_node` when `telegram:=false`, so activation doesn't wait out retries for nodes that were never started. |

**Topics**

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/lifecycle/status` | `std_msgs/String` (JSON) | publish, every 2s | `{system, sleep_mode, degraded_nodes[], recovering_nodes[], pending_ops, tiers: [{id, nodes: {name: {critical, degraded, degraded_reason, respawn_count, recovering, attempts}}}]}`. `system` is one of `idle` (no autostart, waiting for `ACTIVATE`), `starting`, `active`, `fault`, `degraded`, `sleep`, `waking`, `deactivated`, `shutdown`. |
| `/lifecycle/command` | `std_msgs/String` | subscribe | Commands: `ACTIVATE`, `DEACTIVATE`, `SHUTDOWN`, `SLEEP`, `WAKE`, `RESTART_TIER <N>`. `DEACTIVATE` deactivates tiers N..1 top-down (Foundation stays active) and pauses the watchdog and SLEEP/WAKE transitions (the sleep flag is still remembered). `ACTIVATE` after `DEACTIVATE` re-activates tiers 1..N with a fresh retry budget (SLEEP-set nodes stay inactive if the robot is asleep); from `idle` or `fault` it runs the full tier 0..N activation; otherwise it is a no-op. |
| `/robot_sleep` | `std_msgs/Bool` (latched, `TRANSIENT_LOCAL`/`RELIABLE`, depth 1) | subscribe | Drives the same SLEEP/WAKE logic as the command topic; published elsewhere in the system. |

**Services used (per managed node)**

- `/<node_name>/change_state` (`lifecycle_msgs/srv/ChangeState`)
- `/<node_name>/get_state` (`lifecycle_msgs/srv/GetState`)

**Known limitations**

- A long operation (e.g. a recovery sitting in its retry back-off) delays
  the operations queued behind it, including SLEEP/WAKE; only `SHUTDOWN`
  pre-empts.
- The watchdog sees process death only (`get_state` fails or the node is
  back in `UNCONFIGURED`), not a node that is ACTIVE but no longer doing its
  job.

**Tests**: `test/test_lifecycle_manager.py` runs the manager against fake
in-process lifecycle nodes (tier deadline + reconcile, SLEEP/WAKE bursts,
crash → cascade → recovery, SHUTDOWN pre-emption).

## Launch

```bash
ros2 launch inmoov_bringup inmoov.launch.py
```

The launch file ([`launch/inmoov.launch.py`](launch/inmoov.launch.py)) starts
`lifecycle_manager` as a plain `Node` (pointed at `config/lifecycle.yaml` via
the `config_file` parameter, with `disabled_nodes` computed from the `vision`/
`telegram` arguments) plus every managed node as a `LifecycleNode`,
all starting `Unconfigured` — the manager alone drives every transition.
Every managed node is launched with `respawn=True`, `respawn_delay=2.0` so
`ros2 launch` restarts a crashed process and the watchdog can then reactivate
it.

Nodes launched, by tier: `memory_node` (0); `audio_source_node`,
`arduino_right`, `arduino_left`, `face_capture_node`, `oak_node`,
`sound_localization_node` (1); `wakeword_node`, `voice_detector_node`,
`tts_node`, `joint_state_publisher`, `face_expressions`,
`face_detection_node_{left,right}` (2); `face_tracker_node_{left,right}`,
`voice_emotion_node`, `parakeet_stt_node` (3); `face_recognition_node`,
`face_gallery_node`, `emotion_recognition_node`, `vision_head_tracker_node`,
`human_detection_node`, `scene_manager_node`, `llm_node`,
`openhab_bridge_node` (4); `identity_manager_node`, `behavior_manager_node`
(5); `telegram_bridge_node` (6, only if `telegram:=true`). All camera/face/OAK
nodes (`face_capture_node`, `oak_node`, `face_detection_node_*`,
`face_tracker_node_*`, `face_recognition_node`, `face_gallery_node`,
`emotion_recognition_node`, `vision_head_tracker_node`, `human_detection_node`,
`scene_manager_node`) are started only if `vision:=true`.

### Robot configuration (`config/robot.yaml`)

Everything specific to this robot, machine and network lives in
[`config/robot.yaml`](config/robot.yaml): server URLs (LLM, vision LLM, TTS,
openHAB), device paths (Arduino ports, eye cameras) and data paths (memory
DBs, Chroma, face gallery, wake-word model). The launch file uses it for the
defaults of the matching launch arguments below; `~` is expanded. For another
robot, point `INMOOV_ROBOT_CONFIG` at a copy instead of editing the file:

```bash
INMOOV_ROBOT_CONFIG=~/my_robot.yaml ros2 launch inmoov_bringup inmoov.launch.py
```

Secrets are never read from it — they stay in the environment
(`VLLM_BEARER_TOKEN`, `TAVILY_API_KEY`, `TELEGRAM_BOT_TOKEN`,
`TELEGRAM_ALLOWED_CHAT_ID`).

### Launch arguments

Servers / LLM:
- `llm_url` (default `http://192.168.10.118:18020/v1/chat/completions`) — OpenAI-compatible chat.completions endpoint (vLLM), shared by `llm_node` and `identity_manager_node` (name extraction).
- `llm_fallback_url` (default `''`) — optional OpenAI-compatible backup endpoint; empty = no fallback.
- `llm_bearer_token` (default from env `VLLM_BEARER_TOKEN`)
- `llm_model` (default `qwen3.8-27b`), `llm_temperature` (`0.1`), `llm_max_tokens` (`512`)
- `tts_server_url` (default `http://192.168.10.118:8000`), `tts_fallback_url` (default `''` — no fallback)
- `openhab_url` (default `http://192.168.10.118:8080`)

Wake word:
- `wakeword_model` (default `/home/artur/openWakeWord/my_custom_model/ey_lyonya.onnx`)
- `wakeword_threshold` (default `0.9`)
- `wakeword_patience` (default `2`; consecutive 80 ms frames above the threshold)

Audio:
- `audio_device_index` (default `-1`), `audio_device_name` (default `pulse`), `output_device_name` (default `''`)
- `sample_rate` (default `16000`)
- `pa_source_check` (default `Jabra`, empty string disables the check)

VAD / speaker verification:
- `vad_threshold` (`0.4`), `silence_duration_sec` (`2.5`), `pipeline_timeout_sec` (`45.0`)
- `speaker_verification` (`true`), `sv_threshold` (`0.35`, phrase-level), `sv_min_speech_sec` (`1.2`), `sv_debug_dir` (`~/inmoov_sv_debug`, WAVs of judged phrases for tuning)

Tavily:
- `tavily_api_key` (default from env `TAVILY_API_KEY`)

Arduino:
- `port_right` (default `/dev/serial/by-path/pci-0000:c6:00.3-usb-0:5:1.0-port0`)
- `port_left` (default `/dev/serial/by-path/pci-0000:c6:00.3-usb-0:1.3:1.0-port0`)

Memory:
- `memory_db_path` (default `/home/artur/inmoov_memory.db`)

Vision:
- `vision` (default `true`) — gates all camera/face/OAK nodes via `IfCondition`; with `false` they are neither started nor managed by `lifecycle_manager`
- `cam_left` (default `/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1.1:1.0-video-index0`)
- `cam_right` (default `/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1.2:1.0-video-index0`)
- `fps` (`15`), `detection_hz` (`5.0`), `det_thresh` (`0.5`), `analysis_hz` (`2.0`)
- `gain_head` (`0.3`), `gain_eye` (`0.6`)
- `rest_rothead` (`90.0`), `rest_neck` (`40.0`)
- `oak_model` (`yolov6-nano`), `oak_conf_threshold` (`0.5`)
- `scene_location` (default `''`)

Cognition:
- `greet_cooldown_sec` (`120.0`), `bt_tick_rate_hz` (`10.0`)

Telegram:
- `telegram` (default `false`) — gates the whole tier-6 `telegram_bridge_node` group via `IfCondition`
- `allowed_chat_id` (default from env `TELEGRAM_ALLOWED_CHAT_ID`)

Example overrides:

```bash
ros2 launch inmoov_bringup inmoov.launch.py tavily_api_key:=tvly-...
ros2 launch inmoov_bringup inmoov.launch.py vision:=false
ros2 launch inmoov_bringup inmoov.launch.py telegram:=true
```

The `lifecycle_manager` executable can also be run standalone (mainly for
debugging one node's lifecycle behavior in isolation):

```bash
ros2 run inmoov_bringup lifecycle_manager --ros-args -p config_file:=<path-to-lifecycle.yaml>
```

## Requirements / Setup

Per [`package.xml`](package.xml), this package depends on `rclpy`,
`lifecycle_msgs`, and the following sibling packages, all of which must be
built in the same workspace and provide the nodes referenced in
`config/lifecycle.yaml` / `launch/inmoov.launch.py`:

- `inmoov_cognition` — `llm_node`, `openhab_bridge_node`, `identity_manager_node`, `behavior_manager_node`, `telegram_bridge_node`
- `inmoov_control` — `arduino_left_node`, `arduino_right_node`, `joint_state_publisher`, `face_expressions_node`
- `inmoov_memory` — `memory_node`
- `inmoov_vision` — `face_capture_node`, `oak_node`, `face_detection_node`, `face_tracker_node`, `face_recognition_node`, `face_gallery_node`, `emotion_recognition_node`, `vision_head_tracker_node`, `human_detection_node`, `scene_manager_node`
- `inmoov_voice` — `audio_source_node`, `sound_localization_node`, `wakeword_node`, `voice_detector_node`, `tts_node`, `voice_emotion_node`, `parakeet_stt_node`

All of the above nodes must implement the ROS2 managed-lifecycle interface
(`change_state`/`get_state` services) since `lifecycle_manager` drives them
purely through that protocol.

Other assumptions baked into the default launch arguments (all overridable):
- Arduino Mega boards reachable at fixed `/dev/serial/by-path/...` device paths.
- USB eye cameras reachable at fixed `/dev/v4l/by-path/...` device paths.
- PulseAudio/PipeWire user session reachable at `/run/user/<uid>/pulse/native`
  (audio nodes are launched with `PULSE_SERVER` and `DBUS_SESSION_BUS_ADDRESS`
  set accordingly in `additional_env`).
- An OpenAI-compatible LLM endpoint and a TTS HTTP server reachable at the
  configured URLs (defaults point at `192.168.10.118`).
- An openHAB instance for `openhab_bridge_node` (default `http://192.168.10.118:8080`).

The build is `ament_python` (see `package.xml` export), installed via
`setup.py`; no external Python dependencies beyond `setuptools` and
`PyYAML` (imported directly by `lifecycle_manager.py` for
`config/lifecycle.yaml`).

Code comments in this workspace (outside this package) reference an
`inmoov_start.sh` script and a systemd user service used to start/stop/restart
the whole robot stack — that tooling is not part of `inmoov_bringup` itself,
so its details aren't documented here.
