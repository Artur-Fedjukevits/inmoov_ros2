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

- **Startup**: after `autostart_delay_sec`, activates tiers 0→6 in order.
  Within a tier, all nodes are configured+activated in parallel threads, each
  with up to `retry_count` attempts (linearly increasing backoff:
  `retry_interval_sec * attempt`). If a `critical` node in a tier fails all
  retries, the tier — and the whole activation sequence — aborts and system
  state becomes `fault`. A non-critical failure just marks that node
  `degraded` (reason `activation`) and the tier continues.
- **Watchdog**: starts `watchdog_startup_delay_sec` after the system reaches
  `active`, then polls `get_state` on every node every
  `watchdog_interval_sec`. Detects two death scenarios: the service call gets
  no response at all (process gone, no respawn yet), or the node answers but
  is back in `UNCONFIGURED` (process died and `ros2 launch respawn=True`
  already restarted it, but it hasn't been configured/activated again). A
  dead critical node triggers `_cascade_deactivate`, which `DEACTIVATE`s every
  node in every tier above it (they're assumed to depend on it) and marks
  them `degraded` with reason `cascade_<tier_idx>`. Recovery
  (`_recover_node`) re-runs the normal activate-with-retry logic against the
  respawned process; on success for a critical node it waits 3s (to avoid a
  race with an in-flight `_cascade_deactivate`) then calls
  `_cascade_recover`, which re-activates every tier that was cascade-degraded
  because of this node. `max_respawn_count` caps how many times the watchdog
  will attempt to recover a single node before giving up permanently.
- **SLEEP / WAKE**: a fixed set of vision + VAD nodes
  (`_SLEEP_DEACTIVATE` in the source — `face_capture_node`,
  `face_detection_node_{left,right}`, `face_tracker_node_{left,right}`,
  `face_recognition_node`, `face_gallery_node`, `emotion_recognition_node`,
  `vision_head_tracker_node`, `human_detection_node`, `oak_node`,
  `voice_detector_node`) get `DEACTIVATE`d on `SLEEP` and re-activated on
  `WAKE`. Triggered either by the `/lifecycle/command` topic or automatically
  by the latched `/robot_sleep` topic (published elsewhere in the system,
  e.g. by `inmoov_cognition`).
- **Shutdown**: `DEACTIVATE`s, then `CLEANUP`s, then shuts down every node,
  tier by tier from the highest tier down to tier 0, stopping the watchdog
  first.

**Parameters**

| Name | Type | Default | Meaning |
|---|---|---|---|
| `retry_count` | int | `3` | Activation attempts per node before giving up. |
| `retry_interval_sec` | double | `10.0` | Base backoff between retries; actual wait is `retry_interval_sec * attempt`. |
| `transition_timeout_sec` | double | `30.0` | Timeout waiting for a `change_state`/`get_state` service call. |
| `tier_advance_timeout_sec` | double | `90.0` | Max time to wait (via `Thread.join`) for all nodes in a tier to finish activating before moving on. |
| `config_file` | string | `''` | Path to a YAML file (see `config/lifecycle.yaml`) defining `tiers` and overriding the parameters above. If empty, the manager waits for `configure_tiers()` to be called programmatically instead. |
| `autostart_delay_sec` | double | `5.0` | Delay after startup before automatically activating tier 0. If tiers aren't configured yet, or this is `<= 0`, autostart is skipped. |
| `watchdog_interval_sec` | double | `5.0` | Polling interval for the watchdog loop. Set `<= 0` to disable the watchdog entirely. |
| `watchdog_startup_delay_sec` | double | `15.0` | Grace period after full system activation before the watchdog starts polling (avoids false positives while nodes are still settling). |
| `max_respawn_count` | int | `5` | Max watchdog-driven recovery attempts per node before it's left permanently `degraded`. |

**Topics**

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/lifecycle/status` | `std_msgs/String` (JSON) | publish, every 2s | `{system, sleep_mode, degraded_nodes[], recovering_nodes[], tiers: [{id, nodes: {name: {critical, degraded, degraded_reason, respawn_count, recovering, attempts}}}]}`. `system` is one of `starting`, `active`, `fault`, `degraded`, `sleep`, `waking`, `shutdown`. |
| `/lifecycle/command` | `std_msgs/String` | subscribe | Commands: `SHUTDOWN`, `SLEEP`, `WAKE`, `RESTART_TIER <N>`. (`ACTIVATE`/`DEACTIVATE` are described in the module docstring but are not currently implemented as command handlers — activation only happens via autostart or `start_activation()`.) |
| `/robot_sleep` | `std_msgs/Bool` (latched, `TRANSIENT_LOCAL`/`RELIABLE`, depth 1) | subscribe | Drives the same SLEEP/WAKE logic as the command topic; published elsewhere in the system. |

**Services used (per managed node)**

- `/<node_name>/change_state` (`lifecycle_msgs/srv/ChangeState`)
- `/<node_name>/get_state` (`lifecycle_msgs/srv/GetState`)

**Known issues / TODOs (from code comments)**

- The module docstring documents an `ACTIVATE`/`DEACTIVATE` pair of commands
  on `/lifecycle/command`, but `_command_cb` only handles `SHUTDOWN`, `SLEEP`,
  `WAKE`, and `RESTART_TIER N` — `ACTIVATE`/`DEACTIVATE` fall through to the
  "unknown command" branch.
- `_recover_node`'s comment notes a deliberate 3-second sleep before
  `_cascade_recover` to avoid a race where recovery could run before
  `_cascade_deactivate` has finished marking all affected tiers
  `cascade_<N>` — the sleep is a heuristic, not a synchronization guarantee.
- `_cascade_recover` deliberately uses `continue` (not `break`) when a tier
  wasn't cascade-degraded, since tiers can be marked `cascade_N` out of order
  under race conditions with `_cascade_deactivate`.

## Launch

```bash
ros2 launch inmoov_bringup inmoov.launch.py
```

The launch file ([`launch/inmoov.launch.py`](launch/inmoov.launch.py)) starts
`lifecycle_manager` as a plain `Node` (pointed at `config/lifecycle.yaml` via
the `config_file` parameter) plus every managed node as a `LifecycleNode`,
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
(5); `telegram_bridge_node` (6, only if `telegram:=true`).

### Launch arguments

Servers / LLM:
- `llm_url` (default `http://192.168.10.118:18020/v1/chat/completions`) — OpenAI-compatible chat.completions endpoint (vLLM), shared by `llm_node` and `identity_manager_node` (name extraction).
- `llm_fallback_url` (default `http://localhost:11434/v1/chat/completions`) — backup endpoint.
- `llm_bearer_token` (default from env `VLLM_BEARER_TOKEN`)
- `llm_model` (default `qwen3.8-27b`), `llm_temperature` (`0.1`), `llm_max_tokens` (`512`)
- `tts_server_url` (default `http://192.168.10.118:8000`), `tts_fallback_url` (default `http://localhost:8000`)
- `openhab_url` (default `http://192.168.10.118:8080`)

Wake word:
- `wakeword_model` (default `/home/artur/openWakeWord/my_custom_model/ey_lyonya.onnx`)
- `wakeword_threshold` (default `0.3`)

Audio:
- `audio_device_index` (default `-1`), `audio_device_name` (default `pulse`), `output_device_name` (default `''`)
- `sample_rate` (default `16000`)
- `pa_source_check` (default `Jabra`, empty string disables the check)

VAD / speaker verification:
- `vad_threshold` (`0.5`), `silence_duration_sec` (`2.5`), `pipeline_timeout_sec` (`45.0`)
- `speaker_verification` (`true`), `sv_threshold` (`0.45`), `sv_segment_sec` (`3.0`)

Tavily:
- `tavily_api_key` (default from env `TAVILY_API_KEY`)

Arduino:
- `port_right` (default `/dev/serial/by-path/pci-0000:c6:00.3-usb-0:5:1.0-port0`)
- `port_left` (default `/dev/serial/by-path/pci-0000:c6:00.3-usb-0:1.3:1.0-port0`)

Memory:
- `memory_db_path` (default `/home/artur/inmoov_memory.db`)

Vision:
- `vision` (default `true`) — declared but not currently wired to any `IfCondition` in the launch file
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
