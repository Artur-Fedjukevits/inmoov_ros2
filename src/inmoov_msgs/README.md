# inmoov_msgs

Custom ROS2 message, service, and action definitions shared across the InMoov
packages. Pure interface definitions — no nodes, no logic.

## Interfaces

### `msg/SoundDirection.msg`

Direction to a sound source from a stereo microphone pair, produced by
`inmoov_voice`'s `sound_localization_node`. See the message file's header
comment for the full method (band-passed GCC-PHAT + sign-vote) and its
history/limitations — in particular, `angle_deg` is **not** a physical angle,
just a confidence-weighted left/right lean, and front/back sound sources are
inherently ambiguous.

Fields: `header`, `angle_deg`, `tdoa_us`, `confidence`, `rms_dbfs`, `voiced`.

Published on `/sound_direction` by `sound_localization_node`; consumed by
`inmoov_cognition`'s `behavior_manager_node` to help resolve which detected
person is speaking.

### `srv/MemoryQuery.srv`

Generic JSON-in/JSON-out request to `inmoov_memory`'s `memory_node`.

```
string request_json
---
string response_json
bool success
```

`request_json` carries `{"op": "...", ...op-specific args}`; the set of
supported `op` values and their request/response shapes are defined by
`memory_node` — see `inmoov_memory/README.md` for the full list. Called from
`inmoov_cognition` (`identity_manager_node`, `llm_node`) and from
`inmoov_vision` (`face_gallery_node`, `face_recognition_node`) and the
`inmoov_memory/scripts/rebuild_gallery.py` maintenance script.

### `action/Speak.action`

Text-to-speech request, served by `inmoov_voice`'s `tts_node` and called from
`inmoov_cognition` (`behavior_manager_node`, `llm_node`, `telegram_bridge_node`).

```
# Goal — what to say
string text
string voice        # OmniVoice preset name ("neutral"/"happy"/"sad"/"surprise", "" = neutral)
float32 rate        # speed: 0.0/1.0 = normal, 0.5 = slow, 2.0 = fast
---
# Result
bool success
string message      # error description if !success
float32 audio_sec   # actual playback duration
---
# Feedback (published during synthesis and playback)
string status       # "connecting" | "synthesizing" | "playing" | "done"
float32 progress    # 0.0..1.0 if Content-Length is known, else -1.0
int32 bytes_played  # bytes played back (for gesture sync)
```

## Requirements / Setup

Standard `ament_cmake` interface package — no runtime dependencies beyond
`std_msgs` and the `rosidl` toolchain (`rosidl_default_generators` at build
time, `rosidl_default_runtime` at run time). Building it generates the
Python/C++ bindings consumed by every other `inmoov_*` package; run
`colcon build` after any change here before rebuilding dependents.

## License

GNU General Public License v3.0 — see the repository root `LICENSE`.
