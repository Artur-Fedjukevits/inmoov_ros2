# inmoov_cognition

Cognition layer for the InMoov robot: the **Data → Decision → Action**
pipeline that turns raw perception into dialogue and physical behavior.

```
                 ┌────────────────────┐
 face/voice/PIR  │  IdentityManager   │  /social_context (JSON, ~2 Hz)
────────────────▶│ (identity_manager_ │───────────────┐
 scene/OpenHAB    │      node)         │               │
                  └────────────────────┘               ▼
                                              ┌───────────────────────┐
 STT (voice_command)                          │   Behavior Tree       │
────────────────────▶┌──────────────┐         │ (behavior_manager_    │
                      │   llm_node   │  tool   │      node)            │
                      │ (LLM + tools)│  calls  │  ticks @ 10 Hz over   │
                      │              │────────▶│  a shared Blackboard  │
                      └──────┬───────┘/robot_  └──────────┬────────────┘
                     text/llm_response events              │
                             │                              ▼
                             │                    Speak / joints / face /
                             ▼                    arm / torso / search
                         tts_node
                    (outside this package)

     openhab_bridge_node  ⇄  OpenHAB REST/WebSocket  ⇄  smart home
     telegram_bridge_node ⇄  Telegram Bot API         ⇄  remote control
```

- **IdentityManager** (`identity_manager_node`) fuses face recognition, voice
  identification and emotion into a social context and publishes it on
  `/social_context` (JSON, ~2 Hz) plus a few fast side-channel topics
  (`/person_present`, `/introducing`, `/face_expression`). It runs its own
  small state machine (`IDLE → RECOGNIZING → INTERACTING`, with a parallel
  `INTRODUCING` branch for meeting new people) but never commands actuators
  directly — it is a pure data/decision producer.
- **LLM node** (`llm_node`) is the "intent generator": it calls an
  OpenAI-compatible `chat.completions` endpoint (vLLM) with a function-calling
  (`tools`) schema, streams the reply straight into TTS sentence-by-sentence,
  and turns tool calls into either `/robot_events` (physical actions handled
  by the Behavior Tree) or its own side effects (OpenHAB control, memory
  writes, weather, web search, Telegram broadcast). It never touches
  `/joint_command` or the TTS action client for *robot commands* — those go
  through the Behavior Tree.
- **Behavior Tree** (`behavior_manager_node`) is the single orchestrator. A
  `py_trees` "eternal tree" (a no-memory `Selector` re-evaluated from the top
  every tick, `bt_tick_rate_hz` default 10 Hz) reads a shared **Blackboard**
  (the single source of truth) and drives everything physical: speech
  (`Speak` action → `tts_node`), gestures, head/torso motion, face
  expressions, sleep transitions, web search, and PIR/sound-triggered
  scanning for a face.
- **openHAB bridge** (`openhab_bridge_node`) mirrors OpenHAB item state into
  ROS topics for `llm_node`'s smart-home tools, and separately watches
  environmental sensors, pushing Telegram alerts on threshold crossings.
- **Telegram bridge** (`telegram_bridge_node`) is an optional remote-control
  channel: it reuses the exact same `llm_node` pipeline (tool calls, memory,
  smart home) via `/telegram_ask` / `/telegram_response`, plus a few direct
  commands (`/photo`, `/say`, `/status`, `/where`, `/sleep`, `/wake`,
  `/restart_tts`).

All five nodes are ROS 2 **managed-lifecycle nodes**
(`rclpy.lifecycle.LifecycleNode`) and stay `Unconfigured` until driven through
`configure → activate`, normally by `lifecycle_manager` in
[`inmoov_bringup`](../inmoov_bringup/README.md).

> **Note on language**: system prompts, tool/function descriptions sent to
> the LLM, spoken/Telegram-facing text, and keyword/regex lists matched
> against Russian speech transcripts are intentionally left in **Russian** in
> the source — the robot's assistant persona and the household it serves are
> Russian-speaking. Only code comments, docstrings and developer log messages
> have been translated to English for this public release.

## Nodes

Executables registered in [`setup.py`](setup.py):

| Executable | Module |
|---|---|
| `behavior_manager_node` | `inmoov_cognition.behavior_manager_node:main` |
| `identity_manager_node` | `inmoov_cognition.identity_manager_node:main` |
| `llm_node` | `inmoov_cognition.llm_node:main` |
| `openhab_bridge_node` | `inmoov_cognition.openhab_bridge_node:main` |
| `telegram_bridge_node` | `inmoov_cognition.telegram_bridge_node:main` |

---

### `identity_manager_node`

Source: [`inmoov_cognition/identity_manager_node.py`](inmoov_cognition/identity_manager_node.py).

"Social Context Provider" — combines face recognition/tracking, voice
identification and emotion recognition into one social context for the
Behavior Tree. Publishes data only; never commands actuators (with one
exception: a low-level reflex publish to `/face_expression` for
emotion-mirroring).

**State machine**: `IDLE` (no face) → `RECOGNIZING` (face present, waiting
for a confident identification) → `INTERACTING` (identity known, tracking
emotion), with an `INTRODUCING` branch entered from `RECOGNIZING` when the
face is confidently *unknown* — the node then leads a short voice dialogue to
learn the person's name (via `/voice_command`, bypassing `llm_node`) and
enrolls them (see the identity-verification tiers below).

Runs left/right-eye track topics with **left as primary, right as
fallback** (switches after `track_eye_fallback_sec` of silence from the left
camera) and a background watchdog (`_watchdog`, 0.5 s timer) that returns to
`IDLE` when both face and OAK-D body detection have been absent long enough
(see *Known issues* for the exact rules), or when the OAK-D presence signal
never showed up in the first place ("OakD veto" against `face_detection`
false positives).

**Identity verification tiers** (`_verify_claimed_identity`, used when
someone claims a name that already exists in the DB during `INTRODUCING`):
`face_sim < FACE_VETO (0.28)` → treated as a different person (voice
ignored); `FACE_VETO ≤ face_sim < FACE_ACCEPT (0.45)` → uncertain, voice used
as a tiebreaker (`VOICE_ACCEPT = 0.52`) or the person is asked directly;
`face_sim ≥ FACE_ACCEPT` → accepted without checking voice.

**Parameters**

| Name | Type | Default | Meaning |
|---|---|---|---|
| `no_face_timeout_sec` | double | `15.0` | No face seen for this long (×4 while `INTRODUCING`) starts the "face lost" countdown. |
| `no_human_timeout_sec` | double | `20.0` (`30.0` in `behavior_manager.launch.py`) | No OAK-D body signal for this long ⇒ OakD veto ⇒ back to `IDLE`. |
| `max_face_hunt_sec` | double | `90.0` | Once a body is present but the face is lost, how long to keep waiting before giving up and resetting to `IDLE`. |
| `greet_cooldown_sec` | double | `120.0` | Minimum time between greetings for the same name. |
| `emotion_react_thresh` | double | `0.70` | Minimum confidence for a face/voice emotion to be considered for fusion. |
| `introduce_cooldown_sec` | double | `60.0` | Minimum time between starting a new introduction flow. |
| `max_introduce_attempts` | int | `3` | Failed name-collection attempts before giving up and going to `INTERACTING` anonymously. |
| `min_enroll_det_score` | double | `0.65` | Minimum face-detector confidence for an embedding to be accumulated during enrollment. |
| `dialogue_switch_timeout_sec` | double | `30.0` | How long the current person must be absent before switching the active dialogue to someone else. |
| `track_eye_fallback_sec` | double | `2.0` | Silence from the left-eye track topic before falling back to the right eye. |
| `post_goodbye_ignore_sec` | double | `1800.0` | After an explicit `say_goodbye`, ignore that name for this long (or until the wake word). |
| `post_goodbye_track_block_sec` | double | `30.0` | Block face-track processing entirely for this long right after a goodbye (breaks a re-greet loop). |
| `llm_url` / `llm_fallback_url` | string | `http://192.168.10.118:18020/v1/chat/completions` / `http://localhost:11434/v1/chat/completions` | OpenAI-compatible endpoint used *only* for name extraction from a free-form voice reply (`_extract_name`'s LLM fallback). **Private-LAN default — override for your setup.** |
| `bearer_token` | string | `''` | Bearer token for `llm_url`. |
| `name_extract_model` | string | `qwen3.8-27b` | Model used for name extraction. |
| `voice_high_threshold` | double | `0.62` | Voice-similarity threshold for a confident identification. |
| `voice_uncertain_threshold` | double | `0.50` | Voice-similarity threshold below which a match is discarded. |
| `gaze_yaw_threshold` | double | `0.30` | Max nose/eye-midpoint offset (fraction of inter-eye distance) still counted as "looking at the robot". |
| `gaze_frontal_fraction` | double | `0.50` | Fraction of the last 15 frontal-score samples that must be "frontal" for `looking_at_robot=True`. |

**Topics — subscribe**

| Topic | Type | Notes |
|---|---|---|
| `/face/identity` | `std_msgs/String` (JSON) | Per-track identification result from `face_recognition_node` (`is_known`, `confidence`, `person_id`, `name`, `best_candidate_*`). |
| `/face/emotion` | `std_msgs/String` (JSON) | Face emotion for the primary track. |
| `/voice/emotion` | `std_msgs/String` (JSON) | Voice emotion (wav2vec2-IEMOCAP), fused with face emotion within a 15 s window. |
| `/face/tracks/left`, `/face/tracks/right` | `std_msgs/String` (JSON) | Per-eye track lists (bbox, embedding, keypoints, det_score) — dual-eye topology, left primary. |
| `/wake_detected` | `std_msgs/Bool` | Wake-word event — clears post-goodbye cooldowns and exits sleep. |
| `/human_detected` | `std_msgs/Bool` | OAK-D coarse body-presence signal, used as the veto/fallback signal. |
| `/voice_command` | `std_msgs/String` | STT transcript — intercepted only while `INTRODUCING`. |
| `/voice_embedding` | `std_msgs/String` (JSON) | Speaker-verification embedding from `voice_detector_node`; drives voice-based identification/enrollment. |
| `/robot_sleep` | `std_msgs/Bool` (latched) | Sleep mode — an instant reset to `IDLE`, bypassing the normal watchdog path. |
| `/go_idle` | `std_msgs/Bool` | Forced transition to `IDLE` (from the BT's `say_goodbye` handling). |
| `/llm_response` | `std_msgs/String` | Only used to timestamp "dialogue active" (resets the face-hunt timer). |

**Topics — publish**

| Topic | Type | Notes |
|---|---|---|
| `/social_context` | `std_msgs/String` (JSON) | @ 2 Hz (0.5 s timer). Fields: `person_present, person_id, name, is_known, emotion, state, introducing, looking_at_robot, should_greet, greet_text, introduce_pending, introduce_text`. `should_greet`/`introduce_pending` are one-shot. |
| `/person_context` | `std_msgs/String` (JSON) | Fetched from `inmoov_memory` (`get_context`) once a person is identified; consumed by `llm_node` for the system prompt's person block. |
| `/person_present` | `std_msgs/Bool` | Fast interrupt signal, also republished @ 2 Hz to keep the voice pipeline's presence timeout alive during a dialogue. |
| `/introducing` | `std_msgs/Bool` | Gate for `llm_node` (blocks normal command handling during name collection). |
| `/face_expression` | `std_msgs/String` | Low-level emotion-mirroring reflex, published directly (not through the BT). |
| `/voice_anchor` | `std_msgs/String` (JSON) | Voice gallery for a newly identified/enrolled person, sent to `voice_detector_node` for speaker verification. |
| `/robot_sleep` | `std_msgs/Bool` (latched) | Re-published on `False` when a wake word ends sleep. |

**Services used**: `/memory/query` (`inmoov_msgs/srv/MemoryQuery`) — person
lookup/save/update, reminders, voice-gallery storage; all calls are
synchronous-over-async with a 2–5 s timeout via a `threading.Event`.

**External dependencies**: `inmoov_memory` (`MemoryQuery` service), an
OpenAI-compatible LLM endpoint (name extraction only).

**Known issues / TODOs (from code comments)**

- Live bug fixed 2026-08-24: an old "LEFT=120/RIGHT=60" rothead convention
  note was wrong; verified by hand and corrected (see the same convention
  documented in `behavior_manager_node.py`).
- Live bug fixed 2026-08-28: "I am to your left" voice hints used to be
  silently dropped by a `person_present` gate that raced with recognition —
  fixed by not gating the direction-hint publish on current presence.
- Live bug fixed 2026-08-31: track reassignment by embedding similarity
  (instead of bbox area) was tried and reverted — it didn't fix the
  underlying head-drift issue and added complexity; track selection is by
  bounding-box area again.
- `_extract_name` falls back to an LLM call only for phrases a fast regex
  can't parse — the regex/stopword lists and the LLM system prompt are
  intentionally Russian (see the language note above).

---

### `llm_node`

Source: [`inmoov_cognition/llm_node.py`](inmoov_cognition/llm_node.py) (the
largest file in the package, ~3200 lines).

The "intent generator": calls an OpenAI-compatible `chat.completions`
endpoint with function calling, streams sentence-sized chunks straight into
TTS as they arrive (`_stream_with_tts`, target TTFA ~2–4 s — see
`project_llm_tts_streaming.md` in the project memory), and resolves tool
calls in up to three rounds (R1 → R2 → R3) depending on whether a tool
already produced a `speak_text` or the LLM needs to see tool results before
answering. Never controls TTS/servos directly — dialogue text goes to
`/llm_response` and physical actions go to `/robot_events`, both consumed by
`behavior_manager_node`'s Blackboard.

Also serves `/telegram_ask` requests through the *exact same* pipeline
(`_query_llm`), with `telegram=True` in the response so the BT doesn't set
`person_present`.

**Addressee gate**: `command_callback` drops a `/voice_command` utterance
unless the person is looking at the robot (per `/social_context`'s
`looking_at_robot`) or the phrase starts with one of the robot's name
variants ("Лёня"/"Леня"/"Эй" etc., case-insensitive) — see
`_addressed_to_robot`.

**LLM backend configuration** (parameters, all declared with `_dp` so
re-configure is safe):

| Name | Default | Meaning |
|---|---|---|
| `llm_url` | `http://192.168.10.118:18020/v1/chat/completions` | Primary OpenAI-compatible endpoint (vLLM). **Private-LAN default — override.** |
| `llm_fallback_url` | `http://localhost:11434/v1/chat/completions` | Fallback endpoint (currently not a working OpenAI-compatible server — see code comment). |
| `bearer_token` | `''` | Bearer token for `llm_url`. Set via the `llm_bearer_token` launch arg, which defaults to the `VLLM_BEARER_TOKEN` env var — **never hardcode a real token**. |
| `bearer_token_fallback` | `''` | Bearer token for `llm_fallback_url`. |
| `model` / `model_fallback` | `qwen3.8-27b` / `qwen2.5:7b` | Model names for the two backends. |
| `vision_llm_url` | `http://192.168.10.118:18090/v1/chat/completions` | Separate vision-capable endpoint for `look_and_describe`/`look_direction` (accepts 2–4 images per request; the main `llm_url` is limited to 1). |
| `vision_model` | `qwen3vl` | Vision model name. |
| `vision_bearer_token` | `''` | Falls back to `bearer_token` if empty. |
| `temperature` | `0.1` | Sampling temperature. |
| `max_tokens` | `512` | Max tokens for the main response. |
| `connect_timeout_sec` / `timeout_sec` | `5.0` / `120.0` | HTTP connect / total timeouts. |
| `keep_history` | `True` | Whether multi-turn history is sent (vs. only the latest user turn). |
| `history_max_turns` | `8` | History is trimmed by counting `user`-role turns, not raw message count. |
| `openhab_url` | `http://192.168.10.118:8080` | Used by the `items_control`/etc. tools. |
| `tts_server_url` / `tts_fallback_url` | `http://192.168.10.118:8000` / `http://localhost:8000` | Declared but the actual TTS calls go through the `Speak` action client, not a direct HTTP call from here. |
| `cast_volume` | `80` | Default Chromecast volume for `broadcast_message`. |
| `cast_to_file_url` | `http://192.168.10.118:8000/tts/to_file` | TTS-to-WAV-file endpoint used by `broadcast_message`. |

**Tools exposed to the LLM** (`TOOLS`, OpenAI function-calling schema — names
and behavior described here in English; the schema's own `description`
fields sent to the LLM are **not** translated, per the language note above):

| Tool | What it does |
|---|---|
| `get_weather` | yr.no (MET Norway) forecast for a location/date; always preferred over `web_search` for weather. |
| `items_control` | Sets a single OpenHAB item's state (Switch/Dimmer/Number/String/Color). |
| `get_openhab_states` | Reads the cached current state of named OpenHAB items. |
| `search_openhab_items` | Finds OpenHAB items by semantic group / state / name substring. |
| `robot_control` | Physical commands: `arm`, `head` (with optional `partial`/`full` torso assist), `status`, `sleep`, `goodbye` — emitted on `/robot_events` for the BT. |
| `web_search` | Tavily-backed internet search, routed through `/robot_events` → `behavior_manager_node`'s `WebSearchBehaviour` (synchronized back via a `threading.Event`). |
| `set_reminder` | Creates a person-scoped reminder via `/memory/query` (date/time resolved from natural-language hints by `_resolve_reminder_date`/`_resolve_reminder_time`). |
| `confirm_reminder` | Deletes reminders after the user acknowledges them. |
| `set_voice_style` | Buffers a voice preset (`neutral/happy/sad/surprise`) for the next `/llm_response`; `tts_node` mirrors the same emotion on the face for the duration of speech. |
| `save_memory` | Saves a lasting fact (person note or general knowledge) via `/memory/query`. |
| `search_memory` | Semantic search over long-term memory via `/memory/query`. |
| `broadcast_message` | TTS-to-file → sets Chromecast volume → pushes the URI to `LivingRoom_Chromecast_uri` (a "announce on the living-room speaker" tool). |
| `merge_persons` | Merges a duplicate person record into a target one (with a face/voice similarity check in `memory_node`), for fixing accidental double-enrollment. |
| `look_and_describe` | One-shot snapshot from both eye cameras right now, analyzed by the vision model. |
| `look_direction` | Turns the head/torso via `/robot_events`, **waits** for the physical turn to finish, takes two snapshots per eye (before/after settling), then analyzes all frames with the vision model — synchronous, so turn-then-look ordering is guaranteed (unlike separately calling `robot_control` + `look_and_describe`, which can race). |

**Topics — subscribe**: `voice_command`, `/telegram_ask`, `search_result`,
`person_context`, `/social_context`, `/scene/objects`,
`/behavior/face_search_status`, `openhab_schema`, `openhab_items`,
`/camera/eye_left/compressed`, `/camera/eye_right/compressed`,
`/introducing`, `/go_idle`, `/robot_sleep` (latched), `/memory/context`.

**Topics — publish**: `robot_events` (physical/search/sleep/goodbye
commands), `/llm_response` (dialogue text + `voice_instruct` + `streamed` +
`telegram` flags), `/telegram_response`, `/tts_cancel_queue`,
`/conversation_end` (transcript → `memory_node` saves an episode),
`/voice/direction_hint` (parsed "I'm on the left/right" hints for
`behavior_manager_node`'s face-search retry).

**Actions**: `Speak` (`inmoov_msgs/action/Speak`) client, used both for
direct chunk-by-chunk TTS during streaming and for the initial LLM warmup
request.

**Services used**: `/memory/query` (`inmoov_msgs/srv/MemoryQuery`).

**External dependencies**: an OpenAI-compatible LLM server (vLLM) with a
function-calling-capable model; a separate vision-capable OpenAI-compatible
endpoint; a `tts_node` implementing the `Speak` action; `openhab_bridge_node`
for the smart-home tools; `inmoov_memory` for the memory tools;
`inmoov_msgs`; Tavily API for `web_search` (key passed down from
`behavior_manager_node`'s `tavily_api_key` parameter, not held by `llm_node`
itself — the actual HTTP call lives in `behavior_manager_node`); yr.no /
api.met.no (no key required) for `get_weather`; Nominatim (OpenStreetMap) for
geocoding non-default weather locations.

**Known issues / TODOs (from code comments)**

- `llm_fallback_url` defaults to a local Ollama-style endpoint that is
  explicitly noted as "not currently OpenAI-compatible" — it's a placeholder
  for a future local fallback, not a working failover today.
- The Qwen3-served model sometimes emits tool calls as inline text
  (`ᐈ{...}`, `<tool_call>...</tool_call>`, or a `<tools>` block) instead of
  through the proper `tool_calls` API field — `_extract_text_tool_calls` is a
  fallback parser for this, applied at both R1 and R2.
- `_normalize_tool_calls` patches missing `id`/`type` fields — the SSE
  parser only preserves an `id` if the server actually sent one, and the
  text-fallback path never provides one; vLLM's schema requires both to be
  present when the message is replayed in the next turn's history.
- The model occasionally rephrases the user's question back at the start of
  its answer, or switches into Chinese/Japanese/Korean characters on
  creative tasks — `_ECHO_QUESTION_RE` and `_CJK_RE` filter both before they
  reach TTS.
- A single vision request to `llm_url` is limited to 1 image
  ("At most 1 image(s) may be provided", found 2026-08-26); this is why
  `look_and_describe`/`look_direction` use a *separate* `vision_llm_url`.

---

### `behavior_manager_node`

Source: [`inmoov_cognition/behavior_manager_node.py`](inmoov_cognition/behavior_manager_node.py).

A single "eternal tree" (`py_trees`) ticked at `bt_tick_rate_hz` (default
10 Hz). The **Blackboard is the single source of truth** — all incoming ROS
topics are written into it by callbacks, and the tree only ever reads it and
issues actions; nothing here blocks the ROS executor thread for long (TTS,
web search, and torso motion are all async leaves with `RUNNING` status).

**Blackboard keys** (written by `BehaviorManagerNode`, read by the tree):

```
/robot/sleep, /robot/sleep_requested, /robot/sleep_text, /robot/command
/llm/text, /llm/voice_style, /llm/emotion, /llm/has_content
/social/person_present, /social/name, /social/emotion, /social/should_greet,
/social/greet_text, /social/farewell_pending, /social/farewell_text,
/social/introducing, /social/introduce_pending, /social/introduce_text,
/social/face_search_pending
/search/query, /search/result
/pir/scan_active, /sound/scan_active
/scene/person_count, /scene/objects_summary, /scene/location
```

**Tree structure** (Selector, no-memory — re-evaluated from the top every
tick; first child to succeed/run wins):

```
Root
├── SleepTransition   — sleep requested → farewell phrase → SetSleepMode
├── SleepActive        — blocks everything below while /robot/sleep=True
├── RobotCommand        — pending physical command from the LLM (move/arm/head/status)
├── WebSearch           — pending Tavily search query from the LLM
├── SocialBranch        — gated on /social/person_present:
│     SocialSelector
│     ├── IntroducingBlock   — name-collection in progress (blocks here)
│     ├── GreetBranch        — one-shot greeting
│     ├── FaceSearchBranch   — retry face search via sound/voice-hint every utterance
│     ├── DialogueBranch     — Speak + Gesticulate in parallel for the LLM's reply
│     └── IdleGaze           — eye tracking + blinking
├── FarewellBranch      — person just left → farewell speech + neutral face
├── SoundScanBranch     — wake word: turn the TORSO toward the voice until /human_detected
├── PIRScanBranch       — pure PIR motion (no voice): scan with the HEAD
└── GlobalIdle          — idle blinking
```

`SocialBranch` is also the **interrupt buffer**: it's a no-memory `Sequence`
gated on `person_present`, so the moment the person leaves,
`CheckPersonPresent` fails, the sequence is torn down, and `terminate(INVALID)`
on a `RUNNING` `SpeakBehaviour` cancels the in-flight TTS goal.

`FaceSearchBranch` lives *inside* `SocialBranch` (not next to
`SoundScanBranch`/`PIRScanBranch`) specifically so it keeps getting a chance
on every dialogue turn — the wake-word scan branches are never ticked once a
dialogue has started.

**Parameters**

| Name | Type | Default | Meaning |
|---|---|---|---|
| `tavily_api_key` | string | `''` | Tavily Search API key for the `WebSearch` leaf. |
| `bt_tick_rate_hz` | double | `10.0` | Tree tick rate. |
| `pir_scan_cooldown_sec` | double | `20.0` | Minimum time between PIR-triggered head scans. |

**Topics — subscribe**: `/llm_response`, `robot_events`, `/social_context`,
`/scene/objects`, `/person_present`, `/pir_state`, `wake_detected`,
`/robot_sleep` (latched), `/sound_direction`
(`inmoov_msgs/msg/SoundDirection`), `/human_detected`, `/human_angle_deg`,
`/head_tracker/face_locked`, `/voice/direction_hint`.

**Topics — publish**: `/face_detection/enable`, `/head_tracker/enable`
(latched, both toggled solely by the BT — it is the single orchestrator of
vision enable/disable), `/go_idle`, `/behavior/face_search_status` (latched
JSON status consumed by `llm_node` for the "ask where you are" prompt block),
`/joint_command` (head/torso/aim commands), `/face_expression`,
`cmd_vel`, `arm_command`, `status_request`, `search_result`, `/face_command`
(eye blink).

**Actions**: `Speak` (`inmoov_msgs/action/Speak`) client, with cancellation
support via `terminate()` for the interrupt buffer.

**External dependencies**: Tavily Search API (optional — `WebSearchBehaviour`
just logs an error without a key); `inmoov_msgs`; `py_trees`.

**Known issues / TODOs (from code comments)**

- `GesticulationAction` is a stub — it always returns `SUCCESS` and only
  distinguishes a "wave" vs. "home" arm pose by emotion; a proper
  emotion→gesture mapping is a TODO.
- `/lifecycle/command`-style `ACTIVATE`/`DEACTIVATE` are not relevant here,
  but several torso/head-motion races are documented in comments and fixed
  incrementally: PIR self-triggering on the robot's own torso motion
  (`_PIR_SELF_MOTION_BLANK_SEC`), a face-lost grace period to avoid
  overreacting to blinks (`_FACE_LOST_GRACE_SEC`), and a fixed
  rothead/midstom turn-direction convention (`LEFT=60°`/`RIGHT=120°` for
  both — re-verified by hand 2026-08-24 after an earlier, wrong note).
- Face-search retry (`FaceSearchAttempt`/`record_face_search_attempt`) went
  through several rounds of live bug fixes; see the project memory notes
  referenced in code comments (`project_face_search_retry.md`) for the
  history — the final round was committed but not fully re-tested per those
  notes.

---

### `openhab_bridge_node`

Source: [`inmoov_cognition/openhab_bridge_node.py`](inmoov_cognition/openhab_bridge_node.py).

Bridges OpenHAB and ROS 2: loads all items tagged `ChatGPT` over REST on
startup, then subscribes to OpenHAB's WebSocket event stream
(`ItemStateChangedEvent`) for real-time updates, republishing the full item
list and a static schema (name/label/type/options, no state — used to build
the `llm_node` system prompt's device table) on ROS topics. Automatically
reconnects the WebSocket with a fixed delay and reloads items on reconnect.

Also runs independent **environment monitoring**, using shared threshold
logic from `inmoov_memory.openhab_alerts` (`classify_sensor`, `parse_value`,
`evaluate_threshold`) — this is a *separate* mechanism from the reminders
database, pushing straight to Telegram:

| Sensor | Alert threshold |
|---|---|
| Temperature | `> 25°C` or `< 16°C` |
| Humidity | `> 70%` or `< 30%` |
| CO₂ | `> 1200 ppm` |
| VOC | `> 300 ppb` |
| Battery (`*_BatteryLow` switches) | `ON` |

Rate-limited to at most one repeat alert per sensor per
`ALERT_RATE_LIMIT_SEC` (900 s = 15 min); recovery (value back to normal)
always sends one confirmation message regardless of the rate limit. Alert
and recovery message text is user-facing (pushed to Telegram) and is
intentionally kept in Russian.

**Parameters**

| Name | Type | Default | Meaning |
|---|---|---|---|
| `openhab_url` | string | `http://192.168.10.118:8080` | OpenHAB base URL. **Private-LAN default — override.** |
| `items_tag` | string | `ChatGPT` | Only items with this tag are loaded/monitored. |
| `reconnect_sec` | double | `15.0` | Delay before retrying a dropped WebSocket connection. |
| `schema_repeat_sec` | double | `10.0` | Schema re-publish interval (for late subscribers). |
| `ws_ping_interval` | int | `30` | WebSocket ping keepalive interval (seconds). |

**Topics — publish**

| Topic | Type | Notes |
|---|---|---|
| `openhab_items` | `std_msgs/String` (JSON) | Full item list with current state; published on startup and on every WS state-change event. |
| `openhab_schema` | `std_msgs/String` (JSON) | Static schema (name/label/type/options); published once on activation and every `schema_repeat_sec`. |
| `/telegram_push` | `std_msgs/String` (JSON `{"text": ...}`) | Environment alert/recovery pushes, consumed by `telegram_bridge_node`. |

**External dependencies**: an OpenHAB instance with REST + WebSocket enabled
at `openhab_url`; `inmoov_memory.openhab_alerts` for the threshold logic;
`websocket-client` (`import websocket`).

---

### `telegram_bridge_node`

Source: [`inmoov_cognition/telegram_bridge_node.py`](inmoov_cognition/telegram_bridge_node.py).

Optional remote-control channel over Telegram (polling, `python-telegram-bot`,
running its own asyncio loop in a daemon thread alongside normal ROS
spinning). `/ask` and any free-form text are routed through the *same*
`llm_node` pipeline as voice commands (`/telegram_ask` → `/telegram_response`,
streamed back by editing the sent Telegram message as chunks arrive), so all
tool calls, smart-home control and personal memory work identically to
talking to the robot in person.

**Identification**: by `telegram_id` in the `persons` table of the SQLite
memory DB (filled in manually — see `inmoov_memory`). On a match, a
`person_ctx` is injected into the `llm_node` request so the robot knows who
it's talking to and can save/recall personal facts.

**Security**: only `allowed_chat_id` receives replies — everyone else is
silently ignored. The bot token is read only from the `TELEGRAM_BOT_TOKEN`
environment variable and is never hardcoded.

**Commands**

| Command | Action |
|---|---|
| `/status` | Mode, person present, emotion, NUC CPU/RAM/disk/temp/uptime, TTS/LLM server health checks, lifecycle-manager node status summary. |
| `/photo` | Snapshot from the left eye camera (JPEG). |
| `/say <text>` | Direct TTS via the `Speak` action, bypassing the LLM. |
| `/ask <text>` | Full `llm_node` pipeline (smart home, memory, web search, tools). |
| `/where <name>` | Sends a family member's live location (OwnTracks via an OpenHAB `Location` item), with fuzzy/transliterated name matching. A bare "where is X" message (no `/where`) is also intercepted automatically before falling through to the LLM. |
| `/wake` | Publishes `/robot_sleep False`. |
| `/sleep` | Publishes `/robot_sleep True` — an instant transition, no LLM involved. |
| `/restart_tts` | Runs `docker restart cosyvoice_api` (the local ROCm TTS fallback container). |
| *any other text / a photo with caption* | Same as `/ask` (a photo is base64-encoded and sent to the LLM's vision path). |

**Parameters** (also settable via [`config/telegram_params.yaml`](config/telegram_params.yaml))

| Name | Type | Default | Meaning |
|---|---|---|---|
| `allowed_chat_id` | int | `0` | **Required** — the only Telegram chat_id allowed to send commands. `0` matches nothing; must be overridden. |
| `llm_timeout_sec` | double | `35.0` | How long to wait for an `llm_node` reply before showing a timeout message. |
| `memory_db_path` | string | `/home/artur/inmoov_memory.db` | Path to the SQLite DB for `telegram_id → person` lookup. **Personal path — override.** |
| `cam_device` | string | `/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1.1:1.0-video-index0` | Left eye camera device for `/photo`. **Personal hardware path — override.** |

`config/telegram_params.yaml` field reference (placeholders shown for the
values that must be supplied per-deployment):

```yaml
telegram_bridge_node:
  ros__parameters:
    allowed_chat_id: <YOUR_TELEGRAM_CHAT_ID>   # numeric; find via @userinfobot
    llm_timeout_sec: 35.0
    memory_db_path: "<PATH_TO_INMOOV_MEMORY_DB>"
    cam_device: "<PATH_TO_LEFT_EYE_CAMERA_DEVICE>"
```

The bot token itself is **not** a config-file field — it is read exclusively
from the `TELEGRAM_BOT_TOKEN` environment variable at process start.

**Topics — subscribe**: `/social_context`, `/robot_sleep` (latched),
`/telegram_response`, `/camera/eye_left/compressed`, `/telegram_push`,
`/openhab_items`, `/lifecycle/status`.

**Topics — publish**: `/robot_sleep` (latched), `/telegram_ask`.

**Actions**: `Speak` (`inmoov_msgs/action/Speak`) client for `/say`.

**External dependencies**: `python-telegram-bot` (`telegram` package), a
Telegram bot token (`TELEGRAM_BOT_TOKEN` env var) and an allowed chat id
(`TELEGRAM_ALLOWED_CHAT_ID` env var, used as the launch argument default);
`opencv-python` (`cv2`) for `/photo`; `psutil` for `/status`; `docker` CLI on
`PATH` for `/restart_tts`; `inmoov_memory`'s SQLite DB for identification.

---

## Launch

```bash
# behavior_manager + identity_manager + openhab_bridge only
ros2 launch inmoov_cognition behavior_manager.launch.py
ros2 launch inmoov_cognition behavior_manager.launch.py tavily_api_key:=tvly-...

# Telegram bridge only (llm_node must already be running)
ros2 launch inmoov_cognition telegram_bridge.launch.py \
  allowed_chat_id:=$TELEGRAM_ALLOWED_CHAT_ID

# The full stack: inmoov_control + inmoov_voice + this package + inmoov_memory
# + inmoov_vision (optional) + Telegram bridge (optional)
ros2 launch inmoov_cognition inmoov_full.launch.py
ros2 launch inmoov_cognition inmoov_full.launch.py vision:=false
ros2 launch inmoov_cognition inmoov_full.launch.py telegram:=true \
  allowed_chat_id:=$TELEGRAM_ALLOWED_CHAT_ID
```

`inmoov_full.launch.py`'s notable arguments (all forwarded to the includes
above): `llm_url`, `llm_fallback_url`, `llm_bearer_token` (defaults from
`VLLM_BEARER_TOKEN`), `tts_server_url`, `tts_fallback_url`, `openhab_url`,
`llm_model`, `wakeword_model`, `wakeword_threshold`, `audio_device_name`,
`tavily_api_key` (defaults from `TAVILY_API_KEY`), `port_right`/`port_left`
(Arduino serial ports), `memory_db_path`, `vision` (bool, default `true`),
`cam_left`/`cam_right`, `greet_cooldown_sec`, `telegram` (bool, default
`false`), `allowed_chat_id` (defaults from `TELEGRAM_ALLOWED_CHAT_ID`).

In production this package is normally started via
[`inmoov_bringup`](../inmoov_bringup/README.md)'s `lifecycle_manager`
(tiers 4–6), not these launch files directly — see that package's README for
the tier breakdown and the `/lifecycle/status`/`/lifecycle/command` API that
`telegram_bridge_node`'s `/status` command reads.

## Requirements / Setup

Per [`package.xml`](package.xml): `rclpy`, `std_msgs`, `geometry_msgs`,
`action_msgs`, `inmoov_msgs`, `inmoov_memory` (all `<depend>`), plus
`python3-websocket` (`exec_depend`, used by `openhab_bridge_node`).

Python packages imported directly by this package's nodes, not declared in
`package.xml` (install via `pip`/system packages as needed):
- `requests` — HTTP calls throughout (`llm_node`, `behavior_manager_node`
  Tavily search, `openhab_bridge_node`/`llm_node` OpenHAB REST calls,
  `llm_node`'s yr.no/Nominatim weather lookups).
- `py_trees` — the Behavior Tree engine (`behavior_manager_node`).
- `websocket-client` (`import websocket`) — `openhab_bridge_node`'s
  real-time OpenHAB event stream.
- `python-telegram-bot` (`import telegram`) — `telegram_bridge_node`.
- `psutil`, `opencv-python` (`cv2`), `numpy` — `telegram_bridge_node`
  (`/status`, `/photo`).

External services / infrastructure this package expects to reach:
- An OpenAI-compatible LLM server with function-calling support (vLLM
  recommended) at `llm_url`, plus a separate vision-capable
  OpenAI-compatible endpoint at `vision_llm_url` for `look_and_describe`/
  `look_direction`.
- A `tts_node` (outside this package, in `inmoov_voice`) implementing the
  `inmoov_msgs/action/Speak` action.
- An OpenHAB instance with REST and WebSocket enabled, with the relevant
  items tagged `ChatGPT` (see `openhab_bridge_node`'s `items_tag`).
- `inmoov_memory`'s `memory_node` (`/memory/query` service, SQLite DB) for
  person identity, notes, reminders and semantic memory.
- Tavily Search API (optional, for `web_search`) — key via `tavily_api_key`
  (`behavior_manager_node` parameter) / `TAVILY_API_KEY` env var.
- A Telegram bot (optional) — token via `TELEGRAM_BOT_TOKEN` env var, chat id
  via `TELEGRAM_ALLOWED_CHAT_ID` env var / `allowed_chat_id` parameter.
- `docker` CLI reachable on `PATH`, with a `cosyvoice_api` container, for
  `telegram_bridge_node`'s `/restart_tts`.

Sibling packages that must be built in the same workspace: `inmoov_memory`
(`memory_node`, `MemoryQuery` service, `openhab_alerts` module),
`inmoov_msgs` (`Speak` action, `SoundDirection` message, `MemoryQuery`
service), and (for the full stack) `inmoov_control`, `inmoov_voice`,
`inmoov_vision`, `inmoov_bringup`.

Default values embedding a private LAN IP (`192.168.10.118`) or a personal
filesystem path (`/home/artur/...`, `/dev/v4l/by-path/...`,
`/dev/serial/by-path/...`) must be overridden for any other deployment —
they are called out individually in the parameter tables above.

## Known issues to verify

- **Face-search retry** (`behavior_manager_node`): a multi-day round of live
  bug fixes culminated in a fix for the head drifting to its hardware limit;
  per code/project-memory comments the final round was committed but not
  re-verified with a live test.
- **`llm_fallback_url`** points at a local endpoint explicitly marked "not
  yet OpenAI-compatible" in a code comment — treat the LLM fallback path as
  non-functional until that's addressed.
- **Dialogue-vs-search race** (documented in project memory, referenced from
  code): an introduction flow starting while a `web_search` is in flight can
  make the search time out because the Behavior Manager is busy with the
  introduction.
- **OAK-D veto tuning**: `no_human_timeout_sec` differs between
  `identity_manager_node`'s own default (`20.0`) and the value
  `behavior_manager.launch.py` actually passes (`30.0`) — worth confirming
  which is intended before relying on the parameter table above in a fresh
  deployment.
- Several torso/head-motion races in `behavior_manager_node` (PIR
  self-triggering, competing scans, rothead/midstom sign conventions) were
  each fixed after being found live; the fixes are in place but are the kind
  of interaction that regressed before, per the comments.

## License

GPL-3.0-only — see the [LICENSE](../../LICENSE) file at the repository root.
