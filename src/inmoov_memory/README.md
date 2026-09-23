# inmoov_memory

Memory layer of the InMoov humanoid robot. The package contains **one ROS2
node**, `memory_node`, which is the single owner of the robot's databases and
exposes everything through one generic JSON service, `/memory/query`
(`inmoov_msgs/srv/MemoryQuery`). It combines two things that used to be
separate:

1. **Social memory** — who the robot knows: people, their face gallery, voice
   gallery, personal notes, general robot knowledge, and reminders.
2. **Three-layer conversational memory** — what the robot remembers about
   *what happened and what was said*:

| Layer | Class | Storage | Lifetime | How it reaches the LLM |
|---|---|---|---|---|
| Working memory | `WorkingMemory` | RAM only | reset on restart | Published every 30 s on `/memory/context` (time, location, mode, people nearby) |
| Episodic memory | `EpisodicMemory` | SQLite (`episodes` table) | short-term (see [Known issues](#known-issues-to-verify)) | Last 5 episodes are appended to `/memory/context`; searchable through `/memory/query` |
| Semantic memory | `SemanticMemory` | SQLite (`facts`) + ChromaDB vector index | long-term | Only on demand (`search_semantic` / `save_semantic_fact` ops), never injected into the prompt automatically |

After every finished dialogue (`/conversation_end`) `memory_node` asks the LLM
(OpenAI-compatible `chat.completions` endpoint, e.g. vLLM) to summarise it,
stores it as an episode, and, if the episode looks important, extracts
structured facts into semantic memory. It can also extract dated reminders
("my concert is on 5 May") from the dialogue.

Everything is Python (`ament_python`). `memory_node` is a ROS2 **managed
lifecycle node** (`rclpy.lifecycle.LifecycleNode`); in the full robot stack it
is the Tier-0 node started by `lifecycle_manager` from
[`inmoov_bringup`](../inmoov_bringup/README.md).

**Language note.** Text that is fed to the LLM or spoken/pushed to a user
(persona prompt, context headers, extraction prompts, reminder keywords,
alert messages) is intentionally in **Russian** and is not translated in the
source; the code comments, docstrings and log messages are English.

## Nodes and modules

Executable registered in [`setup.py`](setup.py):

| Executable | Module |
|---|---|
| `memory_node` | `inmoov_memory.memory_node:main` |

Library modules (no entry point):

| Module | Purpose |
|---|---|
| [`inmoov_memory/memory_node.py`](inmoov_memory/memory_node.py) | The ROS2 node; social DB, caches, `/memory/query` dispatcher, reminders |
| [`inmoov_memory/memory_manager.py`](inmoov_memory/memory_manager.py) | `MemoryManager`: owns the three layers, LLM helper calls |
| [`inmoov_memory/working_memory.py`](inmoov_memory/working_memory.py) | `WorkingMemory` (RAM) |
| [`inmoov_memory/episodic_memory.py`](inmoov_memory/episodic_memory.py) | `EpisodicMemory` (SQLite) |
| [`inmoov_memory/semantic_memory.py`](inmoov_memory/semantic_memory.py) | `SemanticMemory` (SQLite + ChromaDB) |
| [`inmoov_memory/reminder_db.py`](inmoov_memory/reminder_db.py) | `ReminderDB` (SQLite) |
| [`inmoov_memory/openhab_alerts.py`](inmoov_memory/openhab_alerts.py) | Pure functions for sensor/battery thresholds (imported by `openhab_bridge_node` in `inmoov_cognition`) |
| [`scripts/rebuild_gallery.py`](scripts/rebuild_gallery.py) | Stand-alone tool that rebuilds the face-gallery embeddings from photos |

### `memory_node`

Source: [`inmoov_memory/memory_node.py`](inmoov_memory/memory_node.py). Node
name in code: `memory_node`.

Lifecycle behaviour:

- `on_configure`: declares parameters, opens the social SQLite DB (creating /
  migrating tables), loads the face, voice and voice-gallery caches into RAM,
  opens the reminder DB, creates the `MemoryManager` (episodic + semantic DBs +
  ChromaDB), creates the `/memory/query` service, the subscriptions and the
  lifecycle publishers. The **service is therefore already served after
  `configure`**, before `activate`.
- `on_activate`: activates the publishers, starts the `/memory/context` timer,
  starts the Telegram-reminder timer (only if `telegram_reminder_person_id > 0`)
  and publishes the context once immediately.
- `on_deactivate`: destroys the timers, deactivates the publishers.
- `on_cleanup` / `on_shutdown` / `on_error`: close the SQLite connections
  (`on_cleanup` also clears the caches).

**Parameters**

Declared in `on_configure` (safe against re-declaration on re-configure):

| Name | Type | Default | Meaning |
|---|---|---|---|
| `db_path` | string | `/home/artur/inmoov_memory.db` | Social DB: `persons`, `person_gallery`, `person_notes`, `robot_knowledge`, `voice_gallery`. |
| `episodic_db_path` | string | `/home/artur/inmoov_episodic.db` | Episodic memory DB (`episodes`). |
| `semantic_db_path` | string | `/home/artur/inmoov_semantic.db` | Semantic memory SQLite DB (`facts`). |
| `chroma_path` | string | `/home/artur/inmoov_chroma` | ChromaDB persistent directory. |
| `reminder_db_path` | string | `/home/artur/inmoov_reminders.db` | Reminder DB (`reminders`). |
| `llm_url` | string | `http://192.168.10.118:18020/v1/chat/completions` | OpenAI-compatible chat-completions URL used for summaries, fact and reminder extraction. Default is a LAN address of the author's server; override it. |
| `llm_model` | string | `qwen3.8-27b` | Model name sent to that endpoint. |
| `bearer_token` | string | `''` | Optional `Authorization: Bearer ...` token for the LLM endpoint. |
| `similarity_threshold` | double | `0.55` | Face-recognition threshold for `lookup_person` (`confidence: high`). |
| `uncertain_threshold` | double | `0.40` | Lower face threshold: between it and `similarity_threshold` the result is `uncertain`; also used by `gallery_add` to reject mismatching embeddings. |
| `context_publish_rate` | double | `30.0` | Period (s) of `/memory/context` publication. |
| `telegram_reminder_person_id` | int | `5` | `persons.id` whose due reminders are pushed to Telegram; `<= 0` disables the timer. The default is specific to the author's database. |
| `telegram_reminder_check_sec` | double | `60.0` | Period (s) of the Telegram-reminder check. |
| `reminder_default_time` | string | `07:00` | Time of day used for reminders that have a date but no `trigger_time`. |
| `gallery_dir` | string | `/home/artur/inmoov_faces` | Face photo gallery root (same as `face_gallery_node`'s `gallery_dir`); `merge_persons` removes `persons/<from_id>_*` there. |
| `episodic_retention_days` | int | `7` | Episodic sliding window: episodes older than this with low importance are purged. |
| `episodic_cleanup_max_importance` | double | `0.5` | Only episodes with `importance <=` this are purged; more important ones are kept. |
| `episodic_cleanup_interval_sec` | double | `21600.0` | Period (s) of the episodic cleanup timer (also run once on activation); `<= 0` disables it. |

The full-robot launch files only override `db_path`, `similarity_threshold`,
`llm_url`, `llm_model` and `bearer_token`.

Fixed in code (not parameters): voice embedding dimension `192` (ECAPA-TDNN);
voice gallery max `10` entries per person, refresh age `7` days; voice-gallery
quality filter `0.40`; photo gallery limit on merge `30`.

**Topics**

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/social_context` | `std_msgs/String` (JSON) | subscribe (depth 10) | From `identity_manager`. Reads `person_present`, `name`, `state` (`IDLE`→`idle`, `INTERACTING`/`INTRODUCING`→`conversation`, anything else → `idle`) and updates working memory (people nearby, facing, mode). `emotion` is read but unused. |
| `/conversation_end` | `std_msgs/String` (JSON) | subscribe (depth 10) | Payload `{"transcript": "<text>", "participants": ["<name>", ...]}`. An empty transcript is ignored. Otherwise a background thread runs `MemoryManager.after_conversation` (summary → episode → optional fact extraction), republishes `/memory/context`, and starts reminder extraction. |
| `/robot_sleep` | `std_msgs/Bool` | subscribe (`RELIABLE`, `TRANSIENT_LOCAL`, depth 1) | Sets working-memory mode to `sleep` on `True` (also clears people nearby and facing) and back to `idle` on `False`. While asleep, `/social_context` does not overwrite the mode. |
| `/memory/context` | `std_msgs/String` | publish (lifecycle publisher, depth 10) | Plain text: `== Текущий момент ==` + working memory, and `== Последние события ==` + the last 5 episodes if any. Published every `context_publish_rate` seconds, right after activation and after each processed dialogue. |
| `/telegram_push` | `std_msgs/String` (JSON) | publish (lifecycle publisher, depth 10) | `{"text": "⏰ <b>Напоминание:</b> ...", "parse_mode": "HTML"}` for due reminders of `telegram_reminder_person_id` (consumed by `telegram_bridge_node`). |

**Service**

| Service | Type |
|---|---|
| `/memory/query` | `inmoov_msgs/srv/MemoryQuery` |

`MemoryQuery.srv`:

```
string request_json
---
string response_json
bool success
```

Request: a JSON object with an `op` field selecting the operation plus its
arguments. The response is a JSON object (`ensure_ascii=False`);
`success = ('error' not in result)`. An unknown `op` returns
`{"error": "Unknown op: <op>"}`. Any exception (missing key, invalid JSON, ...)
is caught and returned as `{"error": "<exception text>"}` with `success=false`.

Example:

```bash
ros2 service call /memory/query inmoov_msgs/srv/MemoryQuery \
  "request_json: '{\"op\": \"get_recent_episodes\", \"limit\": 3}'"
```

Supported operations (arguments marked `?` are optional; embeddings are JSON
arrays of floats and are L2-normalised by the node):

*Face / person (social memory)*

| `op` | Request fields | Response |
|---|---|---|
| `lookup_person` | `embedding` | Best match over the per-person photo galleries (falls back to `persons.embedding` if the gallery cache is empty). `>= similarity_threshold`: `{person_id, name, similarity, confidence: "high"}`; `>= uncertain_threshold`: `{person_id: null, similarity, confidence: "uncertain", best_candidate_id, best_candidate_name}`; else `{person_id: null, similarity, confidence: "unknown"}`. |
| `lookup_by_name` | `name` | `{person_id, name}` (case-insensitive exact match) or `{person_id: null}`. |
| `save_person` | `name`, `embedding` | `{person_id, name}` — inserts a new person. |
| `update_embedding` | `person_id`, `embedding`, `alpha?` (0.3) | Blends the stored mean embedding: `(1-alpha)*old + alpha*new`. `{updated, person_id}`. |
| `get_context` | `person_id` | `{person_id, name, first_seen, last_seen, meet_count, gallery_count, notes}`. |
| `update_seen` | `person_id` | Sets `last_seen`, increments `meet_count`. `{updated: true}`. |
| `set_note` | `person_id`, `key`, `value` | Upsert into `person_notes`. `{saved: true}`. |
| `verify_person_claim` | `person_id`, `face_embedding?`, `voice_embedding?` | `{face_sim, voice_sim, has_voice_gallery}` (`0.0` when no data). Face: max over the gallery; voice: cosine to the gallery centroid (or the single voice embedding). |
| `merge_persons` | `from_id`, `to_id`, `check_similarity?` (false) | Moves photos (best quality first, max 30 total), voice entries (newest first, up to 10), notes (existing keys of the target win) and the meeting counter into `to_id`, deletes `from_id`, its gallery photos that do not fit, and its photo directory, and renames the participant in the episodic DB. With `check_similarity`, mean-gallery cosine `< 0.35` aborts with `{merged: false, reason: "similarity_too_low", similarity, message}`. Success: `{merged: true, from_id, to_id, from_name, to_name}`. |

*Face gallery*

| `op` | Request fields | Response |
|---|---|---|
| `gallery_add` | `person_id`, `photo_path`, `embedding`, `quality?` (1.0), `source?` (`auto`) | Rejects near-duplicates (`max_sim > 0.95` → `{added: false, reason: "duplicate"}`) and, once the person has `>= 3` photos, embeddings with `max_sim < uncertain_threshold` (`reason: "embedding_mismatch"`). Success: `{added: true, gallery_count}`. |
| `gallery_remove` | `photo_path` | `{removed: true, person_id}` or `{removed: false, reason: "not_found"}`. |
| `gallery_rebuild_embedding` | `person_id` | Sets `persons.embedding` to the normalised mean of the person's gallery. `{rebuilt: true, person_id, gallery_count}` or `{rebuilt: false, reason: "no_gallery_photos"}`. |
| `gallery_list` | `person_id?` | `{photos: [...]}` — rows `[photo_path, quality, source, created_at]` for one person, or with a leading `person_id` for all. |
| `reload_gallery` | – | Reloads the in-RAM face-gallery cache from SQLite. `{reloaded: true, persons, total}`. |

*Voice*

| `op` | Request fields | Response |
|---|---|---|
| `get_voice_embedding` | `person_id` | `{person_id, embedding, has_voice}` (single legacy embedding). |
| `save_voice_embedding` | `person_id`, `embedding` | Stores `persons.voice_embedding`. `{saved: true, person_id}`. |
| `update_voice_embedding` | `person_id`, `embedding`, `alpha?` (0.3) | Blends into the single embedding (creates it if absent). |
| `lookup_by_voice` | `embedding`, `high_threshold?` (0.72), `uncertain_threshold?` (0.58) | Compares to each person's normalised gallery centroid (or the single embedding). Same `high` / `uncertain` / `unknown` structure as `lookup_person`. |
| `get_voice_gallery` | `person_id` | `{person_id, has_voice, gallery: [{embedding, timestamp}]}`. |
| `add_voice_to_gallery` | `person_id`, `embedding`, `timestamp?` (now) | Zero-norm → `{added: false, reason: "zero_norm"}`. If the gallery is non-empty and the centroid similarity `< 0.40` → `reason: "quality_too_low"`. `< 10` entries → added. Full: replaces the oldest entry only if it is `>= 7` days old, otherwise `reason: "gallery_full_and_fresh"`. |

*Robot knowledge*

| `op` | Request fields | Response |
|---|---|---|
| `get_knowledge` | `key?` | `{value}` for one key or `{knowledge: {key: value}}` for all. |
| `set_knowledge` | `key`, `value` | Upsert into `robot_knowledge`. `{saved: true}`. |

*Three-layer memory*

| `op` | Request fields | Response |
|---|---|---|
| `search_semantic` | `query`, `category?`, `limit?` (5) | `{facts: [{subject, predicate, value, category, score, source}]}`. |
| `save_semantic_fact` | `subject`, `predicate`, `value`, `category?` (`preference`), `confidence?` (1.0), `source?` (`conversation`) | `{fact_id, saved: true}` (`fact_id` is `-1` if a field was empty). |
| `search_episodes` | `keyword`, `date?` (`YYYY-MM-DD`), `limit?` (10) | `{episodes: [...]}` (LIKE search in `summary` and `raw_text`). |
| `get_recent_episodes` | `limit?` (5), `date?` | `{episodes: [...]}`, newest first. |
| `save_episode` | `summary`, `raw_text?`, `participants?`, `location?` (current room), `importance?` (0.3), `type?` (`conversation`), `emotion_tag?` (`neutral`) | `{episode_id, saved: true}`. |
| `get_working_memory` | – | `{working_memory: {...}}`. |
| `update_working_memory` | `location?` `{room, landmark, coordinates}`, `robot_state?` `{mode, battery, current_task, facing}`, `environment?` `{people, noise, lighting}` | `{updated: true}`. |
| `get_memory_context` | `limit?` (5) | `{context: "<text>"}` — same text as `/memory/context`. |

*Reminders* (see [Reminder system](#reminder-system-and-openhab-alerts))

| `op` | Request fields | Response |
|---|---|---|
| `add_reminder` | `person_id`, `message`, `person_name?`, `trigger_date?`, `trigger_time?`, `source?` (`manual`) | `{reminder_id, saved: true}` (an exact undelivered duplicate returns the existing id). Missing `person_id`/`message` → error. |
| `get_due_reminders` | `person_id`, `today?`, `now_time?`, `default_time?` | `{reminders: [...]}`. |
| `mark_reminder_delivered` | `reminder_id` | `{marked: true, reminder_id}`. |
| `delete_reminder` | `reminder_id` | `{deleted: true, reminder_id}`. |
| `confirm_reminders` | `person_id` | Deletes all delivered (`delivered=1`) reminders of that person. `{confirmed: <count>}`. |
| `list_reminders` | `person_id?` | `{reminders: [...]}` for one person or everybody. |

Some validation errors of the reminder ops are returned in Russian (e.g.
`person_id обязателен`); the merge similarity message is Russian too, because
it may be read back to the user by the LLM.

**Known issues / TODOs (from code and cross-checks)**

- All ops run on the service callback of the node's default (single-threaded)
  executor; LLM-based work is moved to background threads, but a slow
  `merge_persons` or ChromaDB query blocks other requests.
- `_gallery_add` / `_gallery_remove` keep the RAM cache in sync, but
  `persons.embedding` (the mean) is only recomputed through
  `gallery_rebuild_embedding`.

### `MemoryManager` (`inmoov_memory/memory_manager.py`)

Constructor: `MemoryManager(db_path="memory.db", chroma_path="./chroma",
llm_url=..., llm_model="qwen3.8-27b", bearer_token="", semantic_db_path=None)`.
`memory_node` passes `db_path=episodic_db_path`. If `semantic_db_path` is not
given the semantic facts share the episodic DB file.

- `after_conversation(dialogue_text, participants)`: LLM summary (1–2
  sentences; on failure the first 200 characters of the dialogue) → heuristic
  importance → `EpisodicMemory.save` → if `importance >= 0.6`, LLM fact
  extraction (`[{"subject","predicate","value","category"}]`) →
  `SemanticMemory.save_fact` for each valid fact.
- Importance heuristic (`_score_importance`): `min(0.3 + 0.15 * hits, 1.0)`
  where `hits` counts occurrences of a fixed keyword list (Russian and English
  words such as "люблю", "запомни", "зовут", "prefer", "remember") in the
  dialogue plus summary. Two keyword hits are enough to reach `0.6`.
- LLM call (`_call_llm`): one blocking, non-streaming `POST` with
  `temperature 0.1`, `max_tokens 512`, `timeout 30 s` and
  `chat_template_kwargs: {enable_thinking: false}` (a vLLM/Qwen extension); the
  bearer token is sent only if configured.
- The LLM context (`/memory/context`) is assembled by `memory_node` itself;
  `llm_node` in `inmoov_cognition` has its own prompt and tool set and reaches
  memory through `/memory/query`.
- `strip_code_fence()` removes a surrounding markdown fence from LLM answers
  before JSON parsing (shared with `memory_node`).

### Memory layers

**`WorkingMemory`** — RAM dict, recomputed time block on every `to_text()`.
Fields: `time` (timestamp, time of day, weekday, season, ... in English and
Russian), `location` (`room`, `coordinates`, `landmark`), `robot_state`
(`mode`: `idle | conversation | sleep | navigation | task`, `battery`,
`current_task`, `facing`), `environment` (`people_present`, `ambient_noise`,
`lighting`). `to_text()` renders a Russian multi-line block for the LLM. Only
`mode`, `people_present`, `facing` are updated by ROS topics; `location`,
`battery`, `current_task`, `noise`, `lighting` change only through
`update_working_memory`.

**`EpisodicMemory`** — table `episodes`:

```
id INTEGER PK AUTOINCREMENT, timestamp TEXT (ISO, seconds), date TEXT (YYYY-MM-DD),
type TEXT DEFAULT 'conversation', participants TEXT DEFAULT '[]' (JSON array of names),
summary TEXT NOT NULL, raw_text TEXT, location TEXT, emotion_tag TEXT DEFAULT 'neutral',
importance REAL DEFAULT 0.3, migrated INTEGER DEFAULT 0
```

Indexes `idx_ep_date(date)` and `idx_ep_importance(importance)`. The
sliding window is enforced by `memory_node`: every
`episodic_cleanup_interval_sec` (and once on activation) `cleanup()` deletes
episodes older than `episodic_retention_days` with
`importance <= episodic_cleanup_max_importance`. Important episodes are kept;
when `after_conversation` extracts facts from an episode (importance `>= 0.6`)
the episode is marked `migrated = 1`.

**`SemanticMemory`** — table `facts`:

```
id INTEGER PK AUTOINCREMENT, category TEXT (person | preference | event | rule),
subject, predicate, value TEXT NOT NULL, confidence REAL DEFAULT 1.0,
source TEXT DEFAULT 'conversation', created_at, updated_at TEXT, UNIQUE(subject, predicate)
```

A fact is unique per `(subject, predicate)`; saving again updates value,
confidence, source, category and `updated_at`. In parallel every fact is
upserted into a ChromaDB collection `robot_memory` (cosine space) with the
document text `"<subject> — <predicate>: <value>"` and id
`md5("<subject>::<predicate>")`. `search()` queries ChromaDB first
(`score = 1 - distance`, source `vector`), then supplements with a SQLite
`LIKE` search (source `sqlite`) if fewer than `limit` results were found.
If `chromadb` is not installed or fails to initialise, only SQLite is used
(a warning is logged). `delete_fact` removes both the SQLite row and the
ChromaDB entry.

## Reminder system and openhab_alerts

### Reminders (`ReminderDB`)

SQLite table `reminders(id, person_id, person_name, trigger_date, trigger_time,
message, source, delivered, created_at)`; `trigger_time` is added by a
migration on older DBs. Semantics:

- `trigger_date IS NULL` — shown at the next meeting (and every one after, until
  confirmed).
- `trigger_date = YYYY-MM-DD` — due from that date. `trigger_time = HH:MM`
  restricts it to that time on that day; without it `default_time`
  (`reminder_default_time`, `07:00`) is used.
- Lifecycle: created (`delivered=0`) → shown (`mark_delivered`, `delivered=1`) →
  confirmed by the user (`confirm_reminders` deletes them).
- `add_reminder` de-duplicates: the same person, date, time and normalised
  (stripped, lower-cased) message with `delivered=0` returns the existing id.
- `get_due` order: undated first, then by date and time.

Producers/consumers in `memory_node`:

- **Manual / LLM-created**: `add_reminder` op (`source` defaults to `manual`).
- **Auto-extracted**: after each dialogue, if the first participant is a known
  person and the transcript contains one of the Russian keywords in
  `_REMINDER_KEYWORDS` (birthday, concert, exam, month names, ...), the LLM is
  asked for future dated events as JSON. Each entry with a future
  `remind_date` and a non-empty `message` is stored with `source='auto'`.
  Entries whose `remind_date` is `<= today` are skipped; entries with an
  empty `remind_date` are stored with `trigger_date=NULL` (next meeting).
- **Telegram push**: a timer (`telegram_reminder_check_sec`) publishes due
  reminders of `telegram_reminder_person_id` to `/telegram_push` and marks them
  delivered, so each is sent once.
- **Meeting-time delivery**: the `get_due_reminders` / `mark_reminder_delivered`
  / `confirm_reminders` ops and the "shown at a meeting" lifecycle exist, but no
  other package in this workspace calls them (a search of `src/` finds no
  client); in the current workspace the only automatic consumer is the
  Telegram timer above.

### `openhab_alerts.py`

Pure functions, no ROS and no I/O. They are imported by `openhab_bridge_node`
(package `inmoov_cognition`), which watches OpenHAB items and publishes a push
notification to `/telegram_push` (rate limit 15 minutes per sensor) when
`evaluate_threshold` reports a violation. Environment alerts are **not** stored
in the reminder DB.

- `classify_sensor(item_name, item_type)` → one of `battery_low`, `battery`,
  `temp`, `humidity`, `co2`, `voc`, `radon_st`, `radon_lt`, or `None`. Rules:
  ignores `Color` items and names containing `setpoint`/`targettemp`; battery
  items are recognised for all sensors (`*_batterylow` Switch, `*_battery`
  Number); environmental items are skipped when the name prefix (text before the
  first `_`) is `EntranceOutside` or `Terrace`.
  `temp` = `Number:Temperature` ending in `_temp`; `humidity` = name contains
  `humidity`; `co2` = contains `_co2`; `voc` = contains `_voc`; `radon_st` /
  `radon_lt` = ends with `_radon_st` / `_radon_lt` (`Number` or
  `Number:Dimensionless`).
- `parse_value(state_str)` → first number in the OpenHAB state string
  (`'23.5 °C'`, `'84'`), or `None` for empty/`NULL`/`UNDEF`/`None`/`-`.
- `evaluate_threshold(item_name, sensor_type, value)` → `(True, message)` or
  `(False, '')`. Message text is Russian and the room prefix is mapped to a
  Russian room name (`Kitchen` → «Кухня», ... ; unknown prefixes are used as is).

| Sensor | Alert when | Constant |
|---|---|---|
| `battery` | `< 10` % (message rounds the level up to the next 5 %: 8 % → `<10%`, 4 % → `<5%`) | `BATTERY_MIN` |
| `temp` | `> 27` °C or `< 16` °C | `TEMP_MAX`, `TEMP_MIN` |
| `humidity` | `> 70` % or `< 30` % | `HUMIDITY_MAX`, `HUMIDITY_MIN` |
| `co2` | `> 1200` ppm | `CO2_MAX` |
| `voc` | `> 300` ppb | `VOC_MAX` |
| `radon_st` | `> 200` Bq/m³ | `RADON_ST_MAX` |
| `radon_lt` | `> 100` Bq/m³ (WHO action level) | `RADON_LT_MAX` |

`battery_low` (a Switch) is classified but `evaluate_threshold` produces no
message for it. Phone battery items named `Phone_<name>_battery` are shown as
"телефон <name>".

## Storage

All paths default to the author's home directory (`/home/artur/...`) and are
plain parameters; nothing is created under the ROS install prefix.

| What | Default path | Tables / content |
|---|---|---|
| Social DB (`db_path`) | `/home/artur/inmoov_memory.db` | `persons` (`embedding` BLOB = float32 face mean, plus `voice_embedding` BLOB and `telegram_id` INTEGER added by migration), `person_gallery` (per-photo face embeddings, `quality`, `source`: `enroll`/`manual`/`auto`), `person_notes`, `robot_knowledge`, `voice_gallery` (`embedding` BLOB, `recorded_at` REAL) |
| Episodic DB (`episodic_db_path`) | `/home/artur/inmoov_episodic.db` | `episodes` |
| Semantic DB (`semantic_db_path`) | `/home/artur/inmoov_semantic.db` | `facts` |
| ChromaDB (`chroma_path`) | `/home/artur/inmoov_chroma` | collection `robot_memory` |
| Reminder DB (`reminder_db_path`) | `/home/artur/inmoov_reminders.db` | `reminders` |
| Face photo gallery (not managed here) | `/home/artur/inmoov_faces/persons/<id>_<name>/` | JPG/PNG photos; `gallery_dir` parameter (used by `merge_persons`, must match `face_gallery_node`'s `gallery_dir`), default `--gallery` of `rebuild_gallery.py` |

Embeddings: face embeddings are InsightFace `buffalo_l` vectors (512-d, stored
as float32, normalised on load); voice embeddings are 192-d ECAPA-TDNN vectors
(vectors of any other length are skipped). The ChromaDB text embedding model is
**not configured** in this package: `SemanticMemory` calls
`get_or_create_collection` without an `embedding_function`, so ChromaDB's
default (`all-MiniLM-L6-v2` via ONNX) is used; its weights are downloaded by
ChromaDB on first use (internet access or a pre-populated ChromaDB cache is
required).

All databases are opened with SQLite defaults (`sqlite3.connect`), one
connection per operation for episodic/semantic memory and one shared
connection guarded by a `threading.Lock` for the social and reminder DBs.

## `scripts/rebuild_gallery.py`

Source: [`scripts/rebuild_gallery.py`](scripts/rebuild_gallery.py). Not
installed by `setup.py` (there is no entry point); run it directly.

Recomputes the `person_gallery` embeddings from the photos on disk. Use it
after manually deleting bad photos, moving photos between people, or adding
photos by hand.

```bash
python3 src/inmoov_memory/scripts/rebuild_gallery.py \
    [--db /path/to/inmoov_memory.db] [--gallery /path/to/inmoov_faces]

# afterwards, tell the running node to reload its cache
ros2 service call /memory/query inmoov_msgs/srv/MemoryQuery \
  "request_json: '{op: reload_gallery}'"
```

| Argument | Default | Meaning |
|---|---|---|
| `--db` | `/home/artur/inmoov_memory.db` | Social SQLite DB. |
| `--gallery` | `/home/artur/inmoov_faces` | Gallery root; photos are read from `<gallery>/persons/<id>_<name>/*.jpg\|*.png`. |

What it does: **deletes all rows of `person_gallery`**, then for every
directory named `<id>_<name>` whose `id` exists in `persons`, runs InsightFace
`buffalo_l` (`CUDAExecutionProvider` with CPU fallback, `det_size=640x640`) on
each photo, takes the face with the largest bounding box, and inserts its
normalised embedding with `quality=1.0` and `source` derived from the file
name (`enroll` / `manual` in the name, else `auto`). Photos without a detected
face are skipped and counted. Output is a progress line per person
(`+` added, `.` skipped).

Notes: it does not touch `persons.embedding` (call `gallery_rebuild_embedding`
per person if you need the mean refreshed), nor the voice gallery. The
`DELETE` of the old gallery and all inserts run in a single transaction, so an
interruption (Ctrl+C, crash) rolls back to the old gallery. At the end it
prints the `reload_gallery` command shown above. Requires `insightface`, `opencv-python`, `numpy`.

## Launch

The package has **no launch file of its own**. `memory_node` is started by:

- `ros2 launch inmoov_bringup inmoov.launch.py` — as a `LifecycleNode` (Tier 0,
  `respawn=True`), configured/activated by `lifecycle_manager`; `db_path`,
  `llm_url`, `llm_model`, `bearer_token` come from the launch arguments
  `memory_db_path`, `llm_url`, `llm_model`, `llm_bearer_token`
  (`llm_bearer_token` defaults to the `VLLM_BEARER_TOKEN` environment variable).

Manual start:

```bash
ros2 run inmoov_memory memory_node --ros-args \
  -p db_path:=$HOME/inmoov_memory.db \
  -p episodic_db_path:=$HOME/inmoov_episodic.db \
  -p semantic_db_path:=$HOME/inmoov_semantic.db \
  -p chroma_path:=$HOME/inmoov_chroma \
  -p reminder_db_path:=$HOME/inmoov_reminders.db \
  -p llm_url:=http://<llm-host>:<port>/v1/chat/completions \
  -p llm_model:=<model> -p bearer_token:=<token>

ros2 lifecycle set /memory_node configure
ros2 lifecycle set /memory_node activate
```

## Requirements / Setup

- ROS2 Jazzy, `ament_python`. `package.xml` depends on `rclpy` and
  `inmoov_msgs` (which provides `srv/MemoryQuery`). `std_msgs` is imported and
  comes with ROS.
- Python packages imported by the code and **not** declared in `package.xml`:
  `numpy` (required); `chromadb` (optional — without it semantic search falls
  back to SQLite `LIKE`); `insightface`, `opencv-python` (only for
  `scripts/rebuild_gallery.py`). The rest (`sqlite3`, `urllib`, `json`, ...) is
  the standard library.
- An OpenAI-compatible chat-completions server (e.g. vLLM) reachable at
  `llm_url` for summaries, fact extraction and reminder extraction. Without it
  the episode summary falls back to the first 200 characters and no facts or
  reminders are extracted (errors are logged as warnings).
- Writable locations for all five paths listed under [Storage](#storage); the
  files are created on first configure. On another machine pass explicit paths
  as parameters (defaults point into `/home/artur`).
- Build: `colcon build --packages-select inmoov_memory` (no
  `--symlink-install` is used in this workspace).

## Known issues to verify

- Shipped defaults contain author-specific values: `/home/artur/...` paths, the
  LAN address `192.168.10.118:18020` for `llm_url`, `telegram_reminder_person_id=5`
  and `qwen3.8-27b` as model name — override them on your installation.
- No automated tests are shipped for this package (`package.xml` lists only the
  ament lint test dependencies).

## License

GNU General Public License v3.0 (GPL-3.0-only) — see the repository root
[`LICENSE`](../../LICENSE).

Author: Artur Fedjukevits. Assisted by: Claude Code (Anthropic).
