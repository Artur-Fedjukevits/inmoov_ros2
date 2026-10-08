# Live tests — pending

Checks that need a person, a voice or eyes on the physical robot: they
cannot be covered by `colcon test`/CI. Each item names the commit(s) to
look at if it fails. Tick an item (`[x]`) when it passes; if it fails, note
the date and what happened under the item. Remove a section once all of it
has passed.

"pre-`cadf411`" means the feature predates the first commit of this
repository.

Russian phrases in «» are said to the robot as written.

## Firmware and servo link (`Arduino/`, `inmoov_control`)

Host-loss failsafe — `baae523`:

- [ ] Left Mega, scripted host loss: the head returns slowly to 90°, `thumb_L`
      reaches a literal 0° (not 50°); after a fast `SET_SPEEDS` gesture the
      return is still slow (~2 s)
- [ ] The same failsafe test on the right Mega (arm/eyes) — so far only the
      left one was tested
- [ ] Real stop mid-gesture: `systemctl --user stop inmoov` while arms/head
      are off rest → slow return to rest

False failsafe in normal use — `7a866a0`, `c6f1187`, `72d9762`:

- [ ] No "entered host-loss FAILSAFE" during a normal day of use. If one
      still appears, its log line says whether the host stalled (TX/RX gaps)
      or not

Servo calibration tools with the full stack running — `7a7b9c1`:

- [ ] `servo_calibration_gui` (`src/inmoov_control/test/`): sliders move the servos (priority 90 overrides
      the tracker/BT); `face_expression_calibrator` too

## Joint arbitration (`inmoov_control`, `inmoov_vision`, `inmoov_cognition`)

Per-joint priority + lease — `7a7b9c1`:

- [ ] «посмотри налево» while the tracker follows a face: the head turns
      without the old REST twitch, returns after ~6.5 s, the tracker resumes
      (watch `/arduino_left/joint_owners` meanwhile)
- [ ] Emotional reply (happy/surprise voice style) while tracking: eyes stay
      on the face, the jaw keeps lip-syncing, eyebrows/cheeks still show the
      expression; no eye jerk to centre after the speech
- [ ] Head tracking after a PIR scan no longer inherits the fast scan speed
      (default step 1 for `rothead`) — judge whether tracking now feels too
      slow; if so, raise the velocity in the tracker commands
- [ ] Blinking still works in idle

## Voice (`inmoov_voice`)

Deactivation hygiene — `62181da`:

- [ ] Put the robot to sleep, then in one breath «Эй Лёня, который час?» →
      the command is answered (`voice_detector` stays ACTIVE in SLEEP; the
      pre-roll VAD runs lazily on activation)
- [ ] Normal dialogue: after the answer the robot keeps listening without a
      wake word (relisten is now one ROS timer instead of four
      `threading.Timer`s)

Speaker verification — `b67ab0d`, `d344008`, `f524833`:

- [ ] Foreign voice: TV on / another person talking while you speak to the
      robot → their phrases are dropped (log: "SV: phrase rejected — not the
      current speaker")
- [ ] Tune `sv_threshold` (0.35, provisional) from the WAVs in
      `~/inmoov_sv_debug`, then switch `sv_debug_dir` off
- [ ] Introducing a new person: the robot asks the name and accepts the
      answer (no endless name loop)

Audio under load — `ca20843` (systemd unit: `CPUAffinity=0-13` instead of
`CPUQuota`):

- [ ] No TTS crackle during long answers. If it is frequent, check
      `cpu.stat` throttling of the unit's cgroup

STT switched to `gigaam_stt_node` (GigaAM-v3 e2e-rnnt, Parakeet only as the
fallback for English):

- [ ] Normal Russian dialogue: transcripts in the log (`gigaam recognized in …`)
      are right, with punctuation; nothing dropped from quiet phrases
- [ ] A phrase in English → `parakeet recognized in …`, answered sensibly
- [ ] «Леонид, который час?» from a distance, not looking at the robot →
      answered (the full name is in `llm_node._ROBOT_NAMES`)

Voice emotion fusion — pre-`cadf411`:

- [ ] Angry face + neutral voice: does the robot react to the emotion at
      all? Decide whether the fusion weights are right

## Dialogue and behaviour (`inmoov_cognition`)

Addressee gate v2 — `cca141a`, `b9a4034`:

- [ ] Phone call next to the robot, not looking at it → it doesn't answer
      (log "LLM: speech not addressed to the robot … — skipping")
- [ ] Walk away mid-dialogue and keep talking → no answers once you are gone
- [ ] Looking at the robot without saying its name → answered
- [ ] Looking away, name anywhere in the phrase («который час, Лёня?») →
      answered; also from a distance, where Parakeet writes the name as
      Леона/Люня/Лення/…
- [ ] Wake word with no face in view → answered for 30 s (`wake_grace_sec`);
      after that the name is required
- [ ] Looking at the robot but talking to someone else in the room → the LLM
      answers `[ignore]`: the robot stays silent (no «я не понял»)
- [ ] Telegram requests are never blocked by the gate

Dialogue guards — `0eb4e92`:

- [ ] Ask something slow (web search / «какая погода») and immediately say a
      second phrase while it thinks → the first answer is spoken fully, then
      the second is answered (log: "command queued" → "Running queued
      command")
- [ ] From Telegram: «поверни голову налево» → the robot does NOT move and
      the bot says it can't; «включи свет» still works
- [ ] After a few tool calls `~/.ros/inmoov_tool_audit.jsonl` has lines with
      source `voice`/`telegram`

Behaviour tree — `62181da`, `b2e5b97`:

- [ ] «посмотри налево», then say goodbye / go to sleep during the 5 s head
      override → the head doesn't jerk after deactivation; otherwise the
      tracker resumes normally
- [ ] No PIR scan while a dialogue is active, even when the face is lost for
      a moment

Face search during a dialogue — `50f038b` (fixes from the 2026-09-01 live
test, never re-tested):

- [ ] A noisy face track near the frame edge: the head keeps following, no
      ~10 s freeze
- [ ] Brief face loss (< 10 s) at the frame edge mid-dialogue: the head stays
      on the person, no trip to REST
- [ ] Face lost during the dialogue: the head searches by the voice hint
      («я справа») or the sound direction; after two failed attempts the
      robot asks where you are

Reminders — `65a14c1`:

- [ ] Reminder via Telegram (set «через минуту» by voice or `/ask`): it
      arrives in Telegram; the `memory_node` log shows "waiting for ACK" →
      "delivered ✓"

Identity verification — pre-`cadf411`, `merge_persons` hardened in
`17011f5`:

- [ ] Known face → greeted by name, no introduction
      (RECOGNIZING → INTERACTING)
- [ ] Face not recognised (hand over the face / dark), recognised by voice →
      «Прости, Артур! Я тебя не узнал по лицу, но узнал по голосу.»
- [ ] Voice in the uncertain zone → «Вы случайно не Артур?» → «да» → greeted
- [ ] Unrecognised face, answer to «Как тебя зовут?» with «Лёня, это я Артур»
      → verified by face/voice, no new person created
- [ ] No voice in the gallery + uncertain face → the robot asks for
      confirmation; «да» → INTERACTING, «нет» → asks the name again
- [ ] A different person calls himself «Артур» → «У меня уже есть Артур, но
      вы на него не похожи…» → a second, separate person in the DB
- [ ] New name → «Вас зовут Николай? Правильно понял?» → «да» → enrolled;
      «нет» → asks again
- [ ] «Лёня, объедини Тест и Артур» with a test duplicate → `merge_persons`
      merges or rejects; the duplicate is gone from the DB and the episodic
      DB is updated
- [ ] Leave the frame in the middle of the name confirmation → the pending
      state is reset; next session as «Артур» goes the normal way

Multi-person — pre-`cadf411`:

- [ ] Two people in view → the robot stays with the first one, doesn't
      switch
- [ ] The first leaves → after 30 s the second becomes the interlocutor
- [ ] With two people the head follows only the interlocutor
- [ ] The gallery doesn't get the other person's face: new photos land only
      in the interlocutor's `persons/{id}_{name}/` folder
- [ ] The voice anchor is set once on entering INTERACTING and doesn't change
      during the dialogue

OAK-D priority over face detection — pre-`cadf411`:

- [ ] Leave the room → IDLE after 30 s, even if the eye cameras "see"
      something
- [ ] Cover the OAK-D with a hand → no introduction for an unknown face
- [ ] A known person falsely put into INTRODUCING says something → the
      voiceprint cancels the introduction

## Vision (`inmoov_vision`)

MJPEG camera path — `7cddc21`:

- [ ] Face recognition / head tracking with a real person: the JPEG now comes
      straight from the camera (~52 KB, camera quality) — recognition
      similarity and tracking look as before; gallery photos look fine
- [ ] Emotion recognition still reacts; `look_and_describe` / Telegram photo
      still get images
- [ ] Cover one eye camera / unplug it: the other eye's frames are mirrored
      with the `frame_id` of the real source; `face_detection` warns "frame
      NOT updating" if the remaining eye freezes

Dual-eye fallback — pre-`cadf411`:

- [ ] Left eye covered: all nodes (not only `identity_manager`) switch to the
      right eye within ~2 s

Face gallery — pre-`cadf411`:

- [ ] `right_emb` differs from `left_emb` → right-eye photos are no longer
      dropped by `gallery_add` as duplicates
- [ ] Rotation at the limit: exactly 1 left + 1 right photo added per
      session, the 2 oldest removed
- [ ] Embedding rebuilt after the person leaves (INTERACTING → IDLE) and on
      `/robot_sleep` True (log "Embedding recomputed: person_id=…")
- [ ] `gallery_remove`: the entry disappears from the DB and from
      `memory_node`'s gallery cache

## Open decisions (not tests)

- [ ] Telegram tool denylist default (`robot_control`, `look_direction`,
      `merge_persons`) — is this the wanted set? Change via the `llm_node`
      parameter `telegram_tool_denylist` (`0eb4e92`)
- [ ] `llm_node`'s weather tool computes an hourly forecast (`hourly`) but
      never returns it — return it to the LLM or delete it
