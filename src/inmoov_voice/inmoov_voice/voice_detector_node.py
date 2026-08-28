import collections
import time

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray, Bool, MultiArrayDimension, String
import numpy as np
import torch


class VoiceDetectorNode(LifecycleNode):
    def __init__(self):
        super().__init__('voice_detector_node')
        # Заглушки — заполняются в on_configure
        self.publisher_      = None
        self._voice_emb_pub  = None
        self._sleep_pub      = None
        self._tts_cancel_pub = None
        self.vad_model       = None
        self._sv_encoder     = None

    # ── Lifecycle: Phase 2 ─────────────────────────────────────────────────

    def _dp(self, name, default=None):
        """Безопасный declare_parameter: игнорирует повторное объявление при re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        # ── Параметры ─────────────────────────────────────────────────────
        self._dp('sample_rate',           16000)
        self._dp('vad_threshold',         0.4)
        self._dp('silence_duration_sec',  2.5)
        self._dp('min_phrase_sec',        0.3)
        self._dp('min_speech_sec',        1.0)
        self._dp('min_speech_sec_introducing', 0.4)
        self._dp('max_phrase_sec',        20.0)
        self._dp('no_speech_timeout_sec', 8.0)
        self._dp('pipeline_timeout_sec',  90.0)
        self._dp('speaker_verification',  True)
        self._dp('sv_threshold',          0.55)
        self._dp('sv_segment_sec',        1.0)

        self.rate              = self.get_parameter('sample_rate').value
        self.vad_threshold     = self.get_parameter('vad_threshold').value
        self.min_phrase_sec    = self.get_parameter('min_phrase_sec').value
        self.min_speech_sec    = self.get_parameter('min_speech_sec').value
        self.min_speech_sec_introducing = self.get_parameter('min_speech_sec_introducing').value
        self.max_phrase_sec    = self.get_parameter('max_phrase_sec').value
        self.no_speech_timeout = self.get_parameter('no_speech_timeout_sec').value
        self.pipeline_timeout  = self.get_parameter('pipeline_timeout_sec').value
        silence_duration_sec   = self.get_parameter('silence_duration_sec').value
        self._sv_enabled       = self.get_parameter('speaker_verification').value
        self._sv_threshold     = self.get_parameter('sv_threshold').value
        self._sv_seg_sec       = self.get_parameter('sv_segment_sec').value

        # ── Подписки ──────────────────────────────────────────────────────
        self.create_subscription(Bool,            'wake_detected',   self.wake_callback,          10)
        self.create_subscription(Bool,            'tts_speaking',    self._tts_speaking_callback, 10)
        self.create_subscription(Float32MultiArray, 'raw_audio',     self._audio_callback,        20)
        self.create_subscription(String,          'voice_command',   self._stt_done_callback,     10)
        self.create_subscription(Bool,            '/person_present', self._person_present_cb,     10)
        self.create_subscription(Bool,            '/introducing',    self._introducing_cb,         10)
        self.create_subscription(Bool,            '/go_idle',        self._go_idle_cb,            10)
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Bool,   '/robot_sleep',  self._robot_sleep_cb,  latched_qos)
        self.create_subscription(String, '/voice_anchor', self._voice_anchor_cb, 10)

        # ── Lifecycle Publishers ───────────────────────────────────────────
        self.publisher_      = self.create_lifecycle_publisher(Float32MultiArray, 'audio_to_whisper', 10)
        self._voice_emb_pub  = self.create_lifecycle_publisher(String, '/voice_embedding', 10)
        self._sleep_pub      = self.create_lifecycle_publisher(Bool, '/robot_sleep', latched_qos)
        self._tts_cancel_pub = self.create_lifecycle_publisher(Bool, '/tts_cancel_queue', 10)

        # ── Silero VAD ────────────────────────────────────────────────────
        self.get_logger().info('Загрузка Silero VAD...')
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
        )
        self.vad_model.eval()
        self.get_logger().info('Silero VAD загружен')

        # ── Speaker Verification (ECAPA-TDNN) ─────────────────────────────
        self._sv_encoder = None
        if self._sv_enabled:
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
                self.get_logger().info('Загрузка ECAPA-TDNN (spkrec-ecapa-voxceleb)...')
                self._sv_encoder = EncoderClassifier.from_hparams(
                    source='speechbrain/spkrec-ecapa-voxceleb',
                    savedir='/home/artur/.cache/speechbrain/spkrec-ecapa-voxceleb',
                    run_opts={'device': 'cpu'},
                )
                self.get_logger().info(
                    f'Speaker Verification включена (ECAPA-TDNN, 192D, '
                    f'threshold={self._sv_threshold}, segment={self._sv_seg_sec}s)')
            except Exception as e:
                self._sv_enabled = False
                self.get_logger().warn(f'ECAPA-TDNN не загружен: {e}')

        # ── Буферы и состояние ────────────────────────────────────────────
        self._chunk_size        = None
        self._silence_threshold = None
        self._max_chunks        = None
        self._silence_secs      = silence_duration_sec
        self._pre_roll_buffer   = collections.deque()
        self._pre_roll_secs     = 1.5
        self._pre_roll_maxlen   = None
        self._onset_buf         = collections.deque()
        self._onset_secs        = 0.4
        self._onset_maxlen      = None
        self._sv_buf            = []
        self._sv_seg_samples    = 0
        self._sv_gallery        = []
        self._sv_gallery_times  = []
        self._sv_anchor_person_id = None  # чей якорь сейчас в _sv_gallery (None = живая сессия без анкора из БД)
        self._SV_GALLERY_MAX    = 10
        self._sv_last_gallery_add      = 0.0
        self._SV_GALLERY_ADD_INTERVAL  = 30.0
        self._introducing       = False
        self.is_active          = False
        self.audio_buffer       = []
        self.silence_counter    = 0
        self.speech_chunks      = 0
        self.activation_time    = 0.0
        self.tts_speaking       = False
        self._stt_sent_time     = 0.0
        self._sleeping          = False
        self._person_present    = None
        self._person_present_time  = 0.0
        self._person_last_seen     = 0.0
        self._person_present_grace = 120.0
        # Живой баг 2026-08-28: сразу после wake word, ДО первой успешной фразы
        # в этой сессии, _is_person_present() всегда False (_person_last_seen
        # ещё не обновлялся ни от /person_present, ни от успешной отправки в
        # STT) — если самая первая запись отбрасывается как слишком короткая
        # (например, VAD зацепил только хвост произнесения будильного слова),
        # авто-переактивация не срабатывает, и микрофон "умирает" до
        # следующего wake word, будто человек вообще ничего не сказал. Даём
        # отдельное окно грейса ПОСЛЕ wake word — не полагаемся только на
        # _is_person_present().
        self._last_wake_time          = 0.0
        self._POST_WAKE_LISTEN_GRACE_SEC = 15.0

        sv_status = 'вкл' if self._sv_enabled else 'выкл'
        self.get_logger().info(
            f'VAD нода готова. Жду wake word... '
            f'(vad_threshold={self.vad_threshold}, silence={silence_duration_sec}с, '
            f'max_phrase={self.max_phrase_sec}с, speaker_verification={sv_status})')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.publisher_.on_activate(state)
        self._voice_emb_pub.on_activate(state)
        self._sleep_pub.on_activate(state)
        self._tts_cancel_pub.on_activate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.is_active = False
        self.publisher_.on_deactivate(state)
        self._voice_emb_pub.on_deactivate(state)
        self._sleep_pub.on_deactivate(state)
        self._tts_cancel_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

    # ── Колбэки управления ────────────────────────────────────────────────

    def wake_callback(self, msg: Bool):
        if not msg.data:
            return

        if self._sleeping:
            wake_msg = Bool()
            wake_msg.data = False
            self._sleep_pub.publish(wake_msg)
            self.get_logger().info('Wake word во сне → публикую /robot_sleep False')

        if self.tts_speaking:
            cancel_msg = Bool()
            cancel_msg.data = True
            self._tts_cancel_pub.publish(cancel_msg)
            self.get_logger().info('Wake word во время TTS → отменяю TTS очередь')

        # Галерею сбрасываем только если нет активного собеседника.
        # В режиме INTERACTING (человек присутствует) галерея сохраняется между фразами.
        person_active = self._is_person_present()

        if self.is_active:
            self.get_logger().warn('Wake word во время записи — сбрасываю буфер, начинаю заново')
            self.audio_buffer    = []
            self.silence_counter = 0
            self.activation_time = time.time()
            self._last_wake_time = time.time()
            self._sv_buf.clear()
            if not person_active:
                self._sv_gallery.clear()
                self._sv_gallery_times.clear()
                self._sv_anchor_person_id = None
                self._sv_last_gallery_add = 0.0
            return

        speech_chunks = [a for a, c in self._pre_roll_buffer if c > self.vad_threshold]
        self._pre_roll_buffer.clear()
        self._onset_buf.clear()
        self.get_logger().info('Проснулся! Слушаю команду...')
        self.is_active       = True
        self.audio_buffer    = speech_chunks
        self.silence_counter = 0
        self.speech_chunks   = 0
        self.activation_time = time.time()
        self._last_wake_time = time.time()
        self._sv_buf.clear()
        if not person_active:
            self._sv_gallery.clear()
            self._sv_gallery_times.clear()
            self._sv_anchor_person_id = None
            self._sv_last_gallery_add = 0.0

    def _stt_done_callback(self, msg: String):
        if self._stt_sent_time <= 0.0:
            return
        self._stt_sent_time = 0.0
        if not msg.data:
            if self._is_person_present() and not self._sleeping:
                self.get_logger().info('STT: тишина — продолжаю слушать (человек рядом)')
                import threading
                threading.Timer(0.5, self._activate_after_tts).start()
            else:
                self.get_logger().info('STT: тишина — возвращаюсь к wake word')
        else:
            # STT вернул текст — LLM сейчас обрабатывает и должен запустить TTS.
            # Если LLM заблокировал фразу (gaze gate, /introducing, busy) — TTS не придёт.
            # Страховочный таймер: активируем через 6с если TTS так и не запустился.
            # Если TTS всё же пришёл раньше — _activate_after_tts проверит tts_speaking и не продублирует.
            if self._is_person_present() and not self._sleeping:
                import threading
                threading.Timer(6.0, self._activate_after_tts).start()

    def _go_idle_cb(self, msg: Bool):
        """Явное прощание от BT — немедленно прекращаем запись, сбрасываем grace period.

        В отличие от person_present=False (grace 120с), go_idle означает: человек
        попрощался и больше не ждёт ответа. Возвращаемся к wake word немедленно.
        """
        if not msg.data:
            return
        self._person_present   = False
        self._person_last_seen = 0.0   # сбрасываем grace period — не ждём 120с
        self._stt_sent_time    = 0.0
        if self.is_active:
            self.audio_buffer    = []
            self.silence_counter = 0
            self.speech_chunks   = 0
            self.is_active       = False
            self._sv_buf.clear()
        self._sv_gallery.clear()
        self._sv_gallery_times.clear()
        self._sv_anchor_person_id = None
        self._sv_last_gallery_add = 0.0
        self._introducing = False
        self.get_logger().info('go_idle: прекращаю запись, возвращаюсь к wake word')

    def _introducing_cb(self, msg: Bool):
        self._introducing = msg.data
        if not msg.data and self._stt_sent_time > 0.0:
            self.get_logger().info('INTRODUCING завершён — сбрасываю pipeline timeout')
            self._stt_sent_time = 0.0

    def _robot_sleep_cb(self, msg: Bool):
        self._sleeping = msg.data
        if msg.data:
            if self.is_active:
                self.audio_buffer    = []
                self.silence_counter = 0
                self.is_active       = False
                self._sv_buf.clear()
            self._stt_sent_time = 0.0
            self._sv_gallery.clear()
            self._sv_gallery_times.clear()
            self._sv_anchor_person_id = None
            self._sv_last_gallery_add = 0.0
            self._introducing = False
            self.get_logger().info('Спящий режим: авто-активация отключена')
        else:
            self.get_logger().info('Пробуждение: авто-активация восстановлена')

    def _person_present_cb(self, msg: Bool):
        now = time.time()
        self._person_present      = msg.data
        self._person_present_time = now
        if msg.data:
            self._person_last_seen = now

    def _activate_after_tts(self):
        if not self.is_active and not self.tts_speaking and not self._sleeping:
            speech_chunks = [a for a, c in self._pre_roll_buffer if c > self.vad_threshold]
            self._pre_roll_buffer.clear()
            self._onset_buf.clear()
            self.is_active       = True
            self.audio_buffer    = []
            self.silence_counter = 0
            self.speech_chunks   = 0
            self.activation_time = time.time()
            self._sv_buf.clear()
            # Галерея НЕ сбрасывается — живёт всю сессию. Сброс только при go_idle / robot_sleep.

            if speech_chunks:
                dur = len(speech_chunks) * (self._chunk_size or 512) / self.rate
                if self._sv_enabled and self._sv_encoder and self._sv_gallery:
                    audio_seg = np.concatenate(speech_chunks)
                    emb = self._sv_embed(audio_seg)
                    if emb is None:
                        self.audio_buffer = speech_chunks
                        self.speech_chunks = len(speech_chunks)
                        self.get_logger().info(
                            f'Активация + pre-roll: захвачено {dur:.2f}с речи (SV: слишком коротко)')
                    else:
                        sim = self._sv_sim(emb)
                        if sim >= self._sv_threshold:
                            self.audio_buffer = speech_chunks
                            self.speech_chunks = len(speech_chunks)
                            self.get_logger().info(
                                f'Активация + pre-roll: захвачено {dur:.2f}с речи '
                                f'(SV sim={sim:.2f}, галерея={len(self._sv_gallery)})')
                        else:
                            self.get_logger().warn(
                                f'Активация: pre-roll отброшен как чужой голос '
                                f'(sim={sim:.2f}, галерея={len(self._sv_gallery)})')
                else:
                    self.audio_buffer = speech_chunks
                    self.speech_chunks = len(speech_chunks)
                    self.get_logger().info(
                        f'Активация + pre-roll: захвачено {dur:.2f}с речи до активации')

    def _is_person_present(self) -> bool:
        now = time.time()
        if self._person_present is None:
            return True
        if (now - self._person_last_seen) < self._person_present_grace:
            return True
        return False

    def _should_keep_listening(self) -> bool:
        """_is_person_present() ИЛИ мы совсем недавно проснулись по wake word.

        Нужно отдельно от _is_person_present(): сразу после wake word, до
        первой успешно отправленной в STT фразы этой сессии, presence ещё
        не подтверждён (_person_last_seen не обновлялся). Если самая первая
        запись отброшена как слишком короткая (VAD зацепил только хвост
        произнесения будильного слова) — без этой проверки микрофон "умирал"
        насовсем, будто человек вообще не сказал ни слова после wake word."""
        if self._is_person_present():
            return True
        return (time.time() - self._last_wake_time) < self._POST_WAKE_LISTEN_GRACE_SEC

    def _tts_speaking_callback(self, msg: Bool):
        was_speaking  = self.tts_speaking
        self.tts_speaking = msg.data

        if msg.data:
            self._stt_sent_time = 0.0
            self._pre_roll_buffer.clear()
            if self.is_active:
                self.get_logger().info('TTS заговорил во время записи — сбрасываю буфер')
                self.audio_buffer    = []
                self.silence_counter = 0
                self.speech_chunks   = 0
                self.is_active       = False
                self._sv_buf.clear()
                # Галерея НЕ сбрасывается — следующая реплика фильтруется по той же галерее
        else:
            if was_speaking and not self.is_active:
                if self._sleeping:
                    self.get_logger().info('TTS закончил — спящий режим, авто-активация пропущена')
                elif self._is_person_present():
                    self.get_logger().info('TTS закончил — жду ответа пользователя...')
                    import threading
                    threading.Timer(1.2, self._activate_after_tts).start()
                else:
                    self.get_logger().info(
                        'TTS закончил — человека нет в кадре, авто-активация отключена')

    def _voice_anchor_cb(self, msg: String):
        """Загружаем голосовую галерею из БД (identity_manager → voice_detector).

        Получаем полный JSON: {person_id, name, gallery: [{embedding, timestamp}]}

        Если текущая живая галерея уже принадлежит ЭТОМУ ЖЕ person_id — не
        перезаписываем (живые записи сессии точнее старого снимка из БД).
        Но если анкор для ДРУГОГО человека (собеседник сменился, а живая
        галерея не была сброшена — напр. тихий IDLE без /go_idle в грейс-окне
        person_present) — заменяем гарантированно, иначе SV будет сверять
        новый голос со старым и отбрасывать его как чужой (инцидент 2026-08-25).
        """
        if not self._sv_enabled or self._sv_encoder is None:
            return
        try:
            import json as _json
            data = _json.loads(msg.data)
            anchor_person_id = data.get('person_id')
        except Exception as e:
            self.get_logger().warn(f'SV: ошибка парсинга голосовой галереи: {e}')
            return
        if self._sv_gallery:
            if anchor_person_id is not None and anchor_person_id == self._sv_anchor_person_id:
                return  # живая галерея уже принадлежит этому же человеку — не перезаписываем
            self.get_logger().info(
                f'SV: якорь сменился (было person_id={self._sv_anchor_person_id}, '
                f'стало {anchor_person_id}) — заменяю живую галерею ({len(self._sv_gallery)} записей)')
            self._sv_gallery.clear()
            self._sv_gallery_times.clear()
        self._sv_anchor_person_id = anchor_person_id
        try:
            gallery = data.get('gallery', [])
            if not gallery:
                # Fallback: legacy single-embedding формат
                emb_list = data.get('embedding')
                if emb_list:
                    emb = np.array(emb_list, dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-8:
                        emb /= norm
                        self._sv_gallery = [emb]
                        self._sv_gallery_times = [0.0]
                        self.get_logger().info('SV: галерея загружена (1 legacy запись)')
                return
            new_gallery, new_times = [], []
            for entry in gallery:
                emb = np.array(entry['embedding'], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm < 1e-8:
                    continue
                emb /= norm
                new_gallery.append(emb)
                new_times.append(float(entry.get('timestamp', 0.0)))
            if new_gallery:
                self._sv_gallery = new_gallery
                self._sv_gallery_times = new_times
                self.get_logger().info(
                    f'SV: галерея загружена из БД ({len(new_gallery)} записей)')
        except Exception as e:
            self.get_logger().warn(f'SV: ошибка парсинга голосовой галереи: {e}')

    # ── Speaker Verification ──────────────────────────────────────────────

    # ECAPA-TDNN требует минимум ~0.5с аудио (иначе conv padding > time_dim → RuntimeError)
    _SV_MIN_SAMPLES = 8000  # 0.5с @ 16kHz

    def _sv_embed(self, audio_seg: np.ndarray) -> 'np.ndarray | None':
        """L2-нормализованный 192-мерный ECAPA-TDNN эмбеддинг.
        Возвращает None если аудио короче _SV_MIN_SAMPLES.
        """
        if len(audio_seg) < self._SV_MIN_SAMPLES:
            return None
        # RMS-нормализация: ECAPA чувствителен к уровню входного сигнала.
        # Выравниваем до RMS=0.05 чтобы эмбеддинги были сопоставимы между сессиями.
        rms = float(np.sqrt(np.mean(audio_seg ** 2)))
        if rms > 1e-6:
            audio_seg = np.clip(audio_seg * (0.05 / rms), -1.0, 1.0)
        wav = torch.tensor(audio_seg).unsqueeze(0)
        wav_lens = torch.tensor([1.0])
        with torch.no_grad():
            emb = self._sv_encoder.encode_batch(wav, wav_lens)  # [1, 1, 192]
        emb = emb.squeeze().numpy().astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-8
        return emb

    def _sv_sim(self, emb: np.ndarray) -> float:
        """Максимальное косинусное сходство с голосовой галереей сессии.

        max вместо mean: галерея содержит записи из разных сессий (якорь из БД) — стale-записи
        при других акустических условиях тянут среднее вниз, скрывая совпадение с актуальными.
        """
        if not self._sv_gallery:
            return 0.0
        sims = [float(np.dot(g, emb)) for g in self._sv_gallery]
        self.get_logger().debug(
            f'SV sims [{len(sims)}]: {[f"{s:.2f}" for s in sorted(sims, reverse=True)]}')
        return float(np.max(sims))

    def _sv_add_to_gallery(self, emb: np.ndarray) -> bool:
        """Добавляет embedding в живую галерею сессии.

        Ограничения: не чаще 1 раза в _SV_GALLERY_ADD_INTERVAL секунд,
        максимум _SV_GALLERY_MAX записей (старая замещается новой).
        Публикует embedding в /voice_embedding → identity_manager → БД.
        """
        now = time.time()
        if (now - self._sv_last_gallery_add) < self._SV_GALLERY_ADD_INTERVAL:
            return False
        if len(self._sv_gallery) >= self._SV_GALLERY_MAX:
            self._sv_gallery.pop(0)
            self._sv_gallery_times.pop(0)
        self._sv_gallery.append(emb)
        self._sv_gallery_times.append(now)
        self._sv_last_gallery_add = now
        n = len(self._sv_gallery)
        self.get_logger().info(f'SV: запись {n}/{self._SV_GALLERY_MAX} добавлена в галерею')
        self._publish_voice_emb(emb, now)
        return True

    def _publish_voice_emb(self, emb: np.ndarray, ts: float = None):
        import json as _json
        import time as _t
        emb_msg = String()
        emb_msg.data = _json.dumps({
            'embedding': emb.tolist(),
            'timestamp': ts if ts is not None else _t.time(),
        })
        self._voice_emb_pub.publish(emb_msg)

    def _sv_threshold_for(self, seg_sec: float) -> float:
        """Прогрессивный порог SV: линейно растёт с длиной сегмента.

        Короткие сегменты дают менее надёжный ECAPA-TDNN эмбеддинг — снижаем порог.
        0.5с (минимум ECAPA) → sv_threshold - 0.15
        sv_seg_sec (полный сегмент) → sv_threshold
        За пределами sv_seg_sec — sv_threshold (клип).
        """
        _SV_MIN_SEC   = 0.5   # минимум для ECAPA-TDNN
        _SV_MAX_DELTA = 0.15  # максимальное снижение порога для коротких сегментов
        span = max(0.01, self._sv_seg_sec - _SV_MIN_SEC)
        ratio = min(1.0, max(0.0, (seg_sec - _SV_MIN_SEC) / span))
        return max(0.20, self._sv_threshold - _SV_MAX_DELTA * (1.0 - ratio))

    def _sv_decide(self, log_reject: bool = True) -> bool:
        """Принять решение по накопленному _sv_buf: сравнить с галереей или добавить первую запись.

        Возвращает True если сегмент принят (voice_buffer пополнен), False если отклонён.
        silence_counter сбрасывается в 0 ТОЛЬКО при принятии — чтобы чужой голос (TV и др.)
        не мешал счётчику тишины накапливаться и не растягивал запись до max_phrase_sec.
        """
        if not self._sv_buf:
            return False
        audio_seg = np.concatenate(self._sv_buf)
        emb = self._sv_embed(audio_seg)

        if emb is None:
            if self._sv_gallery:
                # Галерея установлена — слишком короткий сегмент отклоняем,
                # чтобы pre-roll чужого голоса не попадал в audio_buffer.
                self._sv_buf.clear()
                return False
            # Галерея пуста (знакомство / первая сессия) — принимаем без проверки.
            self.silence_counter = 0
            self.audio_buffer.extend(self._sv_buf)
            self.speech_chunks += len(self._sv_buf)
            self._sv_buf.clear()
            return True

        if not self._sv_gallery:
            if not log_reject:
                # Хвост при пустой галерее: принимаем, но галерею НЕ устанавливаем —
                # для первой надёжной записи нужен полный sv_seg_sec сегмент.
                self.silence_counter = 0
                self.audio_buffer.extend(self._sv_buf)
                self.speech_chunks += len(self._sv_buf)
                # В IDLE публикуем embedding чтобы identity_manager мог попробовать
                # распознать голос — галерея не устанавливается, только идентификация.
                # Проверяем _person_present напрямую: None (старт) и False (IDLE) — публикуем;
                # True (INTERACTING/RECOGNIZING) — пропускаем.
                if self._person_present is not True:
                    self._publish_voice_emb(emb)
                self._sv_buf.clear()
                return True
            # Первый полный сегмент — устанавливаем первую запись в галерею
            self._sv_gallery.append(emb)
            self._sv_gallery_times.append(time.time())
            self._sv_last_gallery_add = time.time()
            self.silence_counter = 0
            self.audio_buffer.extend(self._sv_buf)
            self.speech_chunks += len(self._sv_buf)
            self.get_logger().info(
                f'SV: первая запись в галерею (1/{self._SV_GALLERY_MAX}, ECAPA-TDNN)')
            self._publish_voice_emb(emb)
        else:
            seg_sec = len(audio_seg) / self.rate
            thresh  = self._sv_threshold_for(seg_sec)
            sim     = self._sv_sim(emb)
            if sim >= thresh:
                self.silence_counter = 0
                self.audio_buffer.extend(self._sv_buf)
                self.speech_chunks += len(self._sv_buf)
                # Добавляем в галерею во время знакомства или если есть место
                if self._introducing or len(self._sv_gallery) < self._SV_GALLERY_MAX:
                    self._sv_add_to_gallery(emb)
                if not log_reject:
                    self.get_logger().info(
                        f'SV: хвост принят ({seg_sec:.1f}с, '
                        f'сходство={sim:.2f} >= порог={thresh:.2f})')
                self._sv_buf.clear()
                return True
            else:
                if log_reject:
                    self.get_logger().warn(
                        f'SV: отброшен чужой голос ({seg_sec:.1f}с, '
                        f'сходство={sim:.2f} < порог={thresh:.2f})'
                    )
                else:
                    self.get_logger().info(
                        f'SV: хвост отброшен ({seg_sec:.1f}с, '
                        f'сходство={sim:.2f} < порог={thresh:.2f})'
                    )
                self._sv_buf.clear()
                return False
        self._sv_buf.clear()
        return True  # первая запись в галерею — accepted

    # ── Основной аудио колбэк ─────────────────────────────────────────────

    def _audio_callback(self, msg: Float32MultiArray):
        audio_float32 = np.array(msg.data, dtype=np.float32)

        if self._chunk_size is None:
            self._chunk_size = len(audio_float32)
            chunks_per_sec          = self.rate / self._chunk_size
            self._silence_threshold = int(self._silence_secs * chunks_per_sec)
            self._max_chunks        = int(self.max_phrase_sec * chunks_per_sec)
            self._pre_roll_maxlen   = int(self._pre_roll_secs * chunks_per_sec)
            self._onset_maxlen      = max(1, int(self._onset_secs * chunks_per_sec))
            self._sv_seg_samples    = int(self._sv_seg_sec * self.rate)
            self.get_logger().info(
                f'chunk_size={self._chunk_size} '
                f'silence_threshold={self._silence_threshold} чанков'
            )

        if self.tts_speaking:
            self._pre_roll_buffer.clear()
            return

        with torch.no_grad():
            confidence = self.vad_model(
                torch.from_numpy(audio_float32), self.rate
            ).item()

        if not self.is_active:
            self._pre_roll_buffer.append((audio_float32, confidence))
            if self._pre_roll_maxlen and len(self._pre_roll_buffer) > self._pre_roll_maxlen:
                self._pre_roll_buffer.popleft()

            if (self._stt_sent_time > 0.0
                    and (time.time() - self._stt_sent_time) > self.pipeline_timeout):
                self.get_logger().warn(
                    f'Таймаут пайплайна ({self.pipeline_timeout:.0f}с) — '
                    f'возвращаюсь к wake word'
                )
                self._stt_sent_time = 0.0
            return

        # Таймаут ожидания первого слова
        if not self.audio_buffer and not self._sv_buf:
            if (time.time() - self.activation_time) > self.no_speech_timeout:
                if self._should_keep_listening() and not self._sleeping:
                    self.activation_time = time.time()
                else:
                    self.get_logger().info('Таймаут ожидания речи — возвращаюсь к wake word')
                    self.is_active       = False
                    self.activation_time = 0.0
                return

        if confidence > self.vad_threshold:
            if self._sv_enabled and self._sv_encoder:
                # Режим SV: silence_counter НЕ сбрасывается здесь.
                # Сброс происходит внутри _sv_decide только при принятии сегмента.
                # Это гарантирует что чужой голос (TV, другие люди) не обнуляет
                # счётчик тишины и не растягивает запись до max_phrase_sec.
                if not self.audio_buffer and not self._sv_buf and self._onset_buf:
                    self._sv_buf.extend(self._onset_buf)
                    self._onset_buf.clear()
                self._sv_buf.append(audio_float32)
                # Если накопили достаточно — принимаем решение
                sv_samples = sum(len(c) for c in self._sv_buf)
                if sv_samples >= self._sv_seg_samples:
                    self._sv_decide(log_reject=True)
            else:
                # Без SV: классическое поведение
                self.silence_counter = 0
                if not self.audio_buffer and self._onset_buf:
                    self.audio_buffer.extend(self._onset_buf)
                    self._onset_buf.clear()
                self.audio_buffer.append(audio_float32)
                self.speech_chunks += 1

        else:
            if self.audio_buffer or self._sv_buf:
                # Тишина после речи — сначала сбрасываем недозаполненный sv_buf
                if self._sv_buf:
                    self._sv_decide(log_reject=False)
                self.audio_buffer.append(audio_float32)
                self.silence_counter += 1

                if self.silence_counter >= self._silence_threshold:
                    self._finish_recording(reason='silence')
            else:
                # Тишина до первой речи — накапливаем onset буфер
                self._onset_buf.append(audio_float32)
                if self._onset_maxlen and len(self._onset_buf) > self._onset_maxlen:
                    self._onset_buf.popleft()

        # Защита от бесконечной записи по количеству принятых чанков
        if self.is_active and len(self.audio_buffer) >= self._max_chunks:
            self.get_logger().warn('Достигнут лимит длины фразы — принудительно завершаю')
            self._finish_recording(reason='timeout')

    # ── Завершение записи ─────────────────────────────────────────────────

    def _finish_recording(self, reason: str = 'silence'):
        # Сбрасываем незавершённый SV-сегмент перед отправкой
        if self._sv_buf:
            self._sv_decide(log_reject=False)

        # Во время знакомства ожидаются короткие ответы (имя, "Ника", "да") —
        # порог минимальной длины речи снижается, чтобы их не отбрасывать.
        effective_min_speech_sec = (
            self.min_speech_sec_introducing if self._introducing else self.min_speech_sec
        )
        min_chunks        = int(self.min_phrase_sec * self.rate / self._chunk_size)
        min_speech_chunks = int(effective_min_speech_sec * self.rate / self._chunk_size)
        speech_sec        = self.speech_chunks * self._chunk_size / self.rate

        if len(self.audio_buffer) > min_chunks and self.speech_chunks >= min_speech_chunks:
            full_audio = np.concatenate(self.audio_buffer)
            duration   = len(full_audio) / self.rate

            self.get_logger().info(
                f'Фраза записана ({duration:.1f}с, речь={speech_sec:.1f}с, причина={reason}, '
                f'∆wake={time.time()-self.activation_time:.1f}с). Отправляю в STT...'
            )
            self._stt_sent_time = time.time()
            # Реальная распознанная речь = прямое доказательство присутствия,
            # даже если /person_present (по лицу/телу от identity_manager) ни
            # разу не приходил True в этой сессии (чисто голосовой диалог,
            # лицо не поймано). Без этого _is_person_present() всегда False
            # (см. её реализацию — сверяет только с последним True по лицу),
            # и авто-активация после TTS отключается насовсем — робот
            # переставал слышать пользователя. Живой баг 2026-08-28.
            self._person_last_seen = time.time()
            self._publish(full_audio)
        else:
            self.get_logger().info(
                f'Фраза отброшена: речи {speech_sec:.1f}с < {effective_min_speech_sec:.1f}с'
                f'{" (знакомство)" if self._introducing else ""} — игнорирую'
            )
            if self._should_keep_listening() and not self._sleeping:
                import threading
                threading.Timer(0.5, self._activate_after_tts).start()

        self.audio_buffer    = []
        self.silence_counter = 0
        self.speech_chunks   = 0
        self.is_active       = False
        self._onset_buf.clear()
        self._sv_buf.clear()
        # Галерея НЕ сбрасывается — живёт всю сессию до go_idle / robot_sleep

    def _publish(self, audio: np.ndarray):
        msg = Float32MultiArray()
        dim = MultiArrayDimension()
        dim.label  = 'sample_rate'
        dim.size   = len(audio)
        dim.stride = self.rate
        msg.layout.dim = [dim]
        msg.data = audio.tolist()
        self.publisher_.publish(msg)




def main():
    rclpy.init()
    node = VoiceDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
