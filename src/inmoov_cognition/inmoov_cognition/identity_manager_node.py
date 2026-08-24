#!/usr/bin/env python3
"""
identity_manager_node.py
========================
«Поставщик Социального Контекста» — объединяет данные от face_recognition
и emotion_recognition и публикует богатый контекст для Behavior Tree.

НЕ отправляет прямые команды. Только данные.

Конечный автомат:
  IDLE        — нет лица в кадре
  RECOGNIZING — лицо есть, ждём идентификации
  INTERACTING — знаем кто перед нами, следим за эмоцией
  INTRODUCING — неизвестный человек, собираем имя через голос

Публикует:
  /social_context  (String JSON → behavior_manager Blackboard)
      Поля: person_present, person_id, name, is_known, emotion,
            should_greet, greet_text, introducing,
            introduce_pending, introduce_text, state
  /person_context  (String JSON → llm_node)
  /person_present  (Bool) — быстрый interrupt сигнал
  /introducing     (Bool → llm_node gate)
  /face_expression (String → face_expressions_node, мимика-зеркало)
  /vision/enable   (Bool, latched)

Подписки:
  /face/identity  (String JSON)
  /face/emotion   (String JSON)
  /face/tracks    (String JSON)
  /voice_command  (String) — перехват в режиме INTRODUCING

"""

import collections
import json
import random
import re
import time
import threading

import requests
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String, Bool
from inmoov_msgs.srv import MemoryQuery


class State:
    IDLE        = 'idle'
    RECOGNIZING = 'recognizing'
    INTERACTING = 'interacting'
    INTRODUCING = 'introducing'


class IdentityManagerNode(LifecycleNode):
    _SV_SESSION_MAX = 3      # максимум голосовых записей за сессию INTERACTING
    _SV_SESSION_GAP = 120.0  # минимальный интервал между записями (сек)

    def __init__(self):
        super().__init__('identity_manager_node')


        # ── Состояние ─────────────────────────────────────────────────────
        self._state           = State.IDLE
        self._primary_track   = None
        self._primary_embedding: list | None = None   # последний embedding трека
        self._enroll_embeddings: list[list] = []      # накопленные embeddings для энролмента
        self._enroll_max = 10                          # сколько собираем перед сохранением
        self._current_person  = {}
        self._last_face_time  = 0.0
        self._last_human_time = 0.0   # последний сигнал от human_detection_node
        self._last_greet      = {}     # person_id → timestamp
        self._session_greeted = set()
        self._last_emotion    = None   # последняя распознанная эмоция человека (для social_context)
        # Fusion: оба источника должны согласиться, прежде чем менять мимику
        self._face_emo_pend:  dict | None = None  # {'emotion', 'ts', 'confidence'}
        self._voice_emo_pend: dict | None = None  # {'emotion', 'ts', 'confidence'}
        self._FUSION_WINDOW = 15.0  # секунд, в течение которых оба сигнала должны совпасть
        self._lock            = threading.Lock()

        self._introduce_last     = 0.0
        self._introduce_attempts = 0
        self._enrolled_track_id: int | None = None  # трек, залоченный ДО энролмента — игнорируем
        self._sleeping           = False   # спящий режим — PIR игнорируется
        self._face_hunt_since    = 0.0    # когда начали ждать лицо при живом теле
        self._last_dialogue_ts   = 0.0    # последний ответ LLM (диалог активен)
        self._voice_id_grace_ts  = 0.0    # голосовая идентификация из IDLE (watchdog grace)
        # Greet cooldown по имени (независимо от person_id — tracker может менять id)
        self._greeted_names: dict[str, float] = {}   # name → timestamp
        # После OakD вето: блокируем _tracks_cb до следующего сигнала тела.
        # Это разрывает бесконечный цикл "Привет/До свидания" при false positive face_detection.
        self._waiting_for_body: bool = False

        # Left-primary / right-fallback для трекинга: если левая камера молчит
        # дольше track_eye_fallback_sec — переключаемся на правую.
        self._left_track_last_msg: float = 0.0   # время последнего msg от левой
        self._tracks_eye: str = 'left'            # текущий активный глаз

        # Взгляд: смотрит ли собеседник роботу в глаза.
        # Вычисляется из kps InsightFace — асимметрия нос/глаза (yaw-proxy).
        # Хранится скользящее окно последних 15 значений (~1.5с @ 10Hz детекции).
        self._frontal_scores: collections.deque = collections.deque(maxlen=15)
        self._looking_at_robot: bool = True  # default True пока нет данных

        # Голосовой отпечаток текущей сессии (от voice_detector через /voice_embedding)
        self._session_voice_emb: list | None = None
        # Все голосовые embeddings сессии (для сохранения в галерею при знакомстве)
        self._session_voice_gallery: list = []  # [{embedding, timestamp}]
        # Счётчик: сколько голосовых записей уже сохранено в БД в этой сессии INTERACTING.
        # Разрешаем до 3 записей за сессию с интервалом не менее 2 минут.
        self._session_voice_save_count: int = 0
        self._session_voice_last_save_ts: float = 0.0

        # Верификация заявленной личности (в режиме INTRODUCING)
        # Ступени: face_sim < FACE_VETO → другой человек; >= FACE_ACCEPT → принимаем;
        # между — нужен голос.
        self._FACE_VETO         = 0.28   # явно другое лицо — голос не поможет
        self._FACE_ACCEPT       = 0.45   # лицо говорит ДА — принимаем без голоса
        self._VOICE_ACCEPT      = 0.52   # голос подтверждает в uncertain-зоне лица
        # Три флага состояния pending-ответа:
        #   _pending_name_confirm     — имя, ждём да/нет подтверждения
        #   _pending_name_confirm_pid — person_id если имя уже в БД (для accept_claim),
        #                               None если новый человек (для enroll)
        #   _skip_db_check            — следующее имя не проверяем по БД
        #                               (ситуация "другой Артур, придумайте прозвище")
        self._pending_name_confirm:     str | None = None
        self._pending_name_confirm_pid: int | None = None
        self._skip_db_check:            bool       = False

        # Пост-прощальный кулдаун: после явного goodbye игнорируем человека N минут
        # (или до wake word). Ключ — имя человека, значение — timestamp прощания.
        self._post_goodbye_names: dict[str, float] = {}
        # Блокировка обработки треков лица на N секунд после goodbye — предотвращает
        # петлю IDLE→RECOGNIZING→post_goodbye→IDLE при работающей face_detection.
        self._post_goodbye_track_block_until: float = 0.0

        # Когда последний раз видели текущего человека (по person_id из face_recognition).
        # Переключение на другого собеседника возможно только спустя _dialogue_switch_timeout.
        self._current_person_last_seen: float = 0.0

        # ── Социальный контекст (публикуется в /social_context) ───────────
        # Одноразовые флаги: устанавливаются перед публикацией, сбрасываются после.
        self._should_greet       = False   # BT должен поздороваться
        self._greet_text         = ''      # текст приветствия
        self._introduce_pending  = False   # BT должен произнести фразу знакомства
        self._introduce_text     = ''      # текст для произнесения

        self._watchdog_timer = None
        self._ctx_timer      = None

    # ── Wakeword / Спящий режим ───────────────────────────────────────────

    def _robot_sleep_cb(self, msg: Bool):
        """Получаем команду сна от behavior_manager. Vision управляется BT."""
        if msg.data and not self._sleeping:
            self._sleeping = True
            self.get_logger().info('Спящий режим активирован')
            with self._lock:
                self._state           = State.IDLE
                self._primary_track   = None
                self._current_person  = {}
                self._last_emotion    = None
                self._face_emo_pend   = None
                self._voice_emo_pend  = None
                self._face_hunt_since = 0.0
                self._frontal_scores.clear()
                self._looking_at_robot = True
            self._pub_person_present(False)
        elif not msg.data and self._sleeping:
            self._sleeping = False
            self.get_logger().info('Пробуждение: восстанавливаю нормальный режим')

    def _wakeword_cb(self, msg: Bool):
        if not msg.data:
            return
        # Wake word: снимаем пост-прощальные блокировки — человек сам инициирует диалог
        with self._lock:
            if self._post_goodbye_names:
                names = ', '.join(self._post_goodbye_names.keys())
                self._post_goodbye_names.clear()
                self.get_logger().info(f'Wake word: пост-прощальный кулдаун снят ({names})')
            self._post_goodbye_track_block_until = 0.0
        if self._sleeping:
            self.get_logger().info('Wakeword: выхожу из спящего режима')
            self._sleeping = False
            wake_msg = Bool()
            wake_msg.data = False
            self._robot_sleep_pub.publish(wake_msg)
            face_msg = String()
            face_msg.data = 'neutral'
            self._face_expr_pub.publish(face_msg)
        # Включение face_detection — задача BT (через PIRScanBranch / пробуждение)

    def _llm_response_seen_cb(self, _msg):
        """LLM ответил → диалог активен, сбрасываем таймер поиска лица."""
        with self._lock:
            self._last_dialogue_ts = time.time()
            self._face_hunt_since  = 0.0

    # ── Колбэки ───────────────────────────────────────────────────────────

    def _tracks_left_cb(self, msg: String):
        self._left_track_last_msg = time.time()
        if self._tracks_eye != 'left':
            self._tracks_eye = 'left'
            self.get_logger().info('Tracks: левая камера восстановлена — возврат с правой')
        self._tracks_cb(msg)

    def _tracks_right_cb(self, msg: String):
        elapsed = time.time() - self._left_track_last_msg
        if self._left_track_last_msg > 0.0 and elapsed < self._track_eye_fallback_sec:
            return  # левая активна — правую игнорируем
        if self._tracks_eye != 'right':
            self._tracks_eye = 'right'
            self.get_logger().warn(
                f'Tracks: левая камера недоступна ({elapsed:.1f}с) — переключаюсь на правую')
        self._tracks_cb(msg)

    def _tracks_cb(self, msg: String):
        if self._sleeping:
            return
        if self._waiting_for_body:
            return
        if time.time() < self._post_goodbye_track_block_until:
            return  # пост-прощальная блокировка треков активна
        try:
            data   = json.loads(msg.data)
            tracks = data.get('tracks', [])
        except Exception:
            return

        with self._lock:
            if tracks:
                self._last_face_time  = time.time()
                self._face_hunt_since = 0.0   # лицо снова видно — сбрасываем охоту
                track_ids = {t['track_id'] for t in tracks}

                if self._primary_track not in track_ids:
                    best = max(tracks, key=lambda t: (
                        (t['bbox'][2] - t['bbox'][0]) * (t['bbox'][3] - t['bbox'][1])))
                    new_track = best['track_id']

                    if self._state == State.IDLE:
                        # Новое лицо из IDLE — начинаем распознавание
                        self._primary_track = new_track
                        self._state = State.RECOGNIZING
                        self.get_logger().info('Лицо обнаружено — распознаём...')
                        self._pub_person_present(True)
                    else:
                        # INTRODUCING / INTERACTING / RECOGNIZING:
                        # Просто обновляем track_id. Если человек сменился — _identity_cb это определит
                        # по person_id (или locked=True + unknown). Это предотвращает лишние переходы
                        # состояний при обычной смене track_id из-за движения головы.
                        self.get_logger().debug(
                            f'Трек переназначен: {self._primary_track}→{new_track} '
                            f'(state={self._state})')
                        self._primary_track = new_track

                # Сохраняем embedding и вычисляем frontal score главного трека
                for t in tracks:
                    if t['track_id'] == self._primary_track:
                        emb       = t.get('embedding', [])
                        det_score = t.get('det_score', 1.0)
                        if emb:
                            self._primary_embedding = emb
                            # В режиме знакомства накапливаем только качественные embeddings
                            if self._state == State.INTRODUCING:
                                if (len(self._enroll_embeddings) < self._enroll_max
                                        and det_score >= self._min_enroll_det):
                                    self._enroll_embeddings.append(emb)
                        # Yaw-proxy из 5 InsightFace keypoints:
                        # kps[0]=left_eye, kps[1]=right_eye, kps[2]=nose_tip
                        # Если нос смещён от центра между глазами < 30% inter-eye dist → фронталь
                        kps = t.get('kps', [])
                        if len(kps) >= 3:
                            eye_mid_x   = (kps[0][0] + kps[1][0]) / 2.0
                            eye_dist    = abs(kps[1][0] - kps[0][0])
                            nose_offset = abs(kps[2][0] - eye_mid_x)
                            yaw_proxy   = nose_offset / max(eye_dist, 1.0)
                            self._frontal_scores.append(yaw_proxy)
                            if len(self._frontal_scores) >= 3:
                                frontal_fraction = sum(
                                    1 for s in self._frontal_scores
                                    if s < self._gaze_yaw_threshold
                                ) / len(self._frontal_scores)
                                self._looking_at_robot = (
                                    frontal_fraction >= self._gaze_frontal_fraction
                                )
                        break

    def _identity_cb(self, msg: String):
        if self._sleeping:
            return
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        with self._lock:
            if data.get('track_id') != self._primary_track:
                return
            state = self._state

        is_known  = data.get('is_known', False)
        locked    = data.get('locked', False)
        person_id = data.get('person_id')
        name      = data.get('name', 'Незнакомец')

        confidence  = data.get('confidence', 'unknown')
        candidate_id   = data.get('best_candidate_id')
        candidate_name = data.get('best_candidate_name') or name

        if state == State.RECOGNIZING:
            if is_known and confidence == 'high':
                self._on_known(person_id, name)
            elif not is_known and confidence == 'uncertain' and locked:
                # Вероятно известный человек — тихий INTERACTING без приветствия
                self._on_uncertain(candidate_id, candidate_name)
            elif not is_known and confidence == 'unknown' and locked:
                self._on_unknown()
            else:
                self.get_logger().debug(
                    f'Трек {data.get("track_id")} — ещё не опознан, ждём lock...')

        elif state == State.INTRODUCING:
            # Только если пришло распознавание известного человека — отменяем знакомство
            if is_known:
                self.get_logger().info(
                    f'INTRODUCING: лицо распознано как {name} (id={person_id}) — отменяю знакомство')
                now = time.time()
                with self._lock:
                    self._state               = State.INTERACTING
                    self._introduce_attempts  = 0
                    self._enroll_embeddings   = []
                    self._current_person      = {'person_id': person_id, 'name': name}
                    self._greeted_names[name] = now
                    self._last_emotion        = 'neutral'
                self._set_introducing(False)
                threading.Thread(
                    target=self._update_seen_and_embedding,
                    args=(person_id,), daemon=True).start()
                threading.Thread(
                    target=self._fetch_and_publish_context,
                    args=(person_id,), daemon=True).start()
                self._should_greet = True
                self._greet_text   = f'Ой, {name}! Прости, я тебя не сразу узнал.'

        elif state == State.INTERACTING:
            now = time.time()
            with self._lock:
                current_id = self._current_person.get('person_id')
                current_name = self._current_person.get('name', '?')

            if is_known and confidence == 'high' and person_id == current_id:
                # Тот же человек — обновляем метку последнего появления
                with self._lock:
                    self._current_person_last_seen = now
            elif is_known and confidence == 'high' and person_id != current_id:
                # Другой человек — переключаем только после длительного отсутствия текущего
                with self._lock:
                    absent_sec = (now - self._current_person_last_seen
                                  if self._current_person_last_seen > 0 else float('inf'))
                if absent_sec < self._dialogue_switch_timeout:
                    self.get_logger().debug(
                        f'Игнорирую {name} (id={person_id}) — '
                        f'диалог с {current_name}, последний раз видели {absent_sec:.0f}с назад')
                    return
                self.get_logger().info(
                    f'INTERACTING: смена человека → {name} (id={person_id}) '
                    f'({current_name} не виден {absent_sec:.0f}с)')
                with self._lock:
                    self._state             = State.RECOGNIZING
                    self._current_person    = {}
                    self._enroll_embeddings = []
                    self._last_emotion      = None
                self._on_known(person_id, name)
            elif not is_known and confidence == 'uncertain' and locked:
                # Лицо похоже на кандидата но ниже порога — не меняем состояние
                pass
            elif not is_known and confidence == 'unknown' and locked:
                # Неизвестный — переключаем только после длительного отсутствия текущего
                with self._lock:
                    absent_sec = (now - self._current_person_last_seen
                                  if self._current_person_last_seen > 0 else float('inf'))
                    in_cooldown       = (now - self._introduce_last) < self._introduce_cooldown
                    is_enrolled_track = (data.get('track_id') == self._enrolled_track_id)
                if in_cooldown or is_enrolled_track:
                    return
                if absent_sec < self._dialogue_switch_timeout:
                    self.get_logger().debug(
                        f'Игнорирую неизвестного — '
                        f'диалог с {current_name}, последний раз видели {absent_sec:.0f}с назад')
                    return
                self.get_logger().info(
                    f'INTERACTING: неизвестный после {absent_sec:.0f}с отсутствия '
                    f'{current_name} — знакомство')
                with self._lock:
                    self._state             = State.RECOGNIZING
                    self._current_person    = {}
                    self._enroll_embeddings = []
                    self._last_emotion      = None
                self._on_unknown()

    def _emotion_cb(self, msg: String):
        if self._sleeping:
            return
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        with self._lock:
            if data.get('track_id') != self._primary_track:
                return
            if self._state != State.INTERACTING:
                return

        emotion    = data.get('emotion', 'neutral')
        confidence = data.get('confidence', 0.0)

        if confidence < self._emotion_thresh:
            return

        # Обновляем social_context (для LLM) по лицу без fusion
        if emotion != self._last_emotion:
            self._last_emotion = emotion

        # Fusion: сохраняем pending и проверяем согласие с голосом
        self._face_emo_pend = {'emotion': emotion, 'ts': time.time(), 'confidence': confidence}
        self._check_emotion_fusion()

    def _voice_emotion_cb(self, msg: String):
        if self._sleeping:
            return
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        with self._lock:
            if self._state != State.INTERACTING:
                return

        emotion    = data.get('emotion', 'neutral')
        confidence = data.get('confidence', 0.0)

        if confidence < 0.55:
            return

        self.get_logger().info(f'Голос-эмоция получена: {emotion} (conf={confidence:.2f})')
        self._voice_emo_pend = {'emotion': emotion, 'ts': time.time(), 'confidence': confidence}
        self._check_emotion_fusion()

    def _check_emotion_fusion(self):
        """Реагируем на эмоцию только если лицо и голос согласны в пределах FUSION_WINDOW."""
        face  = self._face_emo_pend
        voice = self._voice_emo_pend
        if face is None or voice is None:
            return

        now = time.time()
        if (now - face['ts']) > self._FUSION_WINDOW:
            return
        if (now - voice['ts']) > self._FUSION_WINDOW:
            return

        if face['emotion'] != voice['emotion']:
            return

        emotion = face['emotion']
        self.get_logger().info(
            f'Fusion: лицо={face["emotion"]}({face["confidence"]:.2f}) '
            f'+ голос={voice["emotion"]}({voice["confidence"]:.2f}) → реагируем'
        )
        # Сбрасываем, чтобы не реагировать повторно на ту же пару
        self._face_emo_pend  = None
        self._voice_emo_pend = None
        self._react_to_emotion(emotion)

    # ── Голосовой отпечаток ───────────────────────────────────────────────

    def _voice_embedding_cb(self, msg: String):
        """Получаем голосовой embedding от voice_detector.

        Пять ролей:
        1. IDLE → пробуем идентифицировать по голосу, переход в INTERACTING если узнали
        2. INTERACTING с известным человеком → add_voice_to_gallery (1 раз за сессию)
        3. INTRODUCING → накапливаем в _session_voice_gallery, проверяем по БД
        4. RECOGNIZING → пробуем идентифицировать по голосу (параллельно с face recognition)
        5. Всегда → обновляем _session_voice_emb (последний) и _session_voice_gallery
        """
        try:
            data = json.loads(msg.data)
            emb  = data.get('embedding')
            if not emb:
                return
            ts = float(data.get('timestamp', time.time()))
        except Exception:
            return

        self._session_voice_emb = emb
        self._session_voice_gallery.append({'embedding': emb, 'timestamp': ts})

        with self._lock:
            state     = self._state
            person_id = self._current_person.get('person_id')

        if state == State.IDLE:
            # Нет лица — пробуем узнать по голосу
            threading.Thread(
                target=self._try_voice_id_from_idle,
                args=(emb,), daemon=True).start()
            return

        if state == State.INTERACTING and person_id is not None:
            # Сохраняем до _SV_SESSION_MAX записей за сессию с интервалом _SV_SESSION_GAP.
            # memory_node применит правило 7 дней: если галерея полная и свежая — пропустит.
            now = time.time()
            gap_ok   = (now - self._session_voice_last_save_ts) >= self._SV_SESSION_GAP
            count_ok = self._session_voice_save_count < self._SV_SESSION_MAX
            if count_ok and gap_ok:
                self._session_voice_save_count  += 1
                self._session_voice_last_save_ts = now
                self.get_logger().info(
                    f'Голосовой embedding принят (pid={person_id}), '
                    f'запись {self._session_voice_save_count}/{self._SV_SESSION_MAX} — сохраняю в БД')
                threading.Thread(
                    target=self._add_voice_to_gallery,
                    args=(person_id, emb, ts), daemon=True).start()
            elif not count_ok:
                self.get_logger().debug(
                    f'Голосовая галерея: лимит {self._SV_SESSION_MAX}/сессию достигнут, пропускаю')
            else:
                remaining = self._SV_SESSION_GAP - (now - self._session_voice_last_save_ts)
                self.get_logger().debug(
                    f'Голосовая галерея: слишком рано (ещё {remaining:.0f}с до следующей записи)')

        elif state == State.RECOGNIZING:
            threading.Thread(
                target=self._try_voice_identification,
                args=(emb,), daemon=True).start()

        elif state == State.INTRODUCING:
            threading.Thread(
                target=self._try_voice_id_in_introducing,
                args=(emb,), daemon=True).start()

    def _try_voice_id_from_idle(self, emb: list):
        """Голосовая идентификация из IDLE (лицо не видно, сработал wake word).

        Если голос узнан с высокой уверенностью — переходим в INTERACTING как при
        обычном распознавании лица. 2Hz цикл опубликует person_present=True на след. тике.
        """
        result = self._call_memory({'op': 'lookup_by_voice', 'embedding': emb,
                                    'high_threshold': self._voice_high_threshold,
                                    'uncertain_threshold': self._voice_uncertain_threshold})
        if not result or result.get('confidence') != 'high':
            sim  = result.get('similarity', 0) if result else 0
            name = result.get('name', '?')     if result else '?'
            conf = result.get('confidence', 'none') if result else 'no_result'
            self.get_logger().info(
                f'Голосовая идентификация из IDLE: не узнан '
                f'(лучший={name}, sim={sim:.3f}, confidence={conf}, '
                f'нужно sim >= {self._voice_high_threshold} для перехода в INTERACTING)')
            return

        person_id = result['person_id']
        name      = result.get('name', '?')

        with self._lock:
            if self._state != State.IDLE:
                return  # состояние изменилось (лицо распозналось пока мы запрашивали)
            # Watchdog grace period: нет лица и тела — это норма при голосовой
            # идентификации. Отдельное поле, не _last_dialogue_ts (тот проверяется
            # в _on_known → dialogue_recently, что заблокировало бы приветствие).
            self._voice_id_grace_ts = time.time()

        self.get_logger().info(
            f'Голосовая идентификация из IDLE: {name} (id={person_id}, '
            f'sim={result.get("similarity", 0):.3f}) — перехожу в INTERACTING')
        self._on_known(person_id, name)

    def _try_voice_identification(self, emb: list):
        """Пробуем узнать человека по голосу пока face recognition ещё работает."""
        result = self._call_memory({'op': 'lookup_by_voice', 'embedding': emb,
                                    'high_threshold': self._voice_high_threshold,
                                    'uncertain_threshold': self._voice_uncertain_threshold})
        if not result or result.get('confidence') != 'high':
            sim  = result.get('similarity', 0) if result else 0
            name = result.get('name', '?')     if result else '?'
            conf = result.get('confidence', 'none') if result else 'no_result'
            self.get_logger().info(
                f'Голосовая идентификация (RECOGNIZING): не узнан '
                f'(лучший={name}, sim={sim:.3f}, confidence={conf})')
            return

        person_id = result['person_id']
        name      = result.get('name', '?')

        with self._lock:
            # Гонка: проверяем что мы всё ещё в RECOGNIZING (face recognition не завершилось)
            if self._state != State.RECOGNIZING:
                return

        self.get_logger().info(
            f'Голосовая идентификация: {name} (id={person_id}, '
            f'sim={result.get("similarity", 0):.3f}) — опережаю face recognition')
        self._on_known(person_id, name)

    def _try_voice_id_in_introducing(self, emb: list):
        """В режиме знакомства: проверяем голосовой отпечаток против галереи.

        high       → сразу INTERACTING (голос узнан уверенно)
        uncertain  → спрашиваем "Вы случайно не {name}?" → pending confirm
        unknown    → игнорируем
        """
        result = self._call_memory({'op': 'lookup_by_voice', 'embedding': emb,
                                    'high_threshold': self._voice_high_threshold,
                                    'uncertain_threshold': self._voice_uncertain_threshold})
        if not result:
            return

        confidence = result.get('confidence', 'unknown')
        sim        = result.get('similarity', 0.0)

        if confidence == 'high':
            person_id = result['person_id']
            name      = result.get('name', '?')

            with self._lock:
                if self._state != State.INTRODUCING:
                    return
                now = time.time()
                self._state                    = State.INTERACTING
                self._introduce_attempts       = 0
                self._pending_name_confirm     = None
                self._pending_name_confirm_pid = None
                self._skip_db_check            = False
                self._enroll_embeddings        = []
                self._current_person           = {'person_id': person_id, 'name': name}
                self._greeted_names[name]      = now
                self._last_emotion             = 'neutral'
                self._current_person_last_seen = now

            self.get_logger().info(
                f'INTRODUCING: голос узнан — {name} (id={person_id}, sim={sim:.3f})')
            self._set_introducing(False)
            self._should_greet = True
            self._greet_text   = f'Прости, {name}! Я тебя не узнал по лицу, но узнал по голосу.'
            threading.Thread(
                target=self._update_seen_and_embedding,
                args=(person_id,), daemon=True).start()
            threading.Thread(
                target=self._fetch_and_publish_context,
                args=(person_id,), daemon=True).start()

        elif confidence == 'uncertain':
            person_id   = result.get('best_candidate_id')
            cand_name   = result.get('best_candidate_name', '?')
            if person_id is None:
                return
            with self._lock:
                if self._state != State.INTRODUCING:
                    return
                if self._pending_name_confirm is not None:
                    return  # уже ждём ответа — не перебиваем
                self._pending_name_confirm     = cand_name
                self._pending_name_confirm_pid = person_id
            self.get_logger().info(
                f'INTRODUCING: голос похож на {cand_name} (sim={sim:.3f}) — спрашиваю')
            self._introduce_pending = True
            self._introduce_text    = f'Вы случайно не {cand_name}?'
        else:
            self.get_logger().debug(
                f'Голосовая идентификация (INTRODUCING): не узнан (sim={sim:.3f})')

    def _add_voice_to_gallery(self, person_id: int, emb: list, ts: float = None):
        """Добавляет embedding в голосовую галерею (memory_node применит правило 7 дней)."""
        import time as _t
        result = self._call_memory({
            'op':        'add_voice_to_gallery',
            'person_id': person_id,
            'embedding': emb,
            'timestamp': ts if ts is not None else _t.time(),
        })
        if result is None:
            self.get_logger().warn(f'Голосовая галерея: _call_memory вернул None (таймаут/ошибка) для pid={person_id}')
        elif result.get('added'):
            self.get_logger().info(
                f'Голосовая галерея сохранена в БД: pid={person_id}, '
                f'count={result.get("count")}/10')
        else:
            self.get_logger().info(
                f'Голосовая галерея НЕ сохранена: pid={person_id}, '
                f'причина={result.get("reason", "?")} (age={result.get("oldest_age_days", "?")}д)')

    def _publish_voice_anchor(self, person_id: int, name: str):
        """Загружаем голосовую галерею из БД и отправляем в voice_detector."""
        result = self._call_memory({'op': 'get_voice_gallery', 'person_id': person_id})
        if result and result.get('has_voice'):
            msg = String()
            msg.data = json.dumps({
                'person_id': person_id,
                'name':      name,
                'gallery':   result['gallery'],
            }, ensure_ascii=False)
            self._voice_anchor_pub.publish(msg)
            n = len(result['gallery'])
            self.get_logger().info(f'Голосовая галерея отправлена в SV: {name} ({n} записей)')
        else:
            self.get_logger().info(f'Нет голосового отпечатка для {name} — SV будет учиться с нуля')

    # ── Голосовой ответ в режиме знакомства ──────────────────────────────

    def _voice_cmd_cb(self, msg: String):
        """Перехватываем voice_command только в режиме INTRODUCING."""
        with self._lock:
            if self._state != State.INTRODUCING:
                return
        text = msg.data.strip()

        # Если ждём подтверждения да/нет — обрабатываем отдельно
        with self._lock:
            pending_confirm = self._pending_name_confirm
        if pending_confirm is not None:
            threading.Thread(
                target=self._handle_name_confirmation, args=(text,), daemon=True).start()
            return

        if not text:
            # STT не распознал речь — переспрашиваем как при неудачной попытке
            with self._lock:
                self._introduce_attempts += 1
                attempts = self._introduce_attempts
            if attempts >= self._max_attempts:
                self.get_logger().warn('STT: нет ответа — перехожу в INTERACTING')
                self._set_introducing(False)
                with self._lock:
                    self._state = State.INTERACTING
                    self._introduce_attempts = 0
            else:
                self.get_logger().info(
                    f'STT: тишина (попытка {attempts}/{self._max_attempts}) — переспрашиваю')
                self._introduce_pending = True
                self._introduce_text    = random.choice(self._RETRY_PHRASES)
            return
        self.get_logger().info(f'INTRODUCING: получен голосовой ответ: "{text}"')
        threading.Thread(
            target=self._handle_introduce_response, args=(text,), daemon=True).start()

    def _handle_introduce_response(self, text: str):
        # Пока LLM думал, watchdog мог перевести в IDLE
        with self._lock:
            if self._state != State.INTRODUCING:
                return
            skip_db = self._skip_db_check

        name = self._extract_name(text)
        if not name:
            with self._lock:
                if self._state != State.INTRODUCING:
                    return
                self._introduce_attempts += 1
                attempts = self._introduce_attempts
            if attempts >= self._max_attempts:
                self.get_logger().warn('Не удалось получить имя — перехожу в INTERACTING')
                self._set_introducing(False)
                with self._lock:
                    self._state = State.INTERACTING
                    self._introduce_attempts = 0
            else:
                self.get_logger().info(
                    f'Имя не найдено (попытка {attempts}/{self._max_attempts}) — переспрашиваю')
                self._introduce_pending = True
                self._introduce_text    = random.choice(self._RETRY_PHRASES)
            return

        self.get_logger().info(f'Имя распознано: "{name}"')

        # Режим "другой человек с тем же именем" — пропускаем проверку по БД
        if skip_db:
            with self._lock:
                self._skip_db_check = False
            # Спрашиваем подтверждение перед энролментом нового человека
            with self._lock:
                self._pending_name_confirm     = name
                self._pending_name_confirm_pid = None
            self._introduce_pending = True
            self._introduce_text    = f'Вас зовут {name}? Я правильно понял?'
            return

        # Проверяем: есть ли такое имя в БД?
        existing = self._call_memory({'op': 'lookup_by_name', 'name': name})
        with self._lock:
            if self._state != State.INTRODUCING:
                return

        if existing and existing.get('person_id') is not None:
            # Имя есть в БД → запускаем верификацию
            person_id = existing['person_id']
            db_name   = existing['name']
            self.get_logger().info(
                f'Имя "{name}" найдено в БД как "{db_name}" (id={person_id}) — верифицирую')
            threading.Thread(
                target=self._verify_claimed_identity,
                args=(person_id, db_name), daemon=True).start()
        else:
            # Новое имя → запрашиваем подтверждение перед энролментом
            with self._lock:
                self._pending_name_confirm     = name
                self._pending_name_confirm_pid = None
            self._introduce_pending = True
            self._introduce_text    = f'Вас зовут {name}? Я правильно понял?'

    # Быстрый regex-путь: находим первое слово с заглавной буквы (рус/лат)
    _NAME_RE = re.compile(r'\b([А-ЯЁ][а-яё]{1,20}|[A-Z][a-z]{1,20})\b')

    # Стоп-слова — не являются именами, даже если начинаются с заглавной
    _NAME_STOPWORDS = {
        'Меня', 'Зовут', 'Мне', 'Моё', 'Моя', 'Мой', 'Это',
        'Да', 'Нет', 'Привет', 'Здравствуй', 'Пожалуйста',
        'Пока', 'Всё', 'Ладно', 'Хорошо', 'Спасибо', 'Понятно',
        'Ничего', 'Прости', 'Извини', 'Стоп', 'Стой', 'Окей',
        'Можно', 'Нельзя', 'Конечно', 'Именно', 'Просто', 'Тогда',
        'Когда', 'Потом', 'Сейчас', 'Здесь', 'Туда', 'Сюда',
    }

    # Триггерные слова для шаблона — IGNORECASE только для них, не для захватываемого имени
    # (?i:...) — inline флаг применяется только внутри группы
    _TRIGGER_RE = re.compile(r'(?i:зовут|зови|называй|меня)\s+([А-ЯЁ][а-яё]{1,20})')

    def _extract_name(self, text: str) -> str | None:
        """Извлекает имя из фразы.

        Сначала пробует быстрый regex без обращения к сети.
        Если фраза сложная (несколько слов, нет очевидного имени) — спрашивает LLM.
        """
        cleaned = text.strip().strip('.,!?"\'').strip()

        # Быстрый путь по шаблону — приоритет, проверяем ПЕРВЫМ
        # "меня зовут Артур" / "зовут Артур" / "меня Артур"
        # IGNORECASE только для триггерных слов, имя должно начинаться с заглавной
        m = self._TRIGGER_RE.search(cleaned)
        if m:
            name = m.group(1)
            if name not in self._NAME_STOPWORDS:
                self.get_logger().info(f'Имя из regex (шаблон): "{name}"')
                return name

        # Быстрый путь: вся фраза — одно-два слова (напр. "Артур" или "Я Артур")
        words = cleaned.split()
        if len(words) <= 2:
            names = [n for n in self._NAME_RE.findall(cleaned)
                     if n not in self._NAME_STOPWORDS]
            if names:
                self.get_logger().info(f'Имя из regex (короткая фраза): "{names[0]}"')
                return names[0]

        # Резерв — LLM для сложных случаев (используем ту же модель что уже загружена).
        # Короткий нестриминговый запрос — не нужна SSE-задержка ради 10 токенов.
        payload = {
            'model': self._name_model,
            'messages': [
                {
                    'role':    'system',
                    'content': (
                        'Извлеки имя человека из фразы. '
                        'Ответь ТОЛЬКО именем в именительном падеже (например: Артур). '
                        'Если имя не упомянуто — ответь: UNKNOWN'
                    ),
                },
                {'role': 'user', 'content': text},
            ],
            'stream':      False,
            'temperature': 0.0,
            'max_tokens':  10,
            'chat_template_kwargs': {'enable_thinking': False},
        }
        headers = {'Authorization': f'Bearer {self._bearer_token}'} if self._bearer_token else {}
        for url in [self._llm_url, self._llm_fallback_url]:
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=(3.0, 25.0))
                r.raise_for_status()
                choices = r.json().get('choices') or []
                name = (choices[0].get('message', {}).get('content', '') if choices else '').strip()
                if name.upper() == 'UNKNOWN' or not name:
                    return None
                name = name.split()[0].strip('.,!?"\'')
                return name if name else None
            except requests.exceptions.ConnectionError:
                continue
            except Exception as e:
                self.get_logger().error(f'Ошибка извлечения имени: {e}')
                return None
        return None

    # ── Верификация заявленной личности ──────────────────────────────────

    _YES_RE = re.compile(r'(?i:^да$|^верно$|^правильно$|^именно$|^точно$|^угу$|^ага$|\bда\b|\bверно\b|\bточно\b)')
    _NO_RE  = re.compile(r'(?i:^нет$|^неверно$|^неправильно$|\bнет\b|\bне\s+(?:так|верно|правильно)\b)')

    def _handle_name_confirmation(self, text: str):
        """Обрабатывает да/нет ответ на вопрос подтверждения имени."""
        with self._lock:
            if self._state != State.INTRODUCING:
                self._pending_name_confirm     = None
                self._pending_name_confirm_pid = None
                return
            name = self._pending_name_confirm
            pid  = self._pending_name_confirm_pid

        if name is None:
            return

        is_yes = bool(self._YES_RE.search(text.strip()))
        is_no  = bool(self._NO_RE.search(text.strip()))

        self.get_logger().info(
            f'Подтверждение имени "{name}": text="{text}", yes={is_yes}, no={is_no}')

        if is_yes:
            with self._lock:
                self._pending_name_confirm     = None
                self._pending_name_confirm_pid = None
            if pid is not None:
                self._accept_identity_claim(pid, name)
            else:
                self._enroll_new_person(name)
        elif is_no:
            with self._lock:
                self._pending_name_confirm     = None
                self._pending_name_confirm_pid = None
                self._introduce_attempts      += 1
                attempts = self._introduce_attempts
            if attempts >= self._max_attempts:
                self.get_logger().warn('Подтверждение отклонено, лимит попыток — INTERACTING')
                self._set_introducing(False)
                with self._lock:
                    self._state              = State.INTERACTING
                    self._introduce_attempts = 0
            else:
                self._introduce_pending = True
                self._introduce_text    = 'Прошу прощения! Как вас зовут?'
        else:
            # Неясный ответ — переспрашиваем
            self._introduce_pending = True
            self._introduce_text    = f'Вас зовут {name}? Скажите "да" или "нет".'

    def _verify_claimed_identity(self, person_id: int, name: str):
        """Тиер-верификация: лицо первично, голос — тайбрейкер.

        face_sim < FACE_VETO               → другой человек (голос не учитываем)
        face_sim in [FACE_VETO, FACE_ACCEPT) → uncertain: нужен голос или вопрос
        face_sim >= FACE_ACCEPT            → принимаем (голос не нужен)
        """
        with self._lock:
            if self._state != State.INTRODUCING:
                return
            face_emb  = self._primary_embedding
            voice_emb = self._session_voice_emb

        result = self._call_memory({
            'op':              'verify_person_claim',
            'person_id':       person_id,
            'face_embedding':  face_emb  if face_emb  else None,
            'voice_embedding': voice_emb if voice_emb else None,
        })
        if result is None:
            result = {}

        face_sim          = result.get('face_sim', 0.0)
        voice_sim         = result.get('voice_sim', 0.0)
        has_voice_gallery = result.get('has_voice_gallery', False)

        self.get_logger().info(
            f'verify_claim "{name}" (id={person_id}): '
            f'face={face_sim:.3f} [veto<{self._FACE_VETO} accept>={self._FACE_ACCEPT}], '
            f'voice={voice_sim:.3f} [accept>={self._VOICE_ACCEPT}] '
            f'has_voice={has_voice_gallery}')

        with self._lock:
            if self._state != State.INTRODUCING:
                return

        if face_sim >= self._FACE_ACCEPT:
            # Лицо убедительно — принимаем
            self._accept_identity_claim(person_id, name)

        elif face_sim >= self._FACE_VETO:
            # Неуверенная зона — нужен голос
            if has_voice_gallery and voice_sim >= self._VOICE_ACCEPT:
                self._accept_identity_claim(person_id, name)
            elif has_voice_gallery and voice_sim < self._VOICE_ACCEPT:
                # Голос опровергает — другой человек с тем же именем
                self._start_same_name_flow(name)
            else:
                # Нет голоса в БД — спрашиваем напрямую
                with self._lock:
                    self._pending_name_confirm     = name
                    self._pending_name_confirm_pid = person_id
                self._introduce_pending = True
                self._introduce_text    = (
                    f'Вы очень похожи на {name} в моей памяти, но я не уверен. '
                    f'Вы точно {name}?'
                )
        else:
            # Лицо явно другое — голос не важен
            self._start_same_name_flow(name)

    def _accept_identity_claim(self, person_id: int, name: str):
        """Переход в INTERACTING с извинением — личность подтверждена."""
        with self._lock:
            if self._state != State.INTRODUCING:
                return
            now = time.time()
            self._state                    = State.INTERACTING
            self._introduce_attempts       = 0
            self._pending_name_confirm     = None
            self._pending_name_confirm_pid = None
            self._skip_db_check            = False
            self._enroll_embeddings        = []
            self._current_person           = {'person_id': person_id, 'name': name}
            self._greeted_names[name]      = now
            self._last_emotion             = 'neutral'
            self._current_person_last_seen = now

        self.get_logger().info(f'Верификация успешна: {name} (id={person_id})')
        self._set_introducing(False)
        self._should_greet = True
        self._greet_text   = f'Прости, {name}! Я тебя не узнал. Больше постараюсь запомнить!'
        threading.Thread(
            target=self._update_seen_and_embedding, args=(person_id,), daemon=True).start()
        threading.Thread(
            target=self._fetch_and_publish_context, args=(person_id,), daemon=True).start()
        threading.Thread(
            target=self._publish_voice_anchor, args=(person_id, name), daemon=True).start()

    def _start_same_name_flow(self, existing_name: str):
        """Запускаем поток знакомства с новым человеком, у которого то же имя."""
        with self._lock:
            if self._state != State.INTRODUCING:
                return
            self._pending_name_confirm     = None
            self._pending_name_confirm_pid = None
            self._skip_db_check            = True   # следующий ответ — прямо в энролмент

        self.get_logger().info(
            f'Другой человек с именем "{existing_name}" — запускаю новое знакомство')
        self._introduce_pending = True
        self._introduce_text    = (
            f'Интересно! У меня уже есть знакомый по имени {existing_name}, '
            f'но вы на него не похожи. Как вас называть, чтобы не перепутать?'
        )

    # ─────────────────────────────────────────────────────────────────────

    def _enroll_new_person(self, name: str):
        """Сохраняем нового человека в память и запускаем приветствие."""
        import numpy as np
        with self._lock:
            collected  = list(self._enroll_embeddings)
            fallback   = self._primary_embedding

        if not collected and not fallback:
            self.get_logger().warn('Нет embedding для энролмента — перехожу в INTERACTING')
            self._set_introducing(False)
            with self._lock:
                self._state = State.INTERACTING
            return

        if collected:
            # Усредняем все собранные embeddings и нормируем результат
            mat = np.array(collected, dtype=np.float32)
            avg = mat.mean(axis=0)
            norm = np.linalg.norm(avg)
            if norm > 0:
                avg /= norm
            embedding = avg.tolist()
            self.get_logger().info(
                f'Энролмент: усреднено {len(collected)} embeddings для "{name}"')
        else:
            embedding = fallback
            self.get_logger().warn(
                f'Энролмент: только 1 embedding (сбор не завершён) для "{name}"')

        result = self._call_memory({
            'op':        'save_person',
            'name':      name,
            'embedding': embedding,
        })

        if result and 'person_id' in result:
            person_id = result['person_id']
            self.get_logger().info(f'Зарегистрирован: {name} (id={person_id})')

            with self._lock:
                self._state                    = State.INTERACTING
                self._current_person           = {'person_id': person_id, 'name': name}
                self._session_greeted.add(person_id)
                self._introduce_attempts       = 0
                self._last_emotion             = 'neutral'
                self._introduce_last           = time.time()
                self._enrolled_track_id        = self._primary_track
                self._current_person_last_seen = time.time()
                voice_gallery = list(self._session_voice_gallery)

            self._set_introducing(False)

            # Сохраняем голосовую галерею нового человека
            if voice_gallery:
                def _save_gallery(pid, entries):
                    for entry in entries:
                        self._call_memory({
                            'op':        'add_voice_to_gallery',
                            'person_id': pid,
                            'embedding': entry['embedding'],
                            'timestamp': entry['timestamp'],
                        })
                    self.get_logger().info(
                        f'Голосовая галерея сохранена для {name} ({len(entries)} записей)')
                threading.Thread(
                    target=_save_gallery, args=(person_id, voice_gallery), daemon=True).start()

            # Публикуем контекст для LLM
            threading.Thread(
                target=self._fetch_and_publish_context,
                args=(person_id,), daemon=True).start()

            # BT приветствует нового знакомого через social_context
            self._should_greet = True
            self._greet_text   = (
                f'Очень приятно познакомиться, {name}! '
                f'Расскажи немного о себе.'
            )
        else:
            self.get_logger().error('Не удалось сохранить человека в память')
            self._set_introducing(False)
            with self._lock:
                self._state = State.INTERACTING

    def _set_introducing(self, active: bool):
        """Публикует /introducing — гейт для llm_node."""
        msg = Bool()
        msg.data = active
        self._introducing_pub.publish(msg)

    # ── Логика состояний ──────────────────────────────────────────────────

    def _on_known(self, person_id: int, name: str):
        """Переход в INTERACTING для известного человека.

        Вызывается из RECOGNIZING (face recognition) или IDLE (голосовая идентификация).
        """
        now = time.time()
        with self._lock:
            # Кулдаун по имени — не зависит от person_id (tracker может менять id)
            already_greeted = (now - self._greeted_names.get(name, 0)) < self._greet_cooldown
            goodbye_ts = self._post_goodbye_names.get(name, 0)

        # Пост-прощальный кулдаун: человек сам попрощался — до wake word не приветствуем
        if (now - goodbye_ts) < self._post_goodbye_ignore_sec:
            remaining_min = (self._post_goodbye_ignore_sec - (now - goodbye_ts)) / 60

            with self._lock:
                self._state         = State.IDLE
                self._primary_track = None
                self._current_person = {}
                # Продляем блокировку треков чтобы не зациклиться
                self._post_goodbye_track_block_until = now + self._post_goodbye_track_block_sec
            self._pub_person_present(False)
            return

        # Если диалог недавно вёлся только по голосу (без распознавания лица),
        # face recognition нашёл человека — не приветствуем снова.
        with self._lock:
            dialogue_recently = (self._last_dialogue_ts > 0.0 and
                                 (now - self._last_dialogue_ts) < self._greet_cooldown)

        if not already_greeted and not dialogue_recently:
            self._greet_known(person_id, name, now)
        else:
            # Тихий переход — уже приветствовали или был активный голосовой диалог
            with self._lock:
                self._state                    = State.INTERACTING
                self._current_person           = {'person_id': person_id, 'name': name}
                self._greeted_names[name]      = now   # обновляем cooldown
                self._last_emotion             = 'neutral'
                self._current_person_last_seen = now
            reason = 'cooldown активен' if already_greeted else 'диалог по голосу был активен'
            self.get_logger().info(
                f'{name} распознан (id={person_id}), {reason} — сразу INTERACTING без приветствия')
            threading.Thread(
                target=self._publish_voice_anchor,
                args=(person_id, name), daemon=True).start()
            threading.Thread(
                target=self._fetch_and_publish_context,
                args=(person_id,), daemon=True).start()

    def _on_uncertain(self, person_id: int | None, name: str):
        """Лицо похоже на известного человека, но sim ниже уверенного порога.
        Переходим в INTERACTING без приветствия и без знакомства.
        Галерея накопит больше фото и следующий раз распознает увереннее.
        """
        now = time.time()
        with self._lock:
            self._state                    = State.INTERACTING
            self._current_person           = {'person_id': person_id, 'name': name}
            self._last_emotion             = 'neutral'
            self._current_person_last_seen = now
        self.get_logger().info(
            f'Неуверенное распознавание: вероятно {name} (id={person_id}) — '
            f'тихий INTERACTING, знакомство не начинаем')
        if person_id is not None:
            threading.Thread(
                target=self._fetch_and_publish_context,
                args=(person_id,), daemon=True).start()

    def _greet_known(self, person_id: int, name: str, now: float):
        with self._lock:
            self._state                    = State.INTERACTING
            self._current_person           = {'person_id': person_id, 'name': name}
            self._greeted_names[name]      = now
            self._last_emotion             = 'neutral'
            self._current_person_last_seen = now

        self.get_logger().info(f'Приветствую: {name} (id={person_id})')

        threading.Thread(
            target=self._update_seen_and_embedding,
            args=(person_id,), daemon=True).start()

        threading.Thread(
            target=self._publish_voice_anchor,
            args=(person_id, name), daemon=True).start()

        # Напоминания запрашиваются асинхронно — should_greet выставляется после
        threading.Thread(
            target=self._fetch_reminders_and_greet,
            args=(person_id, name), daemon=True).start()

        threading.Thread(
            target=self._fetch_and_publish_context,
            args=(person_id,), daemon=True).start()

    # Фразы для начала знакомства — случайный выбор для естественности
    _INTRO_PHRASES = [
        'Привет! Я тебя раньше не видел. Как тебя зовут?',
        'Здравствуй! Мы ещё не знакомы. Как тебя зовут?',
        'О, новое лицо! Рад познакомиться. Как вас зовут?',
        'Привет! Я Лёня. А тебя как зовут?',
        'Добро пожаловать! Я тебя не знаю. Представься, пожалуйста.',
        'Привет! Не помню, чтобы мы встречались. Как тебя зовут?',
        'Здравствуй! Я не знаю твоего имени. Как тебя зовут?',
        'О, привет! Ты новый человек для меня. Как вас зовут?',
    ]

    # Фразы для переспроса — случайный выбор
    _RETRY_PHRASES = [
        'Простите, я не расслышал. Как вас зовут?',
        'Извините, не разобрал. Повторите ваше имя, пожалуйста.',
        'Прошу прощения, не услышал. Как вас зовут?',
        'Не расслышал имя. Не могли бы повторить?',
    ]

    def _on_unknown(self):
        now = time.time()
        with self._lock:
            if self._state == State.INTRODUCING:
                return   # уже спрашиваем
            if (now - self._introduce_last) < self._introduce_cooldown:
                self._state = State.INTERACTING
                return
            # OakD гейт: если OAK-D активен и не видит тела >10с — лицо false positive,
            # не начинаем знакомство (watchdog переведёт в IDLE сам)
            if self._last_human_time > 0.0 and (now - self._last_human_time) > 10.0:
                self.get_logger().warn(
                    f'Неизвестное лицо, но OakD не видит тела '
                    f'{now - self._last_human_time:.0f}с — '
                    f'пропускаю знакомство (вероятно false positive face_detection)')
                return
            self._state = State.INTRODUCING
            self._introduce_last     = now
            self._introduce_attempts = 0
            self._enroll_embeddings  = []   # начинаем накапливать с нуля
            self._enrolled_track_id  = None
            emotion = self._last_emotion or 'surprised'

        phrase = random.choice(self._INTRO_PHRASES)
        self.get_logger().info('Неизвестный человек — инициирую знакомство')
        self._set_introducing(True)
        # Не публикуем прямую команду — BT прочитает introduce_pending из social_context
        self._introduce_pending = True
        self._introduce_text    = phrase

    def _react_to_emotion(self, emotion: str):
        mirror_map = {
            'happy':     'happy',
            'sad':       'sad',
            'angry':     'neutral',
            'surprised': 'surprised',
            'fear':      'neutral',
            'disgust':   'neutral',
            'neutral':   'neutral',
        }
        robot_emotion = mirror_map.get(emotion, 'neutral')
        # Мимика-зеркало: рефлекс, публикуется напрямую без BT (низкоуровневый рефлекс)
        msg = String()
        msg.data = robot_emotion
        self._face_expr_pub.publish(msg)
        self.get_logger().info(f'Эмоция человека: {emotion} → мимика: {robot_emotion}')

    def _human_detected_cb(self, msg: Bool):
        """Вторичный сигнал присутствия от OAK-D body detection."""
        if self._sleeping:
            return
        if msg.data:
            with self._lock:
                self._last_human_time = time.time()
            if self._waiting_for_body:
                self._waiting_for_body = False
                self.get_logger().info('OakD: тело обнаружено → снимаю блокировку face_detection')

    def _go_idle_cb(self, msg: Bool):
        """Принудительный переход в IDLE от BT (say_goodbye tool call LLM)."""
        if not msg.data or self._sleeping:
            return
        with self._lock:
            if self._state == State.IDLE:
                return
            # Сохраняем имя и режим до очистки состояния
            goodbye_name    = self._current_person.get('name', '')
            prev_state      = self._state
            now             = time.time()

            self.get_logger().info('go_idle: принудительный переход в IDLE')
            self._state                    = State.IDLE
            self._primary_track            = None
            self._primary_embedding        = None
            self._enroll_embeddings        = []
            self._current_person           = {}
            self._last_emotion             = None
            self._face_emo_pend            = None
            self._voice_emo_pend           = None
            self._introduce_attempts       = 0
            self._greeted_names            = {}
            self._enrolled_track_id        = None
            self._face_hunt_since          = 0.0
            self._last_dialogue_ts         = 0.0
            self._session_voice_emb        = None
            self._session_voice_gallery    = []
            self._session_voice_save_count   = 0
            self._session_voice_last_save_ts = 0.0
            self._current_person_last_seen = 0.0
            self._pending_name_confirm     = None
            self._pending_name_confirm_pid = None
            self._skip_db_check            = False
            self._waiting_for_body         = True
            self._frontal_scores.clear()
            self._looking_at_robot         = True
            was_introducing                = (prev_state == State.INTRODUCING)

            # Пост-прощальный кулдаун: человек сам попрощался — игнорируем его
            # N минут (или до wake word), даже если face_detection его видит.
            if goodbye_name:
                self._post_goodbye_names[goodbye_name] = now
                self.get_logger().info(
                    f'Пост-прощальный кулдаун: {goodbye_name} — '
                    f'{self._post_goodbye_ignore_sec / 60:.0f} мин (или до wake word)')
            # Блокировка треков на N секунд — предотвращает петлю re-greeting
            self._post_goodbye_track_block_until = now + self._post_goodbye_track_block_sec

        self._pub_context({})
        self._pub_person_present(False)
        # Публикуем /introducing False только если реально были в режиме знакомства —
        # иначе voice_detector и llm_node шумят "INTRODUCING завершён" при обычном прощании.
        if was_introducing:
            self._set_introducing(False)

    def _watchdog(self):
        if self._sleeping:
            return
        now = time.time()
        with self._lock:
            if self._state == State.IDLE:
                return

            go_idle     = False
            idle_reason = ''

            # OakD вето: если OAK-D активен (получали хоть один сигнал) и не видит тела
            # дольше no_human_timeout — считаем face_detection false positive и уходим в IDLE.
            # Это критично: левая камера может детектировать постер/отражение бесконечно,
            # тогда как реального человека давно нет.
            if self._last_human_time > 0.0:
                body_absent_sec = now - self._last_human_time
                if body_absent_sec > self._no_human_timeout:
                    go_idle     = True
                    idle_reason = (
                        f'OakD вето: тело не видно {body_absent_sec:.0f}с '
                        f'(face_detection вероятно false positive)')
                    self._waiting_for_body = True

            if not go_idle:
                # INTRODUCING нужно больше времени: TTS + VAD-задержка + речь + STT
                face_timeout = (self._no_face_timeout * 4
                                if self._state == State.INTRODUCING
                                else self._no_face_timeout)

                face_lost = (now - self._last_face_time) > face_timeout
                if not face_lost:
                    return  # лицо видно — всё хорошо

                # Диалог активен: LLM недавно ответил → даём 60с для ответа пользователя
                dialogue_active = (self._last_dialogue_ts > 0.0 and
                                   (now - self._last_dialogue_ts) < 60.0)
                if dialogue_active:
                    face_lost_sec = now - self._last_face_time
                    dialogue_sec  = now - self._last_dialogue_ts
                    self._face_hunt_since = 0.0
                    return

                # Голосовая идентификация из IDLE: нет лица/тела — норма, человек
                # рядом но за кадром. Даём 60с до первого LLM-ответа (после которого
                # _last_dialogue_ts обновится и подхватит эстафету).
                if (self._voice_id_grace_ts > 0.0 and
                        (now - self._voice_id_grace_ts) < 60.0):
                    self._face_hunt_since = 0.0
                    return

                # Лицо потеряно — проверяем body detection как запасной сигнал
                human_detected = (now - self._last_human_time) < self._no_human_timeout
                if human_detected:
                    face_lost_sec = now - self._last_face_time
                    if self._face_hunt_since == 0.0:
                        self._face_hunt_since = now
                    hunt_sec = now - self._face_hunt_since
                    if hunt_sec >= self._max_face_hunt:
                        idle_reason = (
                            f'Лицо не найдено {hunt_sec:.0f}с при живом теле — '
                            f'сброс в IDLE, BT перезапустит поиск')
                        go_idle = True
                    else:
                        return
                else:
                    idle_reason = 'Человек ушёл (нет лица и тела)'
                    go_idle = True

            if go_idle:
                self.get_logger().info(f'{idle_reason} → IDLE')
                self._state                    = State.IDLE
                self._primary_track            = None
                self._primary_embedding        = None
                self._enroll_embeddings        = []
                self._current_person           = {}
                self._last_emotion             = None
                self._face_emo_pend            = None
                self._voice_emo_pend           = None
                self._introduce_attempts       = 0
                self._greeted_names            = {}
                self._enrolled_track_id        = None
                self._face_hunt_since          = 0.0
                self._last_dialogue_ts         = 0.0
                self._voice_id_grace_ts        = 0.0
                self._session_voice_emb        = None
                self._session_voice_gallery    = []
                self._session_voice_save_count   = 0
                self._session_voice_last_save_ts = 0.0
                self._current_person_last_seen = 0.0
                self._pending_name_confirm     = None
                self._pending_name_confirm_pid = None
                self._skip_db_check            = False
                self._pub_context({})
                self._pub_person_present(False)
                self._set_introducing(False)


    def _update_seen_and_embedding(self, person_id: int):
        """update_seen + EMA обновление embedding при каждой встрече."""
        import numpy as np
        self._call_memory({'op': 'update_seen', 'person_id': person_id})

        with self._lock:
            collected = list(self._enroll_embeddings)
            fallback  = self._primary_embedding

        embeddings = collected if collected else ([fallback] if fallback else [])
        if not embeddings:
            return

        # Усредняем доступные embeddings и обновляем через EMA
        mat = np.array(embeddings, dtype=np.float32)
        avg = mat.mean(axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg /= norm
        self._call_memory({
            'op':        'update_embedding',
            'person_id': person_id,
            'embedding': avg.tolist(),
            'alpha':     0.2,   # мягкое обновление: 20% новый, 80% старый
        })

    # ── Вспомогательные ───────────────────────────────────────────────────

    def _publish_social_context(self):
        
        """Публикует социальный контекст @ 2 Гц → BehaviorManager Blackboard.

        Полностью останавливается в спящем режиме.
        should_greet и introduce_pending — одноразовые сигналы: True публикуется
        один раз, затем автоматически сбрасывается.

        """
        if self._sleeping:
            return
            
        with self._lock:
            state           = self._state
            person          = dict(self._current_person)
            emotion         = self._last_emotion or 'neutral'
            intro           = (state == State.INTRODUCING)
            looking         = self._looking_at_robot
            has_face        = bool(self._frontal_scores)

        person_present = state != State.IDLE
        ctx = {
            'person_present':    person_present,
            'person_id':         person.get('person_id'),
            'name':              person.get('name', ''),
            'is_known':          bool(person.get('person_id')),
            'emotion':           emotion,
            'state':             state,
            'introducing':       intro,
            # True если собеседник смотрит в глаза роботу (yaw-proxy < 30% от inter-eye dist,
            # >50% кадров за последнюю ~1.5с). None если трек без kps / нет данных.
            'looking_at_robot':  looking if has_face else None,
            # Одноразовые флаги (сбрасываются после первой публикации)
            'should_greet':      self._should_greet,
            'greet_text':        self._greet_text if self._should_greet else '',
            'introduce_pending': self._introduce_pending,
            'introduce_text':    self._introduce_text if self._introduce_pending else '',
        }
        # Сброс одноразовых флагов
        self._should_greet      = False
        self._introduce_pending = False

        msg = String()
        msg.data = json.dumps(ctx, ensure_ascii=False)
        self._social_ctx_pub.publish(msg)

        # Обновляем /person_present Bool @ 2Hz — voice_detector использует timestamp
        # этого топика для определения присутствия человека во время диалога.
        # Без этого grace period (120с) истекает и pipeline уходит в wake word режим
        # даже когда человек активно взаимодействует (state=INTERACTING).
        if person_present:
            self._pub_person_present(True)

    def _pub_context(self, ctx: dict):
        msg = String()
        msg.data = json.dumps(ctx, ensure_ascii=False)
        self._context_pub.publish(msg)

    def _pub_person_present(self, present: bool):
        msg = Bool()
        msg.data = present
        self._person_present_pub.publish(msg)

    def _fetch_reminders_and_greet(self, person_id: int, name: str):
        """Запрашивает ручные напоминания и выставляет should_greet с текстом.

        Env-напоминания (source='env:...') в БД больше не хранятся —
        они уходят напрямую в Telegram через openhab_bridge_node.
        Здесь обрабатываем только manual-напоминания (delivered=0):
          - показываем в приветствии
          - помечаем delivered=1 и сразу удаляем (confirm_reminders)
        """
        import datetime as _dt

        now_dt     = _dt.datetime.now()
        today      = now_dt.date().isoformat()
        now_time   = now_dt.strftime('%H:%M')
        greet_text = f'Привет, {name}!'
        manual_ids: list[int] = []

        try:
            result = self._call_memory({
                'op':        'get_due_reminders',
                'person_id': person_id,
                'today':     today,
                'now_time':  now_time,
            })
            if result and isinstance(result.get('reminders'), list):
                for r in result['reminders']:
                    greet_text += ' ' + r['message']
                    manual_ids.append(r['id'])
        except Exception as e:
            self.get_logger().warn(f'_fetch_reminders_and_greet: {e}')

        with self._lock:
            if self._current_person.get('person_id') == person_id:
                self._should_greet = True
                self._greet_text   = greet_text

        # Помечаем delivered=1 и удаляем: сказано лично → можно удалять
        for rid in manual_ids:
            try:
                self._call_memory({'op': 'mark_reminder_delivered', 'reminder_id': rid})
            except Exception:
                pass
        if manual_ids:
            try:
                self._call_memory({'op': 'confirm_reminders', 'person_id': person_id})
            except Exception:
                pass
            self.get_logger().info(
                f'Напоминания для {name}: показано и удалено {len(manual_ids)} ручных')

    def _fetch_and_publish_context(self, person_id: int):
        result = self._call_memory({'op': 'get_context', 'person_id': person_id})
        if result and 'error' not in result:
            with self._lock:
                result['current_emotion'] = self._last_emotion
            self._pub_context(result)
            name = result.get('name', '?')
            self.get_logger().info(
                f'person_context → LLM: {name} (id={person_id}), '
                f'emotion={result.get("current_emotion")}'
            )
        else:
            self.get_logger().warn(
                f'Не удалось получить контекст для person_id={person_id}')

    def _call_memory(self, req: dict) -> dict | None:
        if not self._mem.wait_for_service(timeout_sec=2.0):
            return None
        request = MemoryQuery.Request()
        request.request_json = json.dumps(req)
        future = self._mem.call_async(request)
        done_event = threading.Event()
        future.add_done_callback(lambda _: done_event.set())
        if not done_event.wait(timeout=5.0):
            return None
        try:
            return json.loads(future.result().response_json)
        except Exception:
            return None


    # ── Lifecycle callbacks ────────────────────────────────────────────────

    def _dp(self, name, default=None):
        """Безопасный declare_parameter: игнорирует повторное объявление при re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('no_face_timeout_sec',          15.0)
        self._dp('no_human_timeout_sec',         20.0)
        self._dp('max_face_hunt_sec',            90.0)
        self._dp('greet_cooldown_sec',          120.0)
        self._dp('emotion_react_thresh',          0.70)
        self._dp('introduce_cooldown_sec',       60.0)
        self._dp('max_introduce_attempts',        3)
        self._dp('min_enroll_det_score',         0.65)
        self._dp('dialogue_switch_timeout_sec', 30.0)
        self._dp('track_eye_fallback_sec',        2.0)
        self._dp('post_goodbye_ignore_sec',    1800.0)
        self._dp('post_goodbye_track_block_sec', 30.0)
        self._dp('llm_url', 'http://192.168.10.118:18020/v1/chat/completions')
        self._dp('voice_high_threshold',    0.62)
        self._dp('voice_uncertain_threshold', 0.50)
        self._dp('llm_fallback_url', 'http://localhost:11434/v1/chat/completions')
        self._dp('bearer_token', '')
        self._dp('name_extract_model', 'qwen3.8-27b')
        self._dp('gaze_yaw_threshold',  0.30)
        self._dp('gaze_frontal_fraction', 0.50)

        self._no_face_timeout              = self.get_parameter('no_face_timeout_sec').value
        self._no_human_timeout             = self.get_parameter('no_human_timeout_sec').value
        self._max_face_hunt                = self.get_parameter('max_face_hunt_sec').value
        self._greet_cooldown               = self.get_parameter('greet_cooldown_sec').value
        self._emotion_thresh               = self.get_parameter('emotion_react_thresh').value
        self._introduce_cooldown           = self.get_parameter('introduce_cooldown_sec').value
        self._max_attempts                 = self.get_parameter('max_introduce_attempts').value
        self._min_enroll_det               = self.get_parameter('min_enroll_det_score').value
        self._dialogue_switch_timeout      = self.get_parameter('dialogue_switch_timeout_sec').value
        self._track_eye_fallback_sec       = self.get_parameter('track_eye_fallback_sec').value
        self._post_goodbye_ignore_sec      = self.get_parameter('post_goodbye_ignore_sec').value
        self._post_goodbye_track_block_sec = self.get_parameter('post_goodbye_track_block_sec').value
        self._llm_url                      = self.get_parameter('llm_url').value
        self._llm_fallback_url             = self.get_parameter('llm_fallback_url').value
        self._bearer_token                 = self.get_parameter('bearer_token').value
        self._name_model                   = self.get_parameter('name_extract_model').value
        self._voice_high_threshold         = self.get_parameter('voice_high_threshold').value
        self._voice_uncertain_threshold    = self.get_parameter('voice_uncertain_threshold').value
        self._gaze_yaw_threshold           = self.get_parameter('gaze_yaw_threshold').value
        self._gaze_frontal_fraction        = self.get_parameter('gaze_frontal_fraction').value

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(String, '/face/identity',     self._identity_cb,        10)
        self.create_subscription(String, '/face/emotion',      self._emotion_cb,         10)
        self.create_subscription(String, '/voice/emotion',     self._voice_emotion_cb,   10)
        self.create_subscription(String, '/face/tracks/left',  self._tracks_left_cb,     10)
        self.create_subscription(String, '/face/tracks/right', self._tracks_right_cb,    10)
        self.create_subscription(Bool,   '/wake_detected',     self._wakeword_cb,        10)
        self.create_subscription(Bool,   '/human_detected',    self._human_detected_cb,  10)
        self.create_subscription(String, '/voice_command',     self._voice_cmd_cb,       10)
        self.create_subscription(String, '/voice_embedding',   self._voice_embedding_cb, 10)
        self.create_subscription(Bool,   '/robot_sleep',       self._robot_sleep_cb,     latched_qos)
        self.create_subscription(Bool,   '/go_idle',           self._go_idle_cb,         10)
        self.create_subscription(String, '/llm_response',      self._llm_response_seen_cb, 10)

        self._social_ctx_pub     = self.create_lifecycle_publisher(String, '/social_context',  10)
        self._context_pub        = self.create_lifecycle_publisher(String, '/person_context',  10)
        self._person_present_pub = self.create_lifecycle_publisher(Bool,   '/person_present',  10)
        self._face_expr_pub      = self.create_lifecycle_publisher(String, '/face_expression', 10)
        self._introducing_pub    = self.create_lifecycle_publisher(Bool,   '/introducing',     10)
        self._voice_anchor_pub   = self.create_lifecycle_publisher(String, '/voice_anchor',    10)
        self._robot_sleep_pub    = self.create_lifecycle_publisher(Bool,   '/robot_sleep',     latched_qos)

        self._mem = self.create_client(MemoryQuery, '/memory/query')
        self.get_logger().info('IdentityManager настроен')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._social_ctx_pub.on_activate(state)
        self._context_pub.on_activate(state)
        self._person_present_pub.on_activate(state)
        self._face_expr_pub.on_activate(state)
        self._introducing_pub.on_activate(state)
        self._voice_anchor_pub.on_activate(state)
        self._robot_sleep_pub.on_activate(state)
        self._watchdog_timer = self.create_timer(0.5, self._watchdog)
        self._ctx_timer      = self.create_timer(0.5, self._publish_social_context)
        self.get_logger().info('IdentityManager готов')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        if self._watchdog_timer:
            self.destroy_timer(self._watchdog_timer)
            self._watchdog_timer = None
        if self._ctx_timer:
            self.destroy_timer(self._ctx_timer)
            self._ctx_timer = None
        self._social_ctx_pub.on_deactivate(state)
        self._context_pub.on_deactivate(state)
        self._person_present_pub.on_deactivate(state)
        self._face_expr_pub.on_deactivate(state)
        self._introducing_pub.on_deactivate(state)
        self._voice_anchor_pub.on_deactivate(state)
        self._robot_sleep_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

def main():
    rclpy.init()
    node = IdentityManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
