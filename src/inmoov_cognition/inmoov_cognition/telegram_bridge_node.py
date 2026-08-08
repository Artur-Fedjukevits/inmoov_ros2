#!/usr/bin/env python3
"""
telegram_bridge_node.py
========================
Telegram ↔ ROS2 мост для удалённого управления роботом InMoov.

/ask и свободный текст маршрутизируются через llm_node (/telegram_ask → /telegram_response),
так что работают все tool calls, умный дом и персональная память.

Идентификация: по telegram_id в таблице persons (заполняется вручную).
При совпадении person_ctx инжектируется в llm_node, робот знает с кем общается.

Безопасность:
  - Только allowed_chat_id получает ответы (остальные молча игнорируются)
  - Токен только из TELEGRAM_BOT_TOKEN env var

Команды:
  /status          — режим, человек, CPU/RAM, состояние TTS/LLM серверов
  /photo           — снимок с левой камеры → JPEG
  /say <текст>     — TTS через Speak action (без LLM)
  /ask <текст>     — через llm_node → полный пайплайн
  /wake            — /robot_sleep False
  /sleep           — /robot_sleep True (мгновенный переход в SLEEP, без LLM)
  /restart_tts     — перезапуск локального Docker-контейнера cosyvoice_api
  <любой текст>    — как /ask

Топики:
  /social_context  (in)   — состояние identity_manager
  /robot_sleep     (in/out latched)
  /telegram_ask    (out)  — запрос к llm_node
  /telegram_response (in) — ответ от llm_node
  /telegram_push   (in)   — push-уведомления от других нод (JSON: {"text": "..."})
Actions:
  speak — inmoov_msgs/action/Speak → tts_node
"""

import asyncio
import difflib
import json
import os
import queue
import re
import sqlite3
import subprocess
import threading
import time
import urllib.request
import uuid

import cv2
import numpy as np
import psutil
import rclpy
from rclpy.action import ActionClient
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import CompressedImage
from inmoov_msgs.action import Speak

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)


# ── Геолокация — хелперы ────────────────────────────────────────────────────

_TRANSLIT_MAP = {
    'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'yo','ж':'zh',
    'з':'z','и':'i','й':'j','к':'k','л':'l','м':'m','н':'n','о':'o',
    'п':'p','р':'r','с':'s','т':'t','у':'u','ф':'f','х':'kh','ц':'ts',
    'ч':'ch','ш':'sh','щ':'shch','ъ':'','ы':'y','ь':'','э':'e','ю':'yu','я':'ya',
}

# "где сейчас Настя?" / "где Настя" / "where is Nastja"
_WHERE_RE = re.compile(
    r'(?:где|where\s+is)\s+(?:сейчас\s+|находится\s+)?(\w+)',
    re.IGNORECASE,
)

def _translit(s: str) -> str:
    """Транслитерация кириллицы → латиница для fuzzy-матчинга имён."""
    return ''.join(_TRANSLIT_MAP.get(c, c) for c in s.lower())

def _parse_location_state(state: str) -> tuple[float, float] | None:
    """Парсит состояние OpenHAB Location item: 'lat,lon[,alt]' → (lat, lon)."""
    if not state or state in ('NULL', 'UNDEF', '-', ''):
        return None
    parts = state.split(',')
    if len(parts) < 2:
        return None
    try:
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            return lat, lon
    except (ValueError, IndexError):
        pass
    return None


# ── QoS ─────────────────────────────────────────────────────────────────────

_LATCHED = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)


# ── Основной класс ───────────────────────────────────────────────────────────

class TelegramBridgeNode(LifecycleNode):

    def __init__(self):
        super().__init__('telegram_bridge_node')
        self._sleep_pub           = None
        self._ask_pub             = None
        self._timer               = None
        self._lock                = threading.Lock()
        self._social_context      = {}
        self._lifecycle_status    = {}
        self._sleeping            = False
        self._pending_lock        = threading.Lock()
        self._pending             = {}
        self._latest_jpeg         = None
        self._image_lock          = threading.Lock()
        self._say_queue           = queue.Queue()
        self._push_queue          = asyncio.Queue()
        self._openhab_items_cache = []
        self._openhab_items_lock  = threading.Lock()
        self._loop                = None
        self._tg_thread           = None

    def _dp(self, name, default=None):
        """Безопасный declare_parameter: игнорирует повторное объявление при re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('allowed_chat_id', 0)
        self._dp('llm_timeout_sec', 35.0)
        self._dp('cam_device',
            '/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1.1:1.0-video-index0')
        self._dp('memory_db_path', '/home/artur/inmoov_memory.db')

        self._allowed_chat_id = self.get_parameter('allowed_chat_id').value
        self._llm_timeout     = self.get_parameter('llm_timeout_sec').value
        self._cam_device      = self.get_parameter('cam_device').value
        self._db_path         = self.get_parameter('memory_db_path').value

        self.create_subscription(String,          '/social_context',              self._social_context_cb,    10)
        self.create_subscription(Bool,            '/robot_sleep',                 self._robot_sleep_cb,     _LATCHED)
        self.create_subscription(String,          '/telegram_response',           self._telegram_response_cb, 10)
        self.create_subscription(CompressedImage, '/camera/eye_left/compressed',  self._camera_cb, 5)
        self.create_subscription(String,          '/telegram_push',               self._telegram_push_cb,     10)
        self.create_subscription(String,          '/openhab_items',               self._openhab_items_cb,     10)
        self.create_subscription(String,          '/lifecycle/status',            self._lifecycle_status_cb,  10)

        self._sleep_pub    = self.create_lifecycle_publisher(Bool, '/robot_sleep', _LATCHED)
        self._ask_pub      = self.create_lifecycle_publisher(String, '/telegram_ask', 10)
        self._speak_client = ActionClient(self, Speak, 'speak')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self._sleep_pub.on_activate(state)
        self._ask_pub.on_activate(state)
        self._timer = self.create_timer(0.1, self._process_say_queue)

        self._token = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
        if not self._token or not self._allowed_chat_id:
            self.get_logger().warn('Telegram bridge: нет токена или chat_id — бот не запущен')
            return TransitionCallbackReturn.SUCCESS

        self._loop       = asyncio.new_event_loop()
        # Пересоздаём Queue для нового event loop — старая Queue привязана к
        # предыдущему loop и вызывает RuntimeError при повторной активации.
        self._push_queue = asyncio.Queue()
        self._tg_thread  = threading.Thread(
            target=self._run_telegram_loop, daemon=True, name='telegram_loop')
        self._tg_thread.start()
        self.get_logger().info(
            f'Telegram bridge активирован. allowed_chat_id={self._allowed_chat_id}')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        if self._timer:
            self.destroy_timer(self._timer)
            self._timer = None
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._sleep_pub.on_deactivate(state)
        self._ask_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        return TransitionCallbackReturn.SUCCESS

    # ── ROS callbacks ────────────────────────────────────────────────────────

    def _camera_cb(self, msg: CompressedImage):
        """Кэширует последний сжатый кадр от face_capture_node."""
        with self._image_lock:
            self._latest_jpeg = bytes(msg.data)

    def _social_context_cb(self, msg: String):
        try:
            with self._lock:
                self._social_context = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def _robot_sleep_cb(self, msg: Bool):
        with self._lock:
            self._sleeping = msg.data

    def _lifecycle_status_cb(self, msg: String):
        try:
            with self._lock:
                self._lifecycle_status = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def _telegram_response_cb(self, msg: String):
        """Получает partial/final ответ от llm_node, кладёт в asyncio.Queue."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        req_id = data.get('request_id', '')
        with self._pending_lock:
            q = self._pending.get(req_id)
        if q and hasattr(self, '_loop'):
            # thread-safe push в asyncio event loop
            self._loop.call_soon_threadsafe(q.put_nowait, data)

    def _process_say_queue(self):
        """Таймер 100 мс: /say text → Speak ActionClient."""
        while not self._say_queue.empty():
            try:
                text = self._say_queue.get_nowait()
            except queue.Empty:
                break
            if not self._speak_client.wait_for_server(timeout_sec=0.5):
                self.get_logger().warn('Speak action server недоступен')
                continue
            goal = Speak.Goal()
            goal.text = text
            self._speak_client.send_goal_async(goal)
            self.get_logger().info(f'TG /say → TTS: "{text[:60]}"')

    def _telegram_push_cb(self, msg: String):
        """Push-уведомление из другой ноды (openhab_bridge, memory_node) → Telegram.

        Ожидает JSON: {"text": "...", "parse_mode": "HTML"} (parse_mode опционален).
        Без parse_mode → plain text (Telegram не интерпретирует тэги/спец-символы).
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            data = {'text': msg.data}
        text = data.get('text', '').strip()
        if not text or not self._allowed_chat_id:
            return
        if hasattr(self, '_loop'):
            # Передаём весь dict, чтобы сохранить parse_mode от отправителя
            self._loop.call_soon_threadsafe(self._push_queue.put_nowait, data)

    def _openhab_items_cb(self, msg: String):
        """Кэширует items из openhab_bridge_node (нужно для геолокации)."""
        try:
            items = json.loads(msg.data)
            with self._openhab_items_lock:
                self._openhab_items_cache = items
        except Exception:
            pass

    def _find_person_location(
        self, name_hint: str,
    ) -> tuple[float, float, str, str] | None:
        """Ищет Location item по имени (fuzzy, кириллица/латиница).

        Returns (lat, lon, item_name, display_name) или None.
        None с причиной 'no_fix' если item найден, но GPS-данных нет.
        """
        query_translit = _translit(name_hint.strip())
        query_lo       = name_hint.strip().lower()

        with self._openhab_items_lock:
            items = list(self._openhab_items_cache)

        best_ratio = 0.0
        best_item  = None
        for item in items:
            if item.get('type') != 'Location':
                continue
            iname = item.get('name', '')
            parts = iname.split('_')
            # Формат: Phone_Nastja_location
            if (len(parts) < 3
                    or parts[0].lower() != 'phone'
                    or parts[-1].lower() != 'location'):
                continue
            person_part = '_'.join(parts[1:-1]).lower()   # 'nastja'
            label       = item.get('label', iname).lower()

            r1 = difflib.SequenceMatcher(None, query_translit, person_part).ratio()
            r2 = difflib.SequenceMatcher(None, query_lo,       label).ratio()
            ratio = max(r1, r2)
            if ratio > best_ratio:
                best_ratio = ratio
                best_item  = item

        if best_item is None or best_ratio < 0.5:
            return None

        state  = best_item.get('state', '')
        coords = _parse_location_state(state)
        iname       = best_item.get('name', '')
        parts       = iname.split('_')
        display     = '_'.join(parts[1:-1]) if len(parts) >= 3 else iname
        label_field = best_item.get('label', display)

        if coords is None:
            # Item найден, но телефон офлайн / нет GPS-фикса
            return None, None, iname, label_field or display

        return coords[0], coords[1], iname, label_field or display

    def _is_location_query(self, name_hint: str) -> bool:
        """True если name_hint совпадает с известным Location item (или выглядит как имя)."""
        known = self._list_location_names()
        if known:
            q_t = _translit(name_hint)
            q_l = name_hint.lower()
            return any(
                max(
                    difflib.SequenceMatcher(None, q_t, _translit(k)).ratio(),
                    difflib.SequenceMatcher(None, q_l, k.lower()).ratio(),
                ) >= 0.5
                for k in known
            )
        # OpenHAB недоступен — считаем именем только если начинается с заглавной
        return name_hint[:1].isupper()

    def _list_location_names(self) -> list[str]:
        """Возвращает список имён всех Phone_*_location items."""
        with self._openhab_items_lock:
            items = list(self._openhab_items_cache)
        names = []
        for item in items:
            if item.get('type') != 'Location':
                continue
            iname = item.get('name', '')
            parts = iname.split('_')
            if (len(parts) >= 3
                    and parts[0].lower() == 'phone'
                    and parts[-1].lower() == 'location'):
                names.append('_'.join(parts[1:-1]))
        return names

    # ── Персональная идентификация ───────────────────────────────────────────

    def _lookup_person(self, telegram_id: int) -> dict | None:
        """Ищет человека по telegram_id в SQLite persons.

        Возвращает person_ctx совместимый с llm_node, или None.
        """
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            try:
                row = conn.execute(
                    'SELECT id, name, meet_count FROM persons WHERE telegram_id=?',
                    (telegram_id,),
                ).fetchone()
                if not row:
                    return None
                pid, name, meet_count = row
                notes_rows = conn.execute(
                    'SELECT key, value FROM person_notes WHERE person_id=?',
                    (pid,),
                ).fetchall()
                return {
                    'person_id':      pid,
                    'name':           name,
                    'meet_count':     meet_count or 1,
                    'current_emotion': '',
                    'notes':          {k: v for k, v in notes_rows},
                }
            finally:
                conn.close()
        except Exception as e:
            self.get_logger().warn(f'DB lookup ошибка: {e}')
            return None

    # ── Запрос к llm_node ────────────────────────────────────────────────────

    def _publish_ask(self, req_id: str, text: str, telegram_id: int | None) -> None:
        """Публикует /telegram_ask; person_ctx инжектируется из БД по telegram_id."""
        person_ctx = self._lookup_person(telegram_id) if telegram_id else None
        payload: dict = {'request_id': req_id, 'text': text}
        if person_ctx:
            payload['person_ctx'] = person_ctx
        pub_msg = String()
        pub_msg.data = json.dumps(payload, ensure_ascii=False)
        self._ask_pub.publish(pub_msg)
        self.get_logger().info(
            f'TG ask req={req_id[:8]}'
            + (f' person={person_ctx["name"]}' if person_ctx else ' (анонимно)')
        )

    # ── Вспомогательные ─────────────────────────────────────────────────────

    def _auth(self, chat_id: int) -> bool:
        return chat_id == self._allowed_chat_id

    def _capture_photo(self) -> bytes | None:
        """Берёт кадр из ROS-топика (vision pipeline) или напрямую с камеры."""
        with self._lock:
            sleeping = self._sleeping

        # В режиме SLEEP face_capture_node деактивирован → кэш устарел
        if not sleeping:
            with self._image_lock:
                cached = self._latest_jpeg
        else:
            cached = None

        if cached is not None:
            # Перекодируем в JPEG нужного качества если нужно
            arr = np.frombuffer(cached, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                ok, buf = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    self.get_logger().info('Фото: кадр из ROS-топика')
                    return bytes(buf)

        # Fallback: прямой захват с устройства (vision pipeline не запущен)
        cap = cv2.VideoCapture(self._cam_device)
        if not cap.isOpened():
            self.get_logger().warn(f'Камера недоступна: {self._cam_device}')
            return None
        try:
            ret, frame = cap.read()
            if not ret:
                self.get_logger().warn('cap.read() вернул False')
                return None
            ok, buf = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.get_logger().info('Фото: прямой захват с камеры')
            return bytes(buf) if ok else None
        finally:
            cap.release()

    # ── Геолокация ───────────────────────────────────────────────────────────

    async def _cmd_where(self, update: Update, context):
        """/where <имя> — прислать геолокацию члена семьи из OwnTracks."""
        if not self._auth(update.effective_chat.id):
            return
        name = ' '.join(context.args).strip() if context.args else ''
        if not name:
            await update.message.reply_text('Использование: /where <имя>')
            return
        await self._send_family_location(update, name)

    async def _send_family_location(self, update: Update, name: str):
        """Ищет Location item, отправляет pin на карте или сообщение об ошибке."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._find_person_location, name)

        if result is None:
            # Имя не совпало ни с одним Location item
            known = self._list_location_names()
            if known:
                await update.message.reply_text(
                    f'❓ Не нашёл «{name}».\nИзвестные: {", ".join(known)}')
            else:
                await update.message.reply_text(
                    '❓ Location items недоступны — OwnTracks работает?')
            return

        lat, lon, item_name, display = result
        if lat is None:
            # Item найден, но телефон офлайн или нет GPS-данных
            await update.message.reply_text(
                f'📡 {display}: телефон офлайн или нет GPS-данных')
            return

        await update.message.reply_text(f'📍 {display}:')
        await update.message.reply_location(latitude=lat, longitude=lon)
        self.get_logger().info(
            f'TG /where {name} → {display} ({lat:.5f}, {lon:.5f})')

    # ── Форматирование ROS статуса ───────────────────────────────────────────

    def _format_ros_status(self, lc: dict) -> list[str]:
        if not lc:
            return ['🌿 *ROS*: нет данных (lifecycle\\_manager не запущен?)']

        sys_state        = lc.get('system', 'unknown')
        recovering_nodes = set(lc.get('recovering_nodes', []))
        tiers            = lc.get('tiers', [])

        state_emoji = {
            'active':   '✅', 'degraded': '⚠️', 'sleep': '💤',
            'waking':   '🔄', 'starting': '⏳', 'fault':  '❌',
            'shutdown': '🔴',
        }.get(sys_state, '❓')

        total             = 0
        degraded_details  = []   # (tier_id, name, info)
        respawned_healthy = []   # (name, count)

        for tier in tiers:
            tier_id = tier.get('id', '?')
            for name, info in tier.get('nodes', {}).items():
                total += 1
                if info.get('degraded'):
                    degraded_details.append((tier_id, name, info))
                elif info.get('respawn_count', 0) > 0:
                    respawned_healthy.append((name, info.get('respawn_count', 0)))

        ok_count = total - len(degraded_details)
        lines    = []

        if not degraded_details:
            extra = f'  ⚡{len(respawned_healthy)} авторестарт(ов)' if respawned_healthy else ''
            lines.append(
                f'🌿 *ROS* {state_emoji} {sys_state} — {total} нод{extra}')
        else:
            lines.append(
                f'🌿 *ROS* {state_emoji} {sys_state.upper()} — {ok_count}/{total} OK')
            for _tid, nname, info in degraded_details:
                reason = info.get('degraded_reason', '')
                rc     = info.get('respawn_count', 0)
                is_crit = info.get('critical', False)
                marker  = '🔴' if is_crit else '🟡'

                if reason == 'watchdog':
                    reason_str = 'упала'
                elif reason == 'activation':
                    reason_str = 'не активировалась'
                elif reason.startswith('cascade_'):
                    reason_str = f'каскад T{reason.split("_")[1]}'
                else:
                    reason_str = reason or '?'

                parts = [f'  {marker} `{nname}` [{reason_str}]']
                if rc > 0:
                    parts.append(f'restarts:{rc}')
                if nname in recovering_nodes:
                    parts.append('🔄')
                lines.append(' '.join(parts))

        # Ноды с рестартами (но сейчас OK) — только если нет деградации
        if respawned_healthy and not degraded_details:
            for nname, count in respawned_healthy[:4]:
                lines.append(f'  ⚡ `{nname}` ×{count}')

        return lines

    # ── Telegram handlers ────────────────────────────────────────────────────

    @staticmethod
    def _http_check(url: str, timeout: float = 3.0) -> tuple:
        """Returns (ok: bool, latency_ms: float, err: str)."""
        t0 = time.time()
        try:
            with urllib.request.urlopen(url, timeout=timeout):
                return True, (time.time() - t0) * 1000, ''
        except Exception as e:
            return False, (time.time() - t0) * 1000, str(e)[:80]

    async def _cmd_restart_tts(self, update: Update, context):
        if not self._auth(update.effective_chat.id):
            return
        await update.message.reply_text('🔄 Перезапускаю cosyvoice_api...')
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ['docker', 'restart', 'cosyvoice_api'],
                    capture_output=True, text=True, timeout=30,
                ),
            )
            if result.returncode == 0:
                await update.message.reply_text(
                    '✅ cosyvoice_api перезапущен. Готов примерно через 30с.')
            else:
                await update.message.reply_text(
                    f'❌ docker restart вернул ошибку:\n{result.stderr[:300]}')
        except Exception as e:
            await update.message.reply_text(f'❌ Исключение: {e}')

    async def _cmd_status(self, update: Update, context):
        if not self._auth(update.effective_chat.id):
            return

        with self._lock:
            ctx      = dict(self._social_context)
            sleeping = self._sleeping
            lc       = dict(self._lifecycle_status)

        state   = ctx.get('state', 'unknown')
        present = ctx.get('person_present', False)
        pname   = ctx.get('name', '')
        emotion = ctx.get('emotion', 'neutral')

        # Идентификация пользователя
        loop   = asyncio.get_event_loop()
        person = await loop.run_in_executor(
            None, self._lookup_person, update.effective_chat.id)
        id_line = (f'👤 Привет, *{person["name"]}*!'
                   if person else '👤 Вы не в БД робота')

        # NUC: CPU / RAM
        cpu = psutil.cpu_percent(interval=0.3)
        ram = psutil.virtual_memory()
        ram_used_gb  = ram.used  / (1024 ** 3)
        ram_total_gb = ram.total / (1024 ** 3)
        load = os.getloadavg()

        # Диск
        disk = psutil.disk_usage('/')
        disk_used_gb  = disk.used  / (1024 ** 3)
        disk_total_gb = disk.total / (1024 ** 3)

        # Температура CPU (k10temp для AMD Ryzen)
        temp_str = ''
        try:
            temps = psutil.sensors_temperatures()
            for key in ('k10temp', 'coretemp', 'cpu_thermal', 'acpitz'):
                if key in temps and temps[key]:
                    entries = temps[key]
                    tmax = max(e.current for e in entries)
                    temp_str = f'  Темп: {tmax:.0f}°C'
                    break
        except Exception:
            pass

        # Аптайм системы
        uptime_sec = time.time() - psutil.boot_time()
        uptime_h   = int(uptime_sec // 3600)
        uptime_m   = int((uptime_sec % 3600) // 60)
        uptime_str = f'{uptime_h}ч {uptime_m}м'

        # TTS / LLM health (parallel, 3s timeout each)
        tts_primary_res, tts_local_res, llm_res = await asyncio.gather(
            loop.run_in_executor(
                None, self._http_check, 'http://192.168.10.118:8000/health', 3.0),
            loop.run_in_executor(
                None, self._http_check, 'http://localhost:8000/health', 3.0),
            loop.run_in_executor(
                None, self._http_check, 'http://192.168.10.118:11434/api/version', 3.0),
        )

        def _srv(label, ok, lat, _err):
            icon   = '✅' if ok else '❌'
            detail = f'{lat:.0f}мс' if ok else 'недоступен'
            return f'  {icon} {label}: {detail}'

        # ROS ноды
        ros_lines = self._format_ros_status(lc)

        robot_state = '💤 SLEEP' if sleeping else state.upper()
        person_line = (f'✅ {pname or "неизвестный"}' if present else '❌ нет')

        lines = [
            '*Статус InMoov*',
            id_line,
            '',
            f'Режим: `{robot_state}`',
            f'Человек: {person_line}  |  Эмоция: {emotion}',
            '',
            f'⚙️ *NUC* (uptime {uptime_str})',
            f'  CPU: {cpu:.0f}%  Load: {load[0]:.1f}/{load[1]:.1f}/{load[2]:.1f}',
            f'  RAM: {ram_used_gb:.1f}/{ram_total_gb:.0f} GB',
            f'  Диск: {disk_used_gb:.0f}/{disk_total_gb:.0f} GB ({disk.percent:.0f}%){temp_str}',
            '',
            '🖥️ *Серверы*',
            _srv('TTS RTX 3090', *tts_primary_res),
            _srv('TTS ROCm лок', *tts_local_res),
            _srv('LLM Ollama',   *llm_res),
            '',
        ] + ros_lines

        await update.message.reply_text(
            '\n'.join(lines), parse_mode=ParseMode.MARKDOWN)

    async def _cmd_photo(self, update: Update, context):
        if not self._auth(update.effective_chat.id):
            return
        await update.message.reply_text('📷 Снимаю...')
        loop = asyncio.get_event_loop()
        jpeg = await loop.run_in_executor(None, self._capture_photo)
        if jpeg is None:
            await update.message.reply_text('❌ Камера недоступна')
            return
        await update.message.reply_photo(photo=jpeg)

    async def _cmd_say(self, update: Update, context):
        if not self._auth(update.effective_chat.id):
            return
        text = ' '.join(context.args).strip()
        if not text:
            await update.message.reply_text('Использование: /say <текст>')
            return
        self._say_queue.put(text)
        await update.message.reply_text(f'🔊 Произношу: {text}')

    async def _cmd_ask(self, update: Update, context):
        if not self._auth(update.effective_chat.id):
            return
        text = ' '.join(context.args).strip()
        if not text:
            await update.message.reply_text('Использование: /ask <текст>')
            return
        await self._ask_and_reply(update, text)

    async def _cmd_wake(self, update: Update, context):
        if not self._auth(update.effective_chat.id):
            return
        msg = Bool()
        msg.data = False
        self._sleep_pub.publish(msg)
        await update.message.reply_text('☀️ Команда пробуждения отправлена')

    async def _cmd_sleep(self, update: Update, context):
        if not self._auth(update.effective_chat.id):
            return
        msg = Bool()
        msg.data = True
        self._sleep_pub.publish(msg)
        await update.message.reply_text('💤 Перехожу в спящий режим')

    async def _msg_handler(self, update: Update, context):
        if not self._auth(update.effective_chat.id):
            return
        text = (update.message.text or '').strip()
        if not text or text.startswith('/'):
            return
        # Быстрый путь: "где [сейчас] Имя?" → геолокация без LLM
        # Проверяем совпадение с известными именами, чтобы не перехватывать
        # вопросы вида "где сейчас включён свет?"
        m = _WHERE_RE.search(text)
        if m and self._is_location_query(m.group(1)):
            await self._send_family_location(update, m.group(1))
            return
        await self._ask_and_reply(update, text)

    async def _ask_and_reply(self, update: Update, text: str):
        """Отправляет запрос через llm_node и стримит ответ редактированием сообщения."""
        sent = await update.message.reply_text('⏳ Думаю...')

        req_id = str(uuid.uuid4())
        q: asyncio.Queue = asyncio.Queue()

        with self._pending_lock:
            self._pending[req_id] = q

        # Публикуем запрос в llm_node (синхронно, быстро)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._publish_ask, req_id, text, update.effective_chat.id)

        accumulated = ''
        last_edit = 0.0
        THROTTLE   = 1.2   # минимальный интервал правок (Telegram rate limit)

        try:
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=self._llm_timeout)
                except asyncio.TimeoutError:
                    final = accumulated or '⏱ Таймаут — LLM не ответил'
                    await sent.edit_text(final)
                    return

                chunk   = data.get('text', '').lstrip('ᐈ').strip()
                partial = data.get('partial', True)

                if chunk == '__busy__':
                    await sent.edit_text(
                        '⚙️ Робот сейчас разговаривает — попробуй через несколько секунд')
                    return

                if not partial:
                    # Финал — это сигнал "стоп", а не новый текст.
                    # Весь контент уже пришёл через partial=True чанки.
                    # chunk содержит error-текст только когда accumulated пустой.
                    final = accumulated if accumulated else chunk
                    await sent.edit_text(final or '🤔 Нет ответа')
                    return

                # partial=True: добавляем чанк с пробелом-разделителем
                if chunk:
                    accumulated += (' ' if accumulated else '') + chunk

                now = asyncio.get_event_loop().time()
                if now - last_edit >= THROTTLE:
                    await sent.edit_text(accumulated + ' ▌')
                    last_edit = now

        finally:
            with self._pending_lock:
                self._pending.pop(req_id, None)

    # ── Telegram event loop ──────────────────────────────────────────────────

    def _run_telegram_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._run_bot())

    async def _push_sender(self, app):
        """Дренирует _push_queue и отправляет сообщения в Telegram.

        Каждый элемент очереди — dict {"text": ..., "parse_mode": ...}.
        parse_mode берётся из payload; если не указан — plain text (None).
        """
        while rclpy.ok():
            try:
                data = await asyncio.wait_for(self._push_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            text       = data.get('text', '') if isinstance(data, dict) else str(data)
            parse_mode = data.get('parse_mode') if isinstance(data, dict) else None
            if not text:
                continue
            try:
                await app.bot.send_message(
                    chat_id=self._allowed_chat_id,
                    text=text,
                    parse_mode=parse_mode,
                )
                self.get_logger().info(f'TG push отправлен: {text[:80]}')
            except Exception as e:
                self.get_logger().warn(f'TG push ошибка: {e}')

    async def _run_bot(self):
        app = Application.builder().token(self._token).build()

        asyncio.create_task(self._push_sender(app))

        app.add_handler(CommandHandler('status',      self._cmd_status))
        app.add_handler(CommandHandler('photo',       self._cmd_photo))
        app.add_handler(CommandHandler('say',         self._cmd_say))
        app.add_handler(CommandHandler('ask',         self._cmd_ask))
        app.add_handler(CommandHandler('wake',        self._cmd_wake))
        app.add_handler(CommandHandler('sleep',       self._cmd_sleep))
        app.add_handler(CommandHandler('where',       self._cmd_where))
        app.add_handler(CommandHandler('restart_tts', self._cmd_restart_tts))
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._msg_handler))

        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        self.get_logger().info('Telegram polling запущен')

        while rclpy.ok():
            await asyncio.sleep(1.0)

        self.get_logger().info('Telegram: завершение...')
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = TelegramBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
