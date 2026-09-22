#!/usr/bin/env python3
"""
telegram_bridge_node.py
========================
Telegram ↔ ROS2 bridge for remote control of the InMoov robot.

/ask and free-form text are routed through llm_node (/telegram_ask → /telegram_response),
so all tool calls, smart home control and personal memory work fully.

Identification: by telegram_id in the persons table (filled in manually).
On a match, person_ctx is injected into llm_node so the robot knows who it's talking to.

Security:
  - Only allowed_chat_id receives replies (everyone else is silently ignored)
  - Token comes only from the TELEGRAM_BOT_TOKEN env var

Commands:
  /status          — mode, person present, CPU/RAM, TTS/LLM server health
  /photo           — snapshot from the left camera → JPEG
  /say <text>      — TTS via the Speak action (no LLM)
  /ask <text>      — through llm_node → full pipeline
  /wake            — /robot_sleep False
  /sleep           — /robot_sleep True (instant transition to SLEEP, no LLM)
  /restart_tts     — restarts the local cosyvoice_api Docker container
  <any text>       — same as /ask

Topics:
  /social_context  (in)   — identity_manager state
  /robot_sleep     (in/out latched)
  /telegram_ask    (out)  — request to llm_node
  /telegram_response (in) — response from llm_node
  /telegram_push   (in)   — push notifications from other nodes (JSON: {"text": "..."})
Actions:
  speak — inmoov_msgs/action/Speak → tts_node

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import asyncio
import base64
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


# ── Geolocation helpers ──────────────────────────────────────────────────────

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
    """Transliterates Cyrillic → Latin for fuzzy name matching."""
    return ''.join(_TRANSLIT_MAP.get(c, c) for c in s.lower())

def _parse_location_state(state: str) -> tuple[float, float] | None:
    """Parses an OpenHAB Location item state: 'lat,lon[,alt]' → (lat, lon)."""
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


# ── Main class ───────────────────────────────────────────────────────────────

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
        """Safe declare_parameter: ignores re-declaration on re-configure."""
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
            self.get_logger().warn('Telegram bridge: no token or chat_id — bot not started')
            return TransitionCallbackReturn.SUCCESS

        self._loop       = asyncio.new_event_loop()
        # Recreate the Queue for the new event loop — the old Queue is bound to
        # the previous loop and raises RuntimeError on re-activation.
        self._push_queue = asyncio.Queue()
        self._tg_thread  = threading.Thread(
            target=self._run_telegram_loop, daemon=True, name='telegram_loop')
        self._tg_thread.start()
        self.get_logger().info(
            f'Telegram bridge activated. allowed_chat_id={self._allowed_chat_id}')
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
        """Caches the latest compressed frame from face_capture_node."""
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
        """Receives a partial/final response from llm_node, puts it on the asyncio.Queue."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        req_id = data.get('request_id', '')
        with self._pending_lock:
            q = self._pending.get(req_id)
        if q and hasattr(self, '_loop'):
            # thread-safe push into the asyncio event loop
            self._loop.call_soon_threadsafe(q.put_nowait, data)

    def _process_say_queue(self):
        """100ms timer: /say text → Speak ActionClient."""
        while not self._say_queue.empty():
            try:
                text = self._say_queue.get_nowait()
            except queue.Empty:
                break
            if not self._speak_client.wait_for_server(timeout_sec=0.5):
                self.get_logger().warn('Speak action server unavailable')
                continue
            goal = Speak.Goal()
            goal.text = text
            self._speak_client.send_goal_async(goal)
            self.get_logger().info(f'TG /say → TTS: "{text[:60]}"')

    def _telegram_push_cb(self, msg: String):
        """Push notification from another node (openhab_bridge, memory_node) → Telegram.

        Expects JSON: {"text": "...", "parse_mode": "HTML"} (parse_mode optional).
        Without parse_mode → plain text (Telegram does not interpret tags/special chars).
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            data = {'text': msg.data}
        text = data.get('text', '').strip()
        if not text or not self._allowed_chat_id:
            return
        if hasattr(self, '_loop'):
            # Pass the whole dict along to preserve the sender's parse_mode
            self._loop.call_soon_threadsafe(self._push_queue.put_nowait, data)

    def _openhab_items_cb(self, msg: String):
        """Caches items from openhab_bridge_node (needed for geolocation)."""
        try:
            items = json.loads(msg.data)
            with self._openhab_items_lock:
                self._openhab_items_cache = items
        except Exception:
            pass

    def _find_person_location(
        self, name_hint: str,
    ) -> tuple[float, float, str, str] | None:
        """Looks up a Location item by name (fuzzy, Cyrillic/Latin).

        Returns (lat, lon, item_name, display_name) or None.
        None with reason 'no_fix' if the item was found but has no GPS data.
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
            # Format: Phone_Nastja_location
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
            # Item found, but the phone is offline / has no GPS fix
            return None, None, iname, label_field or display

        return coords[0], coords[1], iname, label_field or display

    def _is_location_query(self, name_hint: str) -> bool:
        """True if name_hint matches a known Location item (or looks like a name)."""
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
        # OpenHAB unavailable — treat as a name only if it starts with a capital letter
        return name_hint[:1].isupper()

    def _list_location_names(self) -> list[str]:
        """Returns the list of names of all Phone_*_location items."""
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

    # ── Personal identification ──────────────────────────────────────────────

    def _lookup_person(self, telegram_id: int) -> dict | None:
        """Looks up a person by telegram_id in the SQLite persons table.

        Returns a person_ctx compatible with llm_node, or None.
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
            self.get_logger().warn(f'DB lookup error: {e}')
            return None

    # ── Request to llm_node ──────────────────────────────────────────────────

    def _publish_ask(self, req_id: str, text: str, telegram_id: int | None,
                      image_base64: str | None = None) -> None:
        """Publishes /telegram_ask; person_ctx is injected from the DB by telegram_id."""
        person_ctx = self._lookup_person(telegram_id) if telegram_id else None
        payload: dict = {'request_id': req_id, 'text': text}
        if person_ctx:
            payload['person_ctx'] = person_ctx
        if image_base64:
            payload['image_base64'] = image_base64
        pub_msg = String()
        pub_msg.data = json.dumps(payload, ensure_ascii=False)
        self._ask_pub.publish(pub_msg)
        self.get_logger().info(
            f'TG ask req={req_id[:8]}'
            + (f' person={person_ctx["name"]}' if person_ctx else ' (anonymous)')
            + (' +photo' if image_base64 else '')
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _auth(self, chat_id: int) -> bool:
        return chat_id == self._allowed_chat_id

    def _capture_photo(self) -> bytes | None:
        """Grabs a frame from the ROS topic (vision pipeline) or directly from the camera."""
        with self._lock:
            sleeping = self._sleeping

        # In SLEEP mode face_capture_node is deactivated → the cache is stale
        if not sleeping:
            with self._image_lock:
                cached = self._latest_jpeg
        else:
            cached = None

        if cached is not None:
            # Re-encode to the desired JPEG quality if needed
            arr = np.frombuffer(cached, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                ok, buf = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    self.get_logger().info('Photo: frame from ROS topic')
                    return bytes(buf)

        # Fallback: direct capture from the device (vision pipeline not running)
        cap = cv2.VideoCapture(self._cam_device)
        if not cap.isOpened():
            self.get_logger().warn(f'Camera unavailable: {self._cam_device}')
            return None
        try:
            ret, frame = cap.read()
            if not ret:
                self.get_logger().warn('cap.read() returned False')
                return None
            ok, buf = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.get_logger().info('Photo: direct camera capture')
            return bytes(buf) if ok else None
        finally:
            cap.release()

    # ── Geolocation ───────────────────────────────────────────────────────────

    async def _cmd_where(self, update: Update, context):
        """/where <name> — sends a family member's geolocation from OwnTracks."""
        if not self._auth(update.effective_chat.id):
            return
        name = ' '.join(context.args).strip() if context.args else ''
        if not name:
            await update.message.reply_text('Использование: /where <имя>')
            return
        await self._send_family_location(update, name)

    async def _send_family_location(self, update: Update, name: str):
        """Looks up a Location item, sends a map pin or an error message."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._find_person_location, name)

        if result is None:
            # Name didn't match any Location item
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
            # Item found, but the phone is offline or has no GPS data
            await update.message.reply_text(
                f'📡 {display}: телефон офлайн или нет GPS-данных')
            return

        await update.message.reply_text(f'📍 {display}:')
        await update.message.reply_location(latitude=lat, longitude=lon)
        self.get_logger().info(
            f'TG /where {name} → {display} ({lat:.5f}, {lon:.5f})')

    # ── ROS status formatting ─────────────────────────────────────────────────

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

        # Nodes with restarts (but currently OK) — only if nothing is degraded
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

        # User identification
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

        # Disk
        disk = psutil.disk_usage('/')
        disk_used_gb  = disk.used  / (1024 ** 3)
        disk_total_gb = disk.total / (1024 ** 3)

        # CPU temperature (k10temp for AMD Ryzen)
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

        # System uptime
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
                None, self._http_check, 'http://192.168.10.118:18020/health', 3.0),
        )

        def _srv(label, ok, lat, _err):
            icon   = '✅' if ok else '❌'
            detail = f'{lat:.0f}мс' if ok else 'недоступен'
            return f'  {icon} {label}: {detail}'

        # ROS nodes
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
            _srv('LLM vLLM',     *llm_res),
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
        # Fast path: "where [is] Name?" → geolocation without the LLM
        # Check the match against known names, so as not to intercept
        # questions like "where is the light on right now?"
        m = _WHERE_RE.search(text)
        if m and self._is_location_query(m.group(1)):
            await self._send_family_location(update, m.group(1))
            return
        await self._ask_and_reply(update, text)

    async def _photo_handler(self, update: Update, context):
        """A photo from the user — download it, encode to base64, hand it to the LLM
        (vision via the same qwen3.8-27b/vLLM, see llm_node._query_llm)."""
        if not self._auth(update.effective_chat.id):
            return
        caption = (update.message.caption or '').strip()
        photo = update.message.photo[-1]  # highest available resolution
        try:
            tg_file = await context.bot.get_file(photo.file_id)
            raw = await tg_file.download_as_bytearray()
        except Exception as e:
            self.get_logger().warn(f'TG photo: failed to download the file: {e}')
            await update.message.reply_text('❌ Не смог скачать фото')
            return
        image_b64 = base64.b64encode(bytes(raw)).decode()
        text = caption or 'Что на этой фотографии?'
        await self._ask_and_reply(update, text, image_base64=image_b64)

    async def _ask_and_reply(self, update: Update, text: str,
                              image_base64: str | None = None):
        """Sends the request via llm_node and streams the reply by editing the message."""
        sent = await update.message.reply_text(
            '👀 Смотрю...' if image_base64 else '⏳ Думаю...')

        req_id = str(uuid.uuid4())
        q: asyncio.Queue = asyncio.Queue()

        with self._pending_lock:
            self._pending[req_id] = q

        # Publish the request to llm_node (synchronous, fast)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._publish_ask, req_id, text, update.effective_chat.id, image_base64)

        accumulated = ''
        last_edit = 0.0
        THROTTLE   = 1.2   # minimum interval between edits (Telegram rate limit)

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
                    # The final message is a "stop" signal, not new text.
                    # All the content already arrived via partial=True chunks.
                    # chunk contains error text only when accumulated is empty.
                    final = accumulated if accumulated else chunk
                    await sent.edit_text(final or '🤔 Нет ответа')
                    return

                # partial=True: append the chunk with a space separator
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
        """Drains _push_queue and sends messages to Telegram.

        Each queue element is a dict {"text": ..., "parse_mode": ...}.
        parse_mode comes from the payload; if not given — plain text (None).
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
                self.get_logger().info(f'TG push sent: {text[:80]}')
            except Exception as e:
                self.get_logger().warn(f'TG push error: {e}')

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
        app.add_handler(MessageHandler(filters.PHOTO, self._photo_handler))
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._msg_handler))

        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        self.get_logger().info('Telegram polling started')

        while rclpy.ok():
            await asyncio.sleep(1.0)

        self.get_logger().info('Telegram: shutting down...')
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
