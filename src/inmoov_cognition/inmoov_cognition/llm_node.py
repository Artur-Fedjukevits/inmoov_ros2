#!/usr/bin/env python3

"""
llm_node.py  (v2 — Intent Generator)
=========================================
LLM node — OpenAI-compatible chat.completions API (vLLM), function calling (tools API).

Principle: the LLM does NOT control TTS and servos directly.
It publishes an intent to /llm_response (text + voice style).
The Behavior Tree orchestrates Speak + Gesticulation in parallel; facial
expression is now synchronized with the actual speech duration — held by
tts_node (see /face_expression_hold), not by the BT.

Tools:
  - items_control        — control OpenHAB devices
  - get_openhab_states   — get current device state (from cache)
  - search_openhab_items — find devices by room/type/state
  - robot_control        — physical robot commands (→ /robot_events → BT)
  - web_search           — internet search (→ /robot_events → BT)
  - set_voice_style      — buffers a voice preset (+ synced facial
                            expression for the duration of speech) for /llm_response
  - save_memory          — save to SQLite via /memory/query
  - search_memory        — search memory via /memory/query

Topics:
  /voice_command   (in)  String — text from Whisper STT
  /voice_command_other (in) String — STT of a phrase by someone other than the
                         interlocutor (SV-rejected); answered only if it names the robot
  /speaker_evidence (in) String JSON — identity_manager lips+gaze per phrase: vetoes
                         a gaze-only address when the face in view did not speak
  /voice/sv_confirm (out) String JSON {segment_id} — an SV-rejected phrase the lips
                         showed was the interlocutor's → voice_detector adds its voice
  /wake_detected   (in)  Bool   — wake word; opens the addressee gate while no face is in view
  /llm_response    (out) String JSON {text, voice_instruct} → BT Blackboard
  /robot_events    (out) String JSON — physical commands (move/arm/head/sleep/search) → BT
  /search_result   (in)  String — search result from behavior_manager
  /openhab_schema  (in)  String — static device schema from openhab_bridge_node
  /openhab_items   (in)  String — current device states
  - broadcast_message     — WAV synthesis via TTS + broadcast to the living-room Chromecast

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import base64
import collections
import concurrent.futures
import datetime
import json
import re
import shutil
import threading
import time
import uuid
import requests

import rclpy
from rclpy.action import ActionClient
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import CompressedImage
from inmoov_msgs.action import Speak
from inmoov_msgs.srv import MemoryQuery

from inmoov_cognition.dialogue_guards import (
    PendingUtterance, ToolAuditLog, filter_tools, parse_tool_list)


# ── Tools (OpenAI tools API) ──────────────────────────────────────────
TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': (
                'Get weather forecast from yr.no (Norwegian Meteorological Institute). '
                'ALWAYS use this tool for ANY weather question — never use web_search for weather. '
                'Default location: Bødalen, Asker (Norway). Default date: today.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'location': {
                        'type': 'string',
                        'description': (
                            'Location name in English or Norwegian. '
                            'Default (if not specified by user): "Bødalen, Asker". '
                            'Examples: "Oslo", "Bergen", "Asker".'
                        ),
                    },
                    'date': {
                        'type': 'string',
                        'description': (
                            'Date: "today", "tomorrow", or YYYY-MM-DD. '
                            'Default: "today".'
                        ),
                    },
                    'speak_text': {
                        'type': 'string',
                        'description': 'NOT used — weather tool always requires LLM interpretation.',
                    },
                },
                'required': [],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'items_control',
            'description': (
                'Control or update state of a SINGLE device/item in openHAB smart home. '
                'The "name" parameter MUST be an exact item name from the schema — never invent names. '
                'To control lights/heaters in a room: first call search_openhab_items to find '
                'the exact names, then call items_control for each item found. '
                'For a single known device: call items_control directly.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'name':       {'type': 'string', 'description': 'Item name in openHAB'},
                    'type':       {'type': 'string', 'description': 'Item type (Switch, Dimmer, Number, String, Color)'},
                    'state':      {'type': 'string', 'description': 'New state. Switch: ON/OFF. Dimmer: 0-100. Number:Temperature: value in degrees C. String: mode value.'},
                    'speak_text': {'type': 'string', 'description': 'What to say to the user after execution (1-2 sentences in Russian). Always provide this.'},
                },
                'required': ['name', 'type', 'state'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_openhab_states',
            'description': (
                'Get current state of one or more OpenHAB items by name. '
                'Use when the user asks about current status of specific devices.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'names': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'List of item names to query',
                    },
                },
                'required': ['names'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'search_openhab_items',
            'description': (
                'Search OpenHAB items by semantic group, state or name/room substring. '
                'Use for questions like "which lights are on", "temperatures in all rooms", '
                '"what heaters are on", "who is home". '
                'AT LEAST ONE filter is required — calling with no parameters is not allowed '
                'and will return an error. Always specify group_filter and/or name_contains.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'group_filter': {
                        'type': 'string',
                        'description': (
                            'Semantic group name (preferred over type_filter): '
                            'AllLights, All_Heaters, Floor_Heaters, TempSensors, '
                            'HumiditySensors, TargetTemp, PhonesAtHome. Empty = any group.'
                        ),
                    },
                    'state_filter': {
                        'type': 'string',
                        'description': (
                            'Filter by state: "ON" (Switch=ON or Dimmer/Color>0), '
                            '"OFF" (Switch=OFF or Dimmer=0), ">0" (non-zero numeric). '
                            'Empty = any state.'
                        ),
                    },
                    'name_contains': {
                        'type': 'string',
                        'description': (
                            'Substring to match in item name or label (case-insensitive). '
                            'Use room names like "Bedroom", "Hall", "Kitchen". Empty = any.'
                        ),
                    },
                },
                'required': [],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'robot_control',
            'description': (
                'Control the physical robot: arms, head (with optional torso assist), enter sleep '
                'mode, or say goodbye. Use action=sleep when user says "выключись", "иди спать", '
                '"спать", "отдыхай" etc. Use action=goodbye when user says "пока", "до свидания", '
                '"увидимся", "прощай" etc. — say a farewell phrase and end the conversation session. '
                'Sleep mode: robot goes quiet, disables vision/PIR, only wakeword wakes it. '
                'For "посмотри туда-то, что ты видишь?" style requests, use look_direction instead — '
                'it turns AND takes/describes a photo in the correct order; robot_control(action=head) '
                'alone only turns, it does not look at anything.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'action':     {'type': 'string', 'enum': ['arm', 'head', 'status', 'sleep', 'goodbye']},
                    'command':    {'type': 'string', 'description': 'For arm: grab|release|home|extend|retract'},
                    'target':     {'type': 'string', 'description': 'For arm: object description'},
                    'pan':        {'type': 'number', 'description': 'For head: -90..90 degrees, positive = right'},
                    'tilt':       {'type': 'number', 'description': 'For head: -45..45 degrees'},
                    'scope': {
                        'type': 'string',
                        'enum': ['head', 'partial', 'full'],
                        'description': (
                            'For head: how much torso to add to the head turn. '
                            '"head" (default) — an ordinary glance/orientation; for |pan| >= 20 the '
                            'torso follows ~30% automatically so head and torso face the same way. '
                            '"partial" — head + ~30% torso rotation same direction as pan; use when '
                            'the person explicitly says they are standing/sitting off to that side '
                            'and a head-only turn will not be enough to bring them into camera view. '
                            '"full" — head + full torso rotation; use for "обернись", "повернись '
                            'полностью", "посмотри что сзади".'
                        ),
                    },
                    'query':      {'type': 'string', 'description': 'For status: battery|position|all'},
                    'text':       {'type': 'string', 'description': 'For sleep/goodbye: farewell phrase (goes to TTS via speak_text, do NOT duplicate in speak_text)'},
                    'speak_text': {'type': 'string', 'description': 'What to say after execution (1-2 sentences in Russian). For sleep/goodbye: use text field instead, leave speak_text empty.'},
                },
                'required': ['action'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'web_search',
            'description': 'Search the internet for current information, news, weather, facts.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'Search query in Russian'},
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'save_memory',
            'description': (
                'Save LASTING FACTS about the current person or household to long-term memory. '
                'Use ONLY when the user reveals a stable personal fact: job, age, hobby, pet, health issue, '
                'family member name+relationship, preference, habit, or household fact. '
                'NEVER use for reminders or scheduled tasks — use set_reminder instead. '
                'Do NOT use for: one-time requests ("передай привет", "включи свет"), '
                'tasks to perform, conversational phrases, or anything that is not a lasting fact. '
                'Example of what TO save: "я работаю врачом", "у меня есть кот Барсик", "мне 35 лет". '
                'Example of what NOT to save: "передай привет маме", "какая погода", '
                '"напомни мне постричь газон" (→ это set_reminder!).'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'person_id': {
                        'type':        'integer',
                        'description': 'Person ID from person context. Omit for general robot knowledge.',
                    },
                    'key': {
                        'type':        'string',
                        'description': 'Key in Russian snake_case, e.g. "работа", "семья", "хобби", "возраст", "питомцы"',
                    },
                    'value': {
                        'type':        'string',
                        'description': 'Value to store (short sentence in Russian)',
                    },
                    'speak_text': {
                        'type':        'string',
                        'description': 'What to say to the user — natural continuation of the conversation (1-2 sentences in Russian). Do NOT say "Запомнил" or "Сохранил" — just continue the dialogue naturally. Always provide this.',
                    },
                },
                'required': ['key', 'value', 'speak_text'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'search_memory',
            'description': (
                'Search long-term semantic memory for facts about people, events, or rules. '
                'Use when the user asks about something you should have remembered from previous conversations: '
                'their job, hobbies, preferences, family members, health info, etc. '
                'Do NOT use for current-session context (already in system prompt). '
                'Examples: "does Artur have a cat?", "what job does this person have?"'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Semantic search query in Russian',
                    },
                    'category': {
                        'type': 'string',
                        'enum': ['person', 'preference', 'event', 'rule', ''],
                        'description': 'Optional category filter (empty = all)',
                    },
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'set_reminder',
            'description': (
                'ЕДИНСТВЕННЫЙ инструмент для создания напоминаний. '
                'ОБЯЗАТЕЛЬНО используй когда пользователь говорит слова '
                '"напомни", "не забудь напомнить", "напоминалка", "напомни мне". '
                'НИКОГДА не используй save_memory для напоминаний — только set_reminder. '
                'Также используй при важных событиях с датой '
                '(день рождения, концерт, выступление, встреча, экзамен, операция). '
                'Без даты — показывается при каждой следующей встрече. '
                'С датой (и опционально временем) — показывается начиная с указанной даты/времени. '
                'Примеры: "напомни постричь газон в субботу" → set_reminder(date=ближайшая_суббота); '
                '"напомни позвонить маме" → set_reminder(date=None); '
                '"напомни в 9 утра завтра" → set_reminder(date="завтра", time="09:00").'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'person_id': {
                        'type': 'integer',
                        'description': 'ID пользователя из person context.',
                    },
                    'person_name': {
                        'type': 'string',
                        'description': 'Имя пользователя.',
                    },
                    'message': {
                        'type': 'string',
                        'description': (
                            'Текст напоминания — как Лёня скажет при встрече. '
                            'Пример: "Привет, Артур! Ты хотел постричь газон — не забыл?" '
                            'Пиши от первого лица Лёни, персонализированно и коротко.'
                        ),
                    },
                    'date': {
                        'type': 'string',
                        'description': (
                            'Дата напоминания. Форматы:\n'
                            '• Название дня недели — "понедельник", "суббота" и т.п. '
                            '  Система САМА вычислит ближайший такой день. НЕ вычисляй дату сам!\n'
                            '• "завтра" / "послезавтра"\n'
                            '• Конкретная дата YYYY-MM-DD — только если пользователь '
                            '  назвал точное число (например, "пятнадцатого июня").\n'
                            '  В этом случае ставь дату за 1 день до события.\n'
                            '• Пусто — напомнить при следующей встрече.'
                        ),
                    },
                    'time': {
                        'type': 'string',
                        'description': (
                            'Время напоминания в формате HH:MM (24ч). '
                            'Указывай только если пользователь назвал конкретное время. '
                            'Примеры: "в девять утра" → "09:00", "в полдень" → "12:00", '
                            '"в 21:30" → "21:30". '
                            'Если время не указано — оставь пустым (используется время по умолчанию).'
                        ),
                    },
                    'speak_text': {
                        'type': 'string',
                        'description': 'Что сказать пользователю после сохранения (1-2 предложения).',
                    },
                },
                'required': ['message'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'confirm_reminder',
            'description': (
                'Удалить напоминание после того как пользователь подтвердил получение. '
                'Вызывай когда пользователь явно говорит "понял", "помню", "ок", "спасибо" '
                'в ответ на напоминание показанное в приветствии.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'person_id': {
                        'type': 'integer',
                        'description': 'ID пользователя из person context.',
                    },
                    'speak_text': {
                        'type': 'string',
                        'description': 'Что сказать пользователю (1-2 предложения).',
                    },
                },
                'required': ['person_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'set_voice_style',
            'description': (
                'Set voice preset for the next speech response. Also makes the robot '
                'physically show the matching face expression for the ENTIRE duration '
                'of that speech (not just a flash) — voice and face always change together. '
                'The TTS server (OmniVoice) uses cloned-voice presets recorded in advance — '
                'free-form tone instructions are NOT supported (voice cloning overrides them). '
                'Only 4 presets exist. Use when the conversation context warrants an '
                'emotional/tonal reaction. Examples: good news → happy; bad news → sad; '
                'unexpected → surprise. Reset to normal → "" or "neutral".'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'style': {
                        'type': 'string',
                        'enum': ['neutral', 'happy', 'sad', 'surprise', ''],
                        'description': 'Voice preset name. Empty string = "neutral" (default).',
                    },
                },
                'required': [],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'broadcast_message',
            'description': (
                'Синтезирует голосовое сообщение через TTS и транслирует его на Chromecast-колонку '
                'в гостиной. Используй когда пользователь говорит "передай на колонку", '
                '"скажи в гостиной", "объяви дома", "передай сообщение" и т.п. '
                'НЕ используй items_control для LivingRoom_Chromecast — только broadcast_message.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'text': {
                        'type': 'string',
                        'description': 'Текст для синтеза и воспроизведения на колонке.',
                    },
                    'volume': {
                        'type': 'integer',
                        'description': 'Громкость колонки 0-100. По умолчанию 80.',
                    },
                    'speak_text': {
                        'type': 'string',
                        'description': 'Что сказать пользователю после отправки (1-2 предложения).',
                    },
                },
                'required': ['text'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'merge_persons',
            'description': (
                'Слить два профиля в один — когда робот ошибочно создал дубликат человека. '
                'Используй если пользователь говорит "ты создал меня под другим именем", '
                '"это был я Артур", "объедини нас", "удали дубликат" и т.п. '
                'Выполняет проверку сходства лиц перед слиянием.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'duplicate_name': {
                        'type': 'string',
                        'description': 'Имя дубликата (профиль который нужно удалить).',
                    },
                    'target_name': {
                        'type': 'string',
                        'description': 'Правильное имя (профиль который нужно сохранить).',
                    },
                    'speak_text': {
                        'type': 'string',
                        'description': 'Что сказать пользователю (1-2 предложения).',
                    },
                },
                'required': ['duplicate_name', 'target_name'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'look_and_describe',
            'description': (
                'Сделать снимок с камеры в глазу робота ПРЯМО СЕЙЧАС, без поворота, и '
                'проанализировать его через vision-модель — для вопросов, требующих реально '
                'посмотреть и понять, что в кадре ПРЯМО ПЕРЕД РОБОТОМ (YOLO-детекция в блоке '
                '"Сцена ПРЯМО СЕЙЧАС" знает только ограниченный набор предметов и не годится для '
                'этого). Используй для: "что это?", "что у меня в руках?", "посмотри", "что ты '
                'видишь?", "что там на столе?" и любых похожих вопросов БЕЗ указания стороны/поворота. '
                'Если нужно посмотреть В СТОРОНУ (направо/налево/вверх/вниз/назад) — используй '
                'look_direction, а не этот инструмент. '
                'НЕ используй robot_control(action="status") для таких вопросов.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': (
                            'Что именно нужно рассмотреть — конкретная формулировка вопроса '
                            'пользователя, например "что у меня в руках" или "что за окном". '
                            'Если неясно — оставь пустым, будет общее описание кадра.'
                        ),
                    },
                },
                'required': [],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'look_direction',
            'description': (
                'Повернуть голову (и при необходимости корпус) в указанную сторону, ДОЖДАТЬСЯ '
                'реального завершения поворота (~2с) и только потом сделать снимок и '
                'проанализировать через vision-модель. '
                'ЕДИНСТВЕННЫЙ инструмент для запросов вида "посмотри направо/налево/вверх/вниз, '
                'что ты видишь?", "обернись, что там?", "посмотри что у меня за спиной", '
                '"глянь налево" — то есть когда нужно И повернуться, И описать увиденное. '
                'НЕ вызывай для этого robot_control(action=head) + look_and_describe по отдельности — '
                'они выполняются параллельно и снимок улетит раньше, чем голова довернётся. '
                'Если нужно просто повернуться без описания — используй robot_control(action=head). '
                'Если нужно посмотреть на то, что ПРЯМО СЕЙЧАС перед роботом, без поворота — '
                'используй look_and_describe.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'pan': {
                        'type': 'number',
                        'description': (
                            '-90..90 градусов, положительное = направо. Обязательно ненулевое '
                            'значение при scope=partial/full. Для "обернись" без уточнения стороны '
                            'бери ±70.'
                        ),
                    },
                    'tilt': {'type': 'number', 'description': '-45..45 градусов, положительное = вверх. По умолчанию 0.'},
                    'scope': {
                        'type': 'string',
                        'enum': ['head', 'partial', 'full'],
                        'description': (
                            '"head" (по умолчанию) — обычный взгляд в сторону; при |pan| >= 20 корпус '
                            'автоматически доворачивается на ~30% туда же, голова и корпус всегда смотрят в одну сторону. '
                            '"partial" — голова + ~30% поворота корпуса туда же; используй, когда '
                            'собеседник явно сказал что стоит/сидит сбоку и одной головы не хватит, '
                            'чтобы он попал в кадр. '
                            '"full" — голова + корпус полностью; для "обернись"/"повернись полностью".'
                        ),
                    },
                    'query': {
                        'type': 'string',
                        'description': (
                            'Что именно нужно рассмотреть после поворота — конкретная формулировка '
                            'вопроса пользователя. Если неясно — оставь пустым, будет общее описание.'
                        ),
                    },
                },
                'required': ['pan'],
            },
        },
    },
]


# Regex for LLM stage directions: (Тихо, с улыбкой) / (Шёпотом) etc.
_STAGE_DIR_RE = re.compile(r'\([А-ЯЁа-яё][^)]{0,60}\)')

# CJK ideographs + CJK punctuation/symbols (Qwen sometimes inserts Chinese text)
_CJK_RE = re.compile(r'[　-鿿豈-￯\U00020000-\U0002a6df]+')

# Qwen3 sometimes returns tool calls as text instead of via the tool_calls API field.
# Format 1: "ᐈ\n{...}" or "<tool_call>{...}</tool_call>"
# Format 2: "<tools>\n{...}\n{...}\n</tools>" (multiple calls)
# IMPORTANT: greedy .*  (not .*?) is needed for nested JSON {"arguments": {...}}
# A non-greedy match would stop at the first } → json.loads would fail → tool call not extracted
# ᐈ[^{]* — Qwen3 sometimes inserts junk between ᐈ and the JSON (e.g. "ᐈC\n{...}"),
# so we match ᐈ + any non-{ characters before the opening brace
_TEXT_TOOL_CALL_RE = re.compile(
    r'(?:ᐈ[^{]*|<tool_call>)\s*(\{.*\})\s*(?:</tool_call>)?',
    re.DOTALL,
)
_TOOLS_BLOCK_RE = re.compile(r'<tools>(.*?)</tools>', re.DOTALL)

# Streaming sentence splitter: period/!/?/… + whitespace or end of line
_SENT_SPLIT_RE    = re.compile(r'(?<=[.!?…])\s+|(?<=[.!?…])$', re.MULTILINE)
# Fallback split on comma/semicolon/colon when the sentence is too long
_COMMA_SPLIT_RE   = re.compile(r'(?<=[,;:])\s+')
_TOOL_START_TOKENS = ('<tool_call>', '<tools>')
# ᐈ — checked separately: only when followed by { (tool call JSON)
# Qwen3 also uses ᐈ as a decorative character ("ᐈ In one..."), so it
# can't be detected without a following {
_TOOL_CALL_AE_RE = re.compile(r'ᐈ.{0,10}\{', re.DOTALL)
_MIN_SENT_CHARS   = 12    # minimum characters before a sentence split
_MAX_CHUNK_CHARS  = 80    # maximum characters before a forced comma split (~5-6s TTS)
# TTS request batching: OmniVoice sounds less stable
# on very short isolated phrases — sentences are accumulated up to ~80-120 characters
# OR 2 sentences (whichever comes first), instead of sending literally one short
# sentence at a time. Measured: TTFA ≈1.1-1.2s, total 7-19% longer than a single request.
_TTS_CHUNK_MIN_CHARS     = 80
_TTS_CHUNK_MAX_SENTENCES = 2
# Qwen3 sometimes starts the answer by rephrasing the question as "Почему X?" — filtered out
_ECHO_QUESTION_RE = re.compile(
    r'^(?:Почему|Почём|Почем|О\s+чём|Зачем|По\s+поводу)\b.{0,120}\?\s*',
    re.IGNORECASE | re.UNICODE,
)
# What the LLM answers instead of a reply when a voice utterance that got past
# the addressee gate still clearly wasn't meant for the robot (a phone call,
# talking to someone else in the room) — see _build_addressing_block/_query_llm.
_NOT_ADDRESSED_MARKER = '[ignore]'
# /voice/speaker older than this does not belong to the current utterance
_SPEAKER_FRESH_SEC = 4.0
# Who said a user turn, prefixed to its text in the history once more than one
# person talks to the robot (see LLMNode._speaker_tag)
_SPEAKER_TAG_PREFIX = '[Говорит '
_SPEAKER_TAG_RE = re.compile(r'\[Говорит [^\]]{1,60}\]:\s*')


def _speaker_tag(name: str) -> str:
    return f'{_SPEAKER_TAG_PREFIX}{name}]: '


def _is_not_addressed(text: str) -> bool:
    return text.lstrip().startswith(_NOT_ADDRESSED_MARKER)


# Qwen3 at low temperature sometimes parrots the user's utterance back ("Нет, Лёня,
# я справа." → "Лёня, я справа."); once one parrot lands in the history it keeps
# doing it every turn. Live bug 2026-09-30. A sentence counts as an echo when
# (almost) all of its words come from the user's utterance.
_ECHO_MIN_WORDS   = 3
_ECHO_WORD_SHARE  = 0.9
_REPEAT_REQUEST_RE = re.compile(r'\b(?:повтори|скажи|произнеси)', re.IGNORECASE)


def _echo_words(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower().replace('ё', 'е'))


def _is_user_echo(text: str, user_text: str) -> bool:
    if not user_text or _REPEAT_REQUEST_RE.search(user_text):
        return False   # "повтори за мной ..." — repeating is the answer
    words = _echo_words(text)
    if len(words) < _ECHO_MIN_WORDS:
        return False
    user_words = set(_echo_words(user_text))
    return sum(w in user_words for w in words) / len(words) >= _ECHO_WORD_SHARE


# Qwen3 sometimes switches to Chinese on creative tasks — strip ideographs from TTS chunks
_CJK_RE = re.compile(
    '[⺀-⿿　-〿぀-ゟ゠-ヿ㐀-䶿一-鿿'
    '豈-﫿\U00020000-\U0002A6DF\U0002A700-\U0002CEAF]+',
    re.UNICODE,
)

# OmniVoice non-speech inline tags — the LLM
# inserts them directly into the response text, in square brackets, anywhere
# in the phrase. The list is the full set supported by the server; the system
# prompt (build_system_prompt) gives the LLM this exact same list verbatim. Any
# other "[...]" in the response is a likely model hallucination rather than a
# real server tag — it is stripped so it doesn't reach TTS as-is.
_INLINE_TAGS = frozenset({
    'laughter', 'sigh', 'confirmation-en', 'question-en', 'question-ah',
    'question-oh', 'question-ei', 'question-yi', 'surprise-ah',
    'surprise-oh', 'surprise-wa', 'surprise-yo', 'dissatisfaction-hnn',
})
# Any short "[...]" — not just Latin script: the LLM may "translate" a tag into
# Russian (e.g. "[смеётся]") or invent something not on the list — that must
# also be stripped rather than passed through to TTS as-is.
_INLINE_TAG_RE = re.compile(r'\[([^\[\]]{1,40})\]')


def _filter_inline_tags(text: str) -> str:
    """Keeps only known OmniVoice tags (normalizes case — the server expects
    exact lowercase spelling), strips any other "[...]"."""
    def _sub(m):
        tag = m.group(1).strip().lower()
        return f'[{tag}]' if tag in _INLINE_TAGS else ''
    return _INLINE_TAG_RE.sub(_sub, text)


def _strip_cjk(text: str) -> str:
    """Removes CJK ideographs (Chinese/Japanese/Korean) and normalizes whitespace."""
    cleaned = _CJK_RE.sub('', text)
    return re.sub(r'  +', ' ', cleaned).strip()


def _extract_text_tool_calls(text: str) -> list[dict]:
    """Extracts tool calls from a text response (fallback for Qwen3)."""
    calls = []
    # Format 1: ᐈ{...} or <tool_call>{...}</tool_call>
    for json_str in _TEXT_TOOL_CALL_RE.findall(text):
        try:
            obj = json.loads(json_str)
            if 'name' in obj and 'arguments' in obj:
                calls.append({'function': {'name': obj['name'], 'arguments': obj['arguments']}})
        except (json.JSONDecodeError, KeyError):
            pass
    # Format 2: <tools>\n{...}\n{...}\n</tools>
    for block in _TOOLS_BLOCK_RE.findall(text):
        for line in block.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if 'name' in obj and 'arguments' in obj:
                    calls.append({'function': {'name': obj['name'], 'arguments': obj['arguments']}})
            except json.JSONDecodeError:
                pass
    return calls


def _build_user_content(text: str, image_b64: str | None):
    """Content for a user message: a plain string if there's no image, otherwise an
    OpenAI vision content array (text + image_url data URI). The image is only
    included in the OUTGOING request — self.history keeps a text placeholder
    instead (see _query_llm), otherwise the base64 would bloat the token count
    of every subsequent turn. EXACTLY ONE image — the vLLM server returns 400 for
    >1 image_url in a single prompt ("At most 1 image(s) may be provided", verified 2026-08-26)."""
    if not image_b64:
        return text
    return [
        {'type': 'text', 'text': text or 'Что на этой фотографии?'},
        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_b64}'}},
    ]


def _normalize_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """
    Guarantees the OpenAI-schema `id`/`type` fields on every tool call.

    vLLM strictly validates the message history: an assistant message with
    tool_calls, sent back in the next request (self.history is replayed as-is
    in the R1 of the new turn), must have `id` (str) and `type: "function"`
    on every element — otherwise 400 Bad Request
    (ChatCompletionMessageFunctionToolCallParam.id/type: Field required).
    The text fallback (_extract_text_tool_calls) doesn't provide these fields
    at all, and the SSE parser (_iter_sse_chunks) only keeps id if the server
    sent one — a synthetic one is substituted if it's missing.
    """
    out = []
    for tc in tool_calls:
        out.append({
            'id':       tc.get('id') or f'call_{uuid.uuid4().hex[:24]}',
            'type':     'function',
            'function': tc['function'],
        })
    return out


def _strip_tool_blocks(text: str) -> str:
    """Removes tool call blocks from the text before sending it to TTS."""
    text = _TOOLS_BLOCK_RE.sub('', text)
    text = _TEXT_TOOL_CALL_RE.sub('', text)
    return re.sub(r'  +', ' ', text).strip()


def _clean_llm_text(text: str) -> str:
    """Strip stage directions, CJK, decorative markers and unrecognized inline
    tags that LLM injects."""
    cleaned = _STAGE_DIR_RE.sub('', text)
    cleaned = _SPEAKER_TAG_RE.sub('', cleaned)   # the model copying the history's tags
    cleaned = _CJK_RE.sub('', cleaned)
    cleaned = _filter_inline_tags(cleaned)
    cleaned = re.sub(r'  +', ' ', cleaned).strip()
    cleaned = re.sub(r'^[ᐈ,.\s]+', '', cleaned).strip()  # leading ᐈ bullets
    return cleaned


def _flatten_tool_history(history: list, reminder: str) -> list:
    """Flatten tool-role messages to be compatible with servers that don't support role=tool.

    Transforms [user, assistant(tool_calls), tool(result)] into
    [user, assistant(text), user(results + reminder)] so the server
    doesn't choke on the 'tool' role or assistant messages with pending tool_calls.
    """
    tool_results: list[str] = []
    flattened: list[dict] = []
    for msg in history:
        role = msg.get('role')
        if role == 'tool':
            tool_results.append(msg.get('content', ''))
        elif role == 'assistant' and msg.get('tool_calls'):
            content = msg.get('content') or '…'
            flattened.append({'role': 'assistant', 'content': content})
        else:
            flattened.append(msg)
    combined = reminder
    if tool_results:
        results_block = '\n'.join(f'[Результат инструмента]: {r}' for r in tool_results)
        combined = f'{results_block}\n{reminder}'
    flattened.append({'role': 'user', 'content': combined})
    return flattened


def _strip_episodic_memory(raw: str) -> str:
    """Keeps only working memory («== Текущий момент ==»), cutting off episodes.

    Episodes are the personal history of a specific interlocutor; they must not
    be shown to the LLM until the person has been identified (person_id known).
    """
    marker = '\n\n== Последние события =='
    idx = raw.find(marker)
    return raw[:idx] if idx != -1 else raw


_SCENE_LABELS_RU = {
    'person': 'человек', 'chair': 'стул', 'sofa': 'диван', 'bed': 'кровать',
    'dining table': 'стол', 'tv monitor': 'телевизор', 'laptop': 'ноутбук',
    'mouse': 'мышь', 'remote': 'пульт', 'keyboard': 'клавиатура',
    'cell phone': 'телефон', 'book': 'книга', 'clock': 'часы', 'vase': 'ваза',
    'bottle': 'бутылка', 'wine glass': 'бокал', 'cup': 'чашка', 'fork': 'вилка',
    'knife': 'нож', 'spoon': 'ложка', 'bowl': 'миска', 'backpack': 'рюкзак',
    'handbag': 'сумка', 'umbrella': 'зонт', 'potted plant': 'растение',
    'scissors': 'ножницы', 'teddy bear': 'плюшевый мишка', 'cat': 'кошка',
    'dog': 'собака', 'refrigerator': 'холодильник', 'microwave': 'микроволновка',
    'oven': 'духовка', 'toaster': 'тостер', 'sink': 'раковина',
}


_SCENE_DIRECTION_RU = {'left': 'слева', 'right': 'справа', 'center': 'по центру'}


def _ru_label(label: str) -> str:
    return _SCENE_LABELS_RU.get(label, label)


def _ru_direction(direction: str) -> str:
    return _SCENE_DIRECTION_RU.get(direction, direction)


# ── Parsing voice direction hints (face-search retry) ─────────────
# Deliberately NOT matching bare "право"/"лево" — collides with "направление",
# "исправить", "справедливо" etc. Whole words/phrases with \b boundaries.
_RIGHT_HINT_RE = re.compile(
    r'\b(направо|справа|правее|по правую руку|с правой стороны|правой рукой)\b',
    re.IGNORECASE)
_LEFT_HINT_RE = re.compile(
    r'\b(налево|слева|левее|по левую руку|с левой стороны|левой рукой)\b',
    re.IGNORECASE)
# Live bug 2026-08-28: "Повернись направо" is a COMMAND to the robot (already
# handled separately via the robot_control/look_direction tool call), NOT a
# hint about where the interlocutor is standing — but by the words
# "направо"/"налево" alone it's indistinguishable from «я справа от тебя».
# Without this exclusion FaceSearchAttempt would erroneously turn the torso
# again on the SAME phrase that had already legitimately turned the head via
# the LLM tool call — an extra/conflicting movement.
_ROBOT_TURN_COMMAND_RE = re.compile(
    r'\b(повернись|поверни\w*|обернись|оберн\w*|посмотри|погляди|взгляни|глянь|оглянись)\b',
    re.IGNORECASE)


def _parse_direction_hint(text: str) -> tuple[str, str]:
    """Returns (direction, phrase), direction ∈ {'left','right','none'}.

    If both directions occur in the utterance (e.g. «не слева, а справа»),
    the LAST mention by position wins — that's usually the speaker's
    corrected/final answer.

    Command phrases to the robot ("повернись направо", "посмотри налево") are
    excluded — see _ROBOT_TURN_COMMAND_RE.
    """
    if _ROBOT_TURN_COMMAND_RE.search(text):
        return 'none', ''
    right_m = list(_RIGHT_HINT_RE.finditer(text))
    left_m = list(_LEFT_HINT_RE.finditer(text))
    if not right_m and not left_m:
        return 'none', ''
    last_right = right_m[-1].start() if right_m else -1
    last_left = left_m[-1].start() if left_m else -1
    if last_right > last_left:
        return 'right', right_m[-1].group(0)
    return 'left', left_m[-1].group(0)


def _build_scene_block(scene_ctx: dict | None) -> str:
    """Short description of the scene (objects + people) for the end of the system prompt."""
    if not scene_ctx:
        return ''

    parts = []
    location = scene_ctx.get('location')
    if location:
        parts.append(f'Ты сейчас в: {location}.')

    person_count = scene_ctx.get('person_count', 0)
    if person_count == 1:
        parts.append('Перед тобой 1 человек.')
    elif person_count > 1:
        parts.append(f'Перед тобой {person_count} человек(а).')

    others = [o for o in scene_ctx.get('objects', []) if o.get('label') != 'person']
    if others:
        items_str = ', '.join(
            f"{_ru_label(o['label'])} ({o['distance_m']:.1f}м, {_ru_direction(o['direction'])})"
            for o in others
        )
        parts.append(f'Рядом: {items_str}.')

    if not parts:
        return ''
    # Explicitly marked as a "right now" snapshot: otherwise at low temperature
    # the model tends to repeat its previous answer from the dialogue history,
    # even if the scene in front of the camera has already changed.
    return ('\nСцена ПРЯМО СЕЙЧАС (может отличаться от того, что ты говорил '
            'раньше в этом разговоре — доверяй этому, а не своим прошлым словам): '
            + ' '.join(parts) + '\n')


_NEVER_FOUND_EXAMPLES = [
    'Извини, я тебя не вижу — ты где?',
    'Прости, никак не могу тебя найти взглядом, ты рядом?',
    'Я не понимаю, где ты — подскажешь?',
]
_LOST_AGAIN_EXAMPLES = [
    'Ой, кажется я тебя потерял из виду — ты всё ещё здесь?',
    'Извини, отвлёкся и не вижу тебя — ты не отошёл?',
    'Погоди, я тебя не вижу сейчас — где ты?',
]


def _build_face_search_block(face_search_ctx: dict | None) -> str:
    """Two independent pieces, both from /behavior/face_search_status:

    1. Honesty about "I see you / I don't" (face_search_ctx['locked']) — ALWAYS,
       whenever the face isn't currently tracked, regardless of ask_now. Live
       bug 2026-08-31: on a direct "can you see me?" question the LLM would
       answer "yes, I see you!" with no check at all, even though the
       head_tracker track wasn't held — a pure hallucination. locked is
       updated in behavior_manager_node IMMEDIATELY on track loss (not via a
       grace period), so it's always current here.
    2. A request to naturally ask "where are you" — only when sound-based
       search yields no result 2+ times in a row (ask_now). Different examples
       for "never found since wake word" (kind='never_found') and "lost mid an
       already ongoing dialogue" (kind='lost_again'). Doesn't require an
       answer strictly in "left/right" format — direction is determined
       separately anyway, by parsing the voice hint and/or /sound_direction
       (see behavior_manager_node.FaceSearchAttempt).
    """
    if not face_search_ctx:
        return ''

    parts = []
    if not face_search_ctx.get('locked', True):
        parts.append(
            'ВАЖНО: ты СЕЙЧАС физически не видишь собеседника (нет активного '
            'трека лица) — даже если видел(а) его секунду назад. НИКОГДА не '
            'говори "вижу тебя", не описывай его внешность/одежду/окружение, '
            'если только в ЭТОМ ЖЕ ответе не вызвал(а) look_and_describe или '
            'look_direction и не получил(а) реальный результат. На прямой '
            'вопрос "ты меня видишь?" — честно ответь, что не видишь прямо '
            'сейчас, вместо автоматического "да". Если решил(а) подтвердить '
            'зрением — используй ТОЛЬКО что вернул инструмент В ЭТОМ ответе '
            '(цвет одежды, поза, окружение) — НЕ повторяй старые детали из '
            'более ранних своих реплик в этом разговоре как будто это видно '
            'прямо сейчас; если инструмент дал путаный/противоречивый '
            'результат — так и скажи, не выдумывай уверенное описание.\n'
        )

    if face_search_ctx.get('ask_now'):
        examples = (_NEVER_FOUND_EXAMPLES if face_search_ctx.get('kind') == 'never_found'
                    else _LOST_AGAIN_EXAMPLES)
        examples_str = ' / '.join(f'"{e}"' for e in examples)
        parts.append(
            'Поиск по звуку/направлению не даёт результата уже пару реплик. '
            'Естественно и коротко вплети в ЭТОТ ответ вопрос о том, где он — '
            f'своими словами, в духе: {examples_str}. Не настаивай на формате '
            '"слева/справа" — подойдёт любой ответ ("я здесь", "у окна" и '
            'т.п.), направление всё равно определяется отдельно по голосу. '
            'Не игнорируй суть его исходного сообщения.\n'
        )

    return '\n' + ''.join(parts) if parts else ''


def _build_addressing_block(addressing: str | None) -> str:
    """Why the current voice utterance was let through the addressee gate.

    addressing: 'name' / 'gaze' / 'wake' / '' (reason unknown, e.g. a queued
    phrase), None — not a voice request (Telegram): no block at all.
    Everything except 'name' also gets the [ignore] rule: the gate can't tell a
    phone call held while facing the robot from a question to it — the LLM can.
    """
    if addressing is None:
        return ''
    if addressing == 'name':
        return '\nАдресность: к тебе обратились по имени — реплика точно тебе, отвечай.\n'
    why = {
        'gaze': 'собеседник смотрит тебе в глаза',
        'wake': 'реплика прозвучала сразу после «Эй, Лёня», лицо ещё не найдено',
    }.get(addressing, 'причина неизвестна')
    return (
        f'\nАдресность ({why}): микрофон слышит всё вокруг, и реплика могла быть '
        f'сказана НЕ тебе — кусок телефонного разговора, человек говорит с кем-то '
        f'другим в комнате (спорит, отчитывает, обращается к другому человеку, '
        f'говорит о том, что к тебе не относится), обрывок фразы без вопроса или '
        f'просьбы к тебе. Если реплика явно не обращена к тебе — не отвечай и не '
        f'вызывай инструменты, выведи ровно {_NOT_ADDRESSED_MARKER} и больше ничего. '
        f'Если хочется ответить «я не совсем понял», «я тут ни при чём», «о ком ты '
        f'говоришь?» — это почти всегда значит, что говорили не тебе → '
        f'{_NOT_ADDRESSED_MARKER}. Если же реплика продолжает ваш разговор (ответ на '
        f'твой вопрос, уточнение) или это понятная просьба/вопрос к тебе — отвечай как обычно.\n'
    )


def _build_speaker_block(speaker: dict | None) -> str:
    """Who is speaking by voice vs who is in frame (/voice/speaker from
    identity_manager). The robot may look at one person while another one, out
    of frame, talks to it — live bug 2026-09-30: "смотрю на Николь", Artur said
    "я справа", the LLM took it as Николь's words and started parroting.
    speaker — the fresh /voice/speaker for this utterance, None — no block."""
    if not speaker:
        return ''
    if speaker.get('other_speaker'):
        return _build_other_speaker_block(speaker)
    conf    = speaker.get('confidence')
    name    = speaker.get('name') or ''
    face    = speaker.get('face_name') or ''
    in_view = bool(speaker.get('face_visible')) or bool(face)
    # No quoted example phrases here: with «я справа» quoted, Qwen started
    # parroting the user again (checked against the server 2026-09-30).
    if conf == 'high' and name:
        if face == name:
            return f'\nКто говорит: {name} (узнал по голосу, он же у тебя в кадре).\n'
        if face:
            return (f'\nКто говорит: {name} (узнал по голосу). В кадре у тебя {face} — '
                    f'это не говорящий, {name} сейчас, скорее всего, вне кадра.\n')
        return f'\nКто говорит: {name} (узнал по голосу), в кадре его нет.\n'
    guess = f', немного похож на {name}' if conf == 'uncertain' and name else ''
    if in_view:
        return (f'\nКто говорит: неизвестно (голос не опознан{guess}). В кадре кто-то есть, '
                f'но это не обязательно говорящий — он может быть вне кадра.\n')
    return f'\nКто говорит: неизвестно (голос не опознан{guess}), в кадре никого нет.\n'


def _build_other_speaker_block(speaker: dict) -> str:
    """The phrase came from someone other than the current interlocutor (SV
    rejected the voice) who called the robot by name — multi-person talk."""
    conf = speaker.get('confidence')
    name = speaker.get('name') or ''
    face = speaker.get('face_name') or ''
    if conf == 'high' and name:
        who = f'{name} (узнал по голосу)'
    elif conf == 'uncertain' and name:
        who = f'другой человек, голос немного похож на {name}'
    else:
        who = 'другой человек, голос не опознан'
    main = f' (твой собеседник сейчас — {face})' if face else ''
    return (f'\nКто говорит: {who}. Это НЕ твой текущий собеседник{main}, а ещё один '
            f'человек рядом, он позвал тебя по имени — ответь именно ему. Рядом несколько '
            f'людей: реплики в истории помечены «[Говорит …]:», не путай, кто что сказал. '
            f'Свой ответ такой пометкой не начинай.\n')


def build_system_prompt(oh_schema: str, person_ctx: dict | None = None,
                        memory_context: str = '', scene_ctx: dict | None = None,
                        face_search_ctx: dict | None = None,
                        addressing: str | None = None,
                        speaker: dict | None = None) -> str:
    """Builds the system prompt with the device schema and the interlocutor's context.

    oh_schema — JSON string with the OpenHAB schema (name/label/type/options, no state).
                Sent once at the start of the dialogue; states are queried
                via get_openhab_states / search_openhab_items as needed.
    scene_ctx — scene summary from scene_manager_node (objects + people around).
    face_search_ctx — face-search status from behavior_manager_node
                       (/behavior/face_search_status), see _build_face_search_block.
    addressing — why a voice utterance passed the addressee gate, see
                 _build_addressing_block (None for Telegram).
    """
    if person_ctx:
        name       = person_ctx.get('name') or 'Незнакомец'
        person_id  = person_ctx.get('person_id')
        meet_count = person_ctx.get('meet_count', 1)
        emotion    = person_ctx.get('current_emotion', '')
        notes      = person_ctx.get('notes', {})
        notes_str  = (', '.join(f'{k}: {v}' for k, v in notes.items())
                      if notes else 'нет заметок')
        pid_str    = f', person_id={person_id}' if person_id is not None else ''
        person_block = (
            f'\nСобеседник: {name}{pid_str} (встреч: {meet_count}, '
            f'эмоция: {emotion or "неизвестно"}, заметки: {notes_str}).\n'
            f'Обращайся к нему по имени. Учитывай его эмоциональное состояние.\n'
            f'Сохраняй в память УСТОЙЧИВЫЕ ФАКТЫ: работа, возраст, хобби, питомцы, здоровье, имена членов семьи, привычки, предпочтения.\n'
            f'НЕ сохраняй: разовые просьбы ("передай привет", "включи свет"), задачи, общие фразы, вопросы.\n'
            f'Критерий: "Будет ли это важно знать через месяц?" → Да → save_memory. Нет → не сохранять.\n'
        )
    else:
        person_block = ''

    if oh_schema:
        try:
            schema_items = json.loads(oh_schema)
            # Only include controllable actuator types — sensors (Number, Contact,
            # DateTime, Location) are queried on demand via get_openhab_states.
            _ACTUATOR_TYPES = {'Switch', 'Dimmer', 'Color', 'Rollershutter', 'Player'}
            controllable = [
                it for it in schema_items
                if it.get('type', '').split(':')[0] in _ACTUATOR_TYPES
                or (it.get('type', '').split(':')[0] == 'String'
                    and len(it.get('options', [])) <= 10)
            ]
            lines = ['name (точное имя для API)              | тип                 | label']
            lines.append('-' * 75)
            for it in controllable:
                opts = f"  options={it['options']}" if 'options' in it else ''
                lines.append(
                    f"{it['name']:<42s}| {it['type']:<20s}| {it.get('label', '')}{opts}"
                )
            schema_text = '\n'.join(lines)
        except Exception:
            schema_text = oh_schema
        devices_block = (
            f'Устройства OpenHAB (ВАЖНО: используй ТОЧНОЕ имя из колонки "name"):\n'
            f'{schema_text}\n\n'
            f'Семантические группы (используй в search_openhab_items → group_filter):\n'
            f'- AllLights      — все светильники (Dimmer, Switch, Color)\n'
            f'- All_Heaters    — все обогреватели (радиаторы + тёплые полы)\n'
            f'- Floor_Heaters  — только тёплые полы\n'
            f'- TempSensors    — датчики температуры (только чтение)\n'
            f'- HumiditySensors — датчики влажности (только чтение)\n'
            f'- TargetTemp     — целевые температуры термостатов (можно менять)\n'
            f'- PhonesAtHome   — телефоны дома (присутствие)\n\n'
            f'Правила управления:\n'
            f'- Switch: ON / OFF\n'
            f'- Dimmer: 0-100 (процент яркости)\n'
            f'- Number:Temperature (термостат/сетпоинт): числовое значение в °C\n'
            f'- String (hvac_mode): одно из доступных options\n'
            f'- Color: H,S,B (0-360, 0-100, 0-100)\n'
            f'- Если нужно узнать текущее состояние — используй get_openhab_states или search_openhab_items\n'
        )
    else:
        devices_block = 'OpenHAB устройства: нет данных (openhab_bridge_node не запущен).\n'

    memory_block      = f'\n{memory_context}\n' if memory_context else ''
    scene_block       = _build_scene_block(scene_ctx)
    face_search_block = _build_face_search_block(face_search_ctx)
    addressing_block  = _build_addressing_block(addressing)
    speaker_block     = _build_speaker_block(speaker)

    # IMPORTANT: the whole DYNAMIC block (person_block/memory_block/scene_block —
    # time, interlocutor, recent events, surrounding scene — changes on EVERY
    # request) is moved to the very end of the prompt. Everything above is static
    # between requests within one session (and often between sessions) — this is
    # needed for the server's prefix cache: if the dynamic part sits in the middle,
    # any cache of the common prefix (system prompt) breaks at that point on every
    # turn of the dialogue, and all the static blocks after it (the OpenHAB table
    # etc.) have to be recomputed from scratch.
    # Static first, dynamic last (keeps the vLLM prefix cache warm).
    return f"""Ты робот по имени Лёня. Ты член семьи. Твоя главная задача - общение. Стараться узнать о собеседнике или семье что-то новое и сохранять в базу данных с помощью инструментов. Так же твоя задача отвечать на любые вопросы, и выполнять команды. Ты можешь управлять умным домом через OpenHAB, двигаться и выражать эмоции.
Используй инструменты (tools) для выполнения команд.
ВАЖНО: Никогда не используй азиатские языки в ответах, никаких иероглифов!
ВАЖНО: Никогда не повторяй и не перефразируй вопрос пользователя в начале ответа. Не начинай ответ со слов "Почему", "Почём", "Зачем", "О чём", "По поводу" или любого пересказа вопроса. Отвечай сразу по существу.
ВАЖНО: Текстовый ответ озвучивается напрямую TTS. НЕ добавляй ремарки или сценические указания в скобках — например, (Тихо), (С улыбкой), (Шёпотом). Для изменения стиля голоса используй инструмент set_voice_style. Не используй URL в ответе - TTS их плохо произносит.
ВАЖНО: TTS понимает неречевые теги ПРЯМО ВНУТРИ текста ответа, в квадратных скобках, в любом месте фразы (это НЕ ремарка из строки выше — те в круглых скобках запрещены полностью, а эти теги — фиксированный список звуков TTS). Разрешены ТОЛЬКО эти, дословно: [laughter] [sigh] [confirmation-en] [question-en] [question-ah] [question-oh] [question-ei] [question-yi] [surprise-ah] [surprise-oh] [surprise-wa] [surprise-yo] [dissatisfaction-hnn]. Вставляй умеренно и только когда это естественно усиливает фразу — например «[laughter] Ну ты даёшь!», «[sigh] Ладно, попробую ещё раз», «[surprise-ah] Ого, не ожидал!». НЕ придумывай свои теги и не переводи их на русский — неизвестные теги будут вырезаны из речи. Это дополняет set_voice_style, а не заменяет его.
ВАЖНО: Ответы — РАЗГОВОРНЫЕ и КРАТКИЕ, 1-2 предложения максимум. Отвечай ТОЛЬКО на то, что спросили — не пересказывай все данные из инструмента. Примеры: вопрос «будет ли дождь?» → «Да, завтра ожидается небольшой дождь, около трёх миллиметров» (не надо перечислять почасовой прогноз и скорость ветра). Вопрос «какая температура?» → «Завтра от пяти до тринадцати градусов, пасмурно». Если хотят подробности — спросят.
ВАЖНО: При вызове save_memory, items_control, robot_control — всегда включай параметр speak_text с ответом пользователю (1-2 предложения, естественное продолжение разговора). При save_memory НЕ говори "Запомнил/Сохранил" — просто продолжай диалог как будто ты это уже знаешь. При get_openhab_states, search_openhab_items — speak_text не нужен, ответ формируй после получения данных.

- Если пользователь управляет роботом — используй robot_control
- Если пользователь прощается ("пока", "до свидания", "увидимся", "прощай") — вызови robot_control(action="goodbye", text="[твоя прощальная фраза]"). Не отвечай просто текстом на прощание — нужен tool call чтобы завершить сессию.
- Для ЛЮБОГО вопроса о погоде — используй get_weather (не web_search). По умолчанию: Bødalen, Asker, сегодня.
- Если нужна информация из интернета (кроме погоды) — используй web_search
- Если нужно вспомнить факты о человеке из прошлых разговоров — используй search_memory
- Если пользователь просит НАПОМНИТЬ что-либо ("напомни", "не забудь напомнить") — ВСЕГДА используй set_reminder. НИКОГДА не используй save_memory для напоминаний!
- Используй set_voice_style когда контекст разговора вызывает эмоциональную/тональную реакцию
  (хорошая новость → happy, плохая новость → sad, неожиданность → surprise). Он одновременно
  меняет и голос, и мимику лица на всё время произнесения фразы (доступны только 4 пресета:
  neutral/happy/sad/surprise — не описывай тон текстом)
- Если просто разговор — отвечай текстом без tool call
- Если пользователь просит "передай на колонку", "скажи в гостиной", "объяви" — используй broadcast_message. НЕ используй items_control для LivingRoom_Chromecast.
- Если спрашивают "что ты видишь", "что это", "что у меня в руках", "что за окном", "кто перед тобой", "опиши что рядом" БЕЗ указания стороны/поворота — ОБЯЗАТЕЛЬНО вызывай look_and_describe (query = суть вопроса). Блок "Сцена ПРЯМО СЕЙЧАС" ниже — это только фоновый YOLO-контекст (кто рядом, сколько человек) для твоей ОБЩЕЙ ориентации, а не источник ответа на прямой вопрос "что ты видишь" — YOLO распознаёт ограниченный набор предметов и часто ошибается. НЕ вызывай robot_control(action="status") для таких вопросов — status только для батареи/позиции сервоприводов.
- Если просят посмотреть В КАКУЮ-ТО СТОРОНУ и сказать что там ("посмотри направо, что видишь?", "глянь налево", "обернись, что там?", "посмотри что у меня за спиной") — используй ТОЛЬКО look_direction (сам поворачивает голову/корпус, ждёт завершения поворота и только потом смотрит). НЕ вызывай для этого robot_control(action=head) — оно только повернёт голову, но не посмотрит и не опишет, а если вызвать его вместе с look_and_describe в одном ответе, снимок улетит раньше, чем голова довернётся.
- Если пользователь говорит, что стоит/сидит сбоку от тебя (например «я справа от тебя») и просит повернуться к нему или посмотреть на него, а одной головы физически не хватает — используй scope="partial" (в robot_control или look_direction). "Обернись"/"повернись полностью" — scope="full".

ВАЖНО — управление устройствами:
- НИКОГДА не придумывай имена устройств — используй только точные имена из таблицы ниже.
- Таблица содержит ВСЕ управляемые устройства. Если устройство нужной комнаты есть в таблице — вызывай items_control НАПРЯМУЮ, без поиска.
- Пример: "выключи свет в гостевой" → вижу GuestRoom_Dimmer и GuestRoom_Color в таблице → сразу items_control для каждого.
- search_openhab_items используй ТОЛЬКО если нужного устройства нет в таблице или нужно узнать текущее состояние группы.
- speak_text в items_control/robot_control пиши ТОЛЬКО уверенный ответ. Если не уверен в результате — не пиши speak_text, сформируй ответ после выполнения.
- КРИТИЧНО: НИКОГДА не пиши текстом, что включил/выключил/изменил устройство, если не вызвал items_control. Ответ "выключил"/"готово"/"сделал" БЕЗ соответствующего tool call — это ложь пользователю о реальном состоянии его дома. Если сомневаешься, что именно нужно выключить/включить — это НЕ повод промолчать и ответить текстом, а повод выполнить действие максимально широко (см. ниже) или переспросить.
- Если цель команды широкая или неточная ("выключи везде", "выключи весь свет", "выключи всё", "во всём доме") — это означает ВСЕ устройства группы AllLights (для света) или All_Heaters (для отопления). СНАЧАЛА вызови search_openhab_items(group_filter=AllLights, state_filter="ON") чтобы найти включённые устройства, ЗАТЕМ вызови items_control для КАЖДОГО найденного. Не отвечай текстом вместо этой последовательности вызовов.

{devices_block}
{person_block}{memory_block}{scene_block}{face_search_block}{addressing_block}{speaker_block}"""


_RU_WEEKDAY = {
    'понедельник': 0, 'вторник': 1,
    'среда': 2, 'среду': 2,
    'четверг': 3,
    'пятница': 4, 'пятницу': 4,
    'суббота': 5, 'субботу': 5,
    'воскресенье': 6, 'воскресение': 6,
    'monday': 0, 'tuesday': 1, 'wednesday': 2,
    'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6,
}


def _resolve_reminder_date(date_str: str | None) -> str | None:
    """Converts a human-readable date name into ISO YYYY-MM-DD.

    Supports: weekday names (Russian/English), 'завтра'/'послезавтра',
    'tomorrow'/'day after tomorrow', a ready ISO date. None/empty → None.
    """
    if not date_str:
        return None
    s = date_str.strip().lower()
    if not s:
        return None

    today = datetime.date.today()

    if s in ('сегодня', 'today'):
        return today.isoformat()
    if s in ('завтра', 'tomorrow'):
        return (today + datetime.timedelta(days=1)).isoformat()
    if s in ('послезавтра', 'day after tomorrow'):
        return (today + datetime.timedelta(days=2)).isoformat()

    if s in _RU_WEEKDAY:
        target = _RU_WEEKDAY[s]
        days_ahead = (target - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7  # today is that day → next week
        return (today + datetime.timedelta(days=days_ahead)).isoformat()

    # ISO YYYY-MM-DD
    try:
        datetime.date.fromisoformat(date_str.strip())
        return date_str.strip()
    except ValueError:
        return date_str  # return as-is, memory_node will figure it out


def _resolve_reminder_time(time_str: str | None) -> str | None:
    """Normalizes a time string into HH:MM format. None/empty → None."""
    if not time_str:
        return None
    s = time_str.strip()
    if not s:
        return None
    # Already in HH:MM or H:MM format
    m = re.fullmatch(r'(\d{1,2}):(\d{2})', s)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mn <= 59:
            return f'{h:02d}:{mn:02d}'
    return None


def _base_url(chat_url: str) -> str:
    """Extracts the base URL from an endpoint: http://host:port/api/chat → http://host:port"""
    from urllib.parse import urlparse
    p = urlparse(chat_url)
    return f'{p.scheme}://{p.netloc}'


def _models_url(chat_url: str) -> str:
    """.../v1/chat/completions → .../v1/models — health-check endpoint OpenAI API."""
    if chat_url.endswith('/chat/completions'):
        return chat_url[: -len('/chat/completions')] + '/models'
    return _base_url(chat_url) + '/v1/models'


# Camera frames: newest only, no retransmits (face_capture publishes RELIABLE —
# a BEST_EFFORT subscriber is compatible with it)
_CAMERA_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


class LLMNode(LifecycleNode):
    def __init__(self):
        super().__init__('llm_node')

        # Introduction gate — initialized before subscriptions in on_configure
        self._introducing = False

        # ── State ────────────────────────────────────────────────────────
        self.history         = []
        self._processing     = False
        self._lock           = threading.Lock()
        # Voice command that arrived while _processing — run after the current reply
        self._pending        = PendingUtterance(max_age_sec=30.0)
        # Warmup is skipped once a real request ran; cancelled on deactivate
        self._warmup_cancel  = threading.Event()
        self._queried_since_activate = False
        self._tg_denied_tools: frozenset = frozenset()
        self._audit: ToolAuditLog | None = None
        self._tg_req_id:     str = ''   # request_id of the current Telegram request
        self._person_context = None
        # OpenHAB cache from openhab_bridge_node
        self._oh_schema      = ''        # schema JSON string (name/label/type/options)
        self._oh_items       = []        # list of dicts with the current state
        # Latest frames from the eye cameras (JPEG bytes) — for look_and_describe/look_direction.
        # Both eyes are cached separately: one photo may be blurry/out of focus,
        # so both cameras go into the vision request at once (see _call_vision_model).
        self._latest_eye_jpeg:       bytes | None = None   # left
        self._latest_eye_jpeg_right: bytes | None = None
        # Voice style (+ synchronized facial expression for the duration of speech,
        # see tts_node) — buffered by set_voice_style, included in /llm_response
        self._voice_style    = {'emotion': ''}
        # Interlocutor's gaze: True/False/None (None = no face / no data from the detector)
        # Used to filter out speech not addressed to the robot, see _addressing_reason.
        self._looking_at_robot: bool | None = None
        # identity_manager: an unknown face is waiting silently — an addressed
        # utterance starts the introduction instead of going to the LLM.
        self._introduce_on_address = False
        self._social_ctx_ts: float = 0.0   # monotonic time of the last /social_context
        self._wake_ts:       float = 0.0   # monotonic time of the last wake word

        # web_search sync: a background thread waits for the result from BM
        self._search_event          = threading.Event()
        self._search_result_data: str | None = None
        self._waiting_for_search    = False

        # Memory context from memory_node — inserted into the system prompt
        self._memory_context: str = ''
        # Scene summary (objects + people) from scene_manager_node — inserted into the system prompt
        self._scene_context: dict = {}
        # Face-search status from behavior_manager_node (/behavior/face_search_status)
        # — inserted into the system prompt via _build_face_search_block
        self._face_search_status: dict = {}
        self._speaker: dict = {}   # /voice/speaker — who spoke the last utterance (voice-id)
        # /speaker_evidence (lips + gaze per phrase) for the gaze veto, see _gaze_lips_check
        self._lips_cond     = threading.Condition()
        self._lips_evidence = collections.deque(maxlen=8)
        self._lips_used_seg = None
        # Accumulates the transcript of the current dialogue
        self._dialogue_lines: list[str] = []

    @property
    def model(self) -> str:
        """The model is chosen depending on the active server."""
        return self.model_primary if self._active_url == self.llm_url else self.model_fallback

    # ── Server check ─────────────────────────────────────────────────────

    def _check_servers(self):
        """Checks both servers and sets the active one."""
        primary_ok  = self._probe_llm(self.llm_url,  self.model_primary, self.bearer_token)
        fallback_ok = bool(self.llm_fallback_url) and self._probe_llm(
            self.llm_fallback_url, self.model_fallback, self.bearer_token_fallback)

        if primary_ok:
            self._active_url = self.llm_url
            self.get_logger().info(f'LLM: primary server available ({self.llm_url})')
        elif fallback_ok:
            self._active_url = self.llm_fallback_url
            self.get_logger().warn(
                f'Primary LLM server unavailable! Using fallback: {self.llm_fallback_url}')
        else:
            self.get_logger().error('LLM server(s) unavailable!' + ('' if self.llm_fallback_url else ' (no fallback configured)'))

    def _probe_llm(self, chat_url: str, model: str, bearer: str) -> bool:
        """Checks LLM server availability via GET /v1/models. Returns True if OK."""
        try:
            headers = {'Authorization': f'Bearer {bearer}'} if bearer else {}
            r = requests.get(_models_url(chat_url), headers=headers,
                             timeout=self.connect_timeout)
            r.raise_for_status()
            models = [m['id'] for m in r.json().get('data', [])]
            model_base = model.split(':')[0]
            if any(model_base in m for m in models):
                self.get_logger().info(f'  {chat_url}: model {model} found')
            else:
                self.get_logger().warn(
                    f'  {chat_url}: model {model} not found among {models}')
            return True
        except Exception:
            return False

    # ── LLM → TTS streaming ──────────────────────────────────────────────

    @staticmethod
    def _iter_sse_chunks(resp: requests.Response):
        """
        Reads the OpenAI/vLLM SSE stream (`data: {...}` line by line, ends with `data: [DONE]`).
        tool_calls arrive in fragments across chunks (id/name separate from arguments,
        arguments — a few characters per chunk, all tied to the same
        index) — we accumulate them and hand them out as a single unit only once the
        stream is done (otherwise the calling code would get a half-assembled JSON in arguments).
        The end of the stream is a separate chunk with empty choices and a filled-in usage
        (because of stream_options.include_usage), AFTER the chunk with finish_reason —
        that's the one we use to consider the stream done for token log statistics.
        Yields (delta_content, done, tool_calls, raw_line, log_chunk, ttft_signal).
        ttft_signal=True on the first real token of activity (text OR the first
        tool_call fragment) — since the tool_calls themselves accumulate and are only
        handed out in the done-chunk, `tc` alone isn't enough for an accurate TTFT (see below).
        """
        tc_acc: dict[int, dict] = {}
        for line in resp.iter_lines():
            if not line or not line.startswith(b'data:'):
                continue
            raw = line[len(b'data:'):].strip()
            if raw == b'[DONE]':
                return
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            choices = chunk.get('choices') or []
            if not choices:
                # The final usage chunk (empty choices) — end of the stream.
                if chunk.get('usage'):
                    yield '', True, [dict(v) for v in tc_acc.values()], line, chunk, False
                continue
            delta = choices[0].get('delta', {}) or {}
            first_tc_fragment = False
            for tcd in (delta.get('tool_calls') or []):
                idx = tcd.get('index', 0)
                if idx not in tc_acc:
                    first_tc_fragment = True
                    # id usually only arrives in the first fragment of this index —
                    # save it right away; _normalize_tool_calls will substitute a synthetic one
                    # if the server didn't send one (needed for history — see its docstring).
                    tc_acc[idx] = {'id': tcd.get('id') or '',
                                   'function': {'name': '', 'arguments': ''}}
                slot = tc_acc[idx]
                if tcd.get('id') and not slot['id']:
                    slot['id'] = tcd['id']
                fn = tcd.get('function') or {}
                if fn.get('name'):
                    slot['function']['name'] += fn['name']
                if fn.get('arguments'):
                    slot['function']['arguments'] += fn['arguments']
            content = delta.get('content') or ''
            if content:
                yield content, False, [], line, chunk, True
            elif first_tc_fragment:
                # tool_calls are assembled in full and only handed out in the done chunk (see
                # the docstring), but the first fragment is the real moment of the first token:
                # send an empty "ping" (tc=[], ttft_signal=True) — otherwise the TTFT in
                # _stream_llm would be logged as "time to the end of generation".
                yield '', False, [], line, chunk, True
        # The connection closed without a final usage chunk (the server didn't send
        # include_usage, or it dropped) — hand out whatever we managed to collect.
        if tc_acc:
            yield '', True, [dict(v) for v in tc_acc.values()], b'', {}, False

    def _stream_llm(self, payload: dict, read_timeout: float):
        """
        Streams the LLM's response via OpenAI chat.completions (stream=True).
        Yields (delta, done, api_tool_calls). On a connection error, tries the
        fallback server.

        `payload` — the already-built body of the OpenAI request (model/messages/tools/
        temperature/max_tokens/chat_template_kwargs), assembled by the calling
        code; here we only force stream=True and add
        stream_options for usage statistics in the done chunk.
        """
        payload = dict(payload)
        payload['stream'] = True
        payload['stream_options'] = {'include_usage': True}
        # vLLM rejects `tools: []` with HTTP 400 ("must not be an empty array") —
        # R3 sends exactly that, and the reply was lost. Live bug 2026-09-30.
        if not payload.get('tools'):
            payload.pop('tools', None)

        urls = [self._active_url]
        other = self.llm_fallback_url if self._active_url == self.llm_url \
            else self.llm_url
        if other and other != self._active_url:
            urls.append(other)

        _DBG_LAST  = '/tmp/llm_last_payload.json'
        _DBG_ERROR = '/tmp/llm_error_payload.json'
        last_exc = None
        for url in urls:
            try:
                if url != self._active_url:
                    self._active_url = url
                    payload = dict(payload)
                    payload['model'] = self.model
                    self.get_logger().warn(f'Streaming: switched to the fallback server: {url}')

                bearer  = (self.bearer_token if url == self.llm_url
                           else self.bearer_token_fallback)
                headers = {'Authorization': f'Bearer {bearer}'} if bearer else {}
                try:
                    with open(_DBG_LAST, 'w', encoding='utf-8') as _f:
                        json.dump(payload, _f, ensure_ascii=False, indent=2)
                except OSError:
                    pass
                r = requests.post(
                    url, json=payload, headers=headers,
                    stream=True,
                    timeout=(self.connect_timeout, read_timeout),
                )
                r.raise_for_status()
                _first_line    = None
                _t_start       = time.time()
                _t_first_token = None
                try:
                    for delta, done, tc, raw_line, chunk, ttft_signal in self._iter_sse_chunks(r):
                        if _first_line is None and raw_line:
                            _first_line = raw_line[:300]
                        if _t_first_token is None and ttft_signal:
                            _t_first_token = time.time()
                            self.get_logger().info(f'TTFT: {_t_first_token - _t_start:.2f}s')
                        if done:
                            usage = chunk.get('usage') or {}
                            p_tok = usage.get('prompt_tokens', 0)
                            c_tok = usage.get('completion_tokens', 0)
                            think_tok = (usage.get('completion_tokens_details') or {}).get(
                                'reasoning_tokens', 0)
                            try:
                                with open('/tmp/llm_done_chunk.json', 'w') as _f:
                                    json.dump(chunk, _f, ensure_ascii=False, indent=2)
                            except OSError:
                                pass
                            decode_s = ((c_tok - 1) / (time.time() - _t_first_token)
                                        if c_tok > 1 and _t_first_token else 0)
                            c_str = (f'{c_tok} ({decode_s:.1f} tok/s)' if decode_s
                                     else f'{c_tok} tok' if c_tok
                                     else '? (not reporting)')
                            self.get_logger().info(
                                f'Tokens: prompt={p_tok}, completion={c_str}, think={think_tok}'
                            )
                        yield (delta, done, tc)
                finally:
                    if _first_line and b'"error"' in _first_line:
                        self.get_logger().warn(
                            f'stream error from server: {_first_line[:200]}')
                        try:
                            shutil.copy2(_DBG_LAST, _DBG_ERROR)
                            self.get_logger().warn(
                                f'Error payload saved: {_DBG_ERROR}')
                        except OSError:
                            pass
                return
            except requests.exceptions.ConnectionError as e:
                self.get_logger().warn(f'LLM stream {url} unavailable: {e}')
                last_exc = e
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if status in (404, 503):
                    self.get_logger().warn(f'LLM stream {url}: HTTP {status}')
                    last_exc = e
                else:
                    try:
                        shutil.copy2(_DBG_LAST, _DBG_ERROR)
                        self.get_logger().warn(
                            f'LLM stream HTTP {status}: payload saved to {_DBG_ERROR}')
                    except OSError:
                        pass
                    raise
        raise requests.exceptions.ConnectionError(
            'LLM server(s) unavailable') from last_exc

    def _send_tts_chunk(self, text: str, voice_style: str = '') -> None:
        """Sends a block of text (one or two sentences, see _TTS_CHUNK_*)
        to tts_node as a separate Speak goal, or to Telegram (in TG mode)."""
        text = text.strip()
        if not text:
            return
        with self._lock:
            tg_req_id = self._tg_req_id
        if tg_req_id:
            # TG mode: instead of TTS, stream the text back into Telegram
            self._tg_stream_partial(text, tg_req_id)
            return
        if not self._tts_direct_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn('TTS server unavailable for a streaming chunk')
            return
        goal = Speak.Goal()
        goal.text  = text
        goal.voice = voice_style
        self._tts_direct_client.send_goal_async(goal)
        self.get_logger().debug(f'TTS chunk: "{text[:50]}"')

    def _tg_stream_partial(self, text: str, tg_req_id: str) -> None:
        """Publishes a partial text to /telegram_response (partial=True)."""
        msg = String()
        msg.data = json.dumps(
            {'request_id': tg_req_id, 'text': text, 'partial': True},
            ensure_ascii=False,
        )
        self._tg_resp_pub.publish(msg)
        self.get_logger().debug(f'TG partial: "{text[:50]}"')

    def _stream_with_tts(self, payload: dict, echo_of: str = '') -> tuple[str, list]:
        """
        Streams the LLM's response and speaks it as it becomes ready via the 'speak'
        action (POST /tts/stream on the TTS server — OmniVoice
        does not provide a WS bistream, hence one Speak goal per text block,
        rather than one WS session for the whole response).
        Returns (full_content, api_tool_calls).
        Stops sending to TTS once a tool-call marker is detected.
        echo_of — the user's utterance: leading sentences that merely parrot it
        (_is_user_echo) are not spoken; the caller decides what to do with the reply.
        """
        with self._lock:
            voice_style = self._voice_style.get('emotion', '')
            tg_req_id   = self._tg_req_id

        buf: str                  = ''
        content_parts: list[str]  = []
        api_tool_calls: list      = []
        tool_call_detected        = False
        not_addressed             = False
        first_chunk_sent          = False
        chunk_buf: list[str]      = []  # sentences waiting to be sent as one TTS request
        chunk_chars                = 0
        echo_prefix                = bool(echo_of)  # every sentence so far parroted the user

        def _is_echo(sentence: str) -> bool:
            nonlocal echo_prefix
            if echo_prefix and _is_user_echo(sentence, echo_of):
                self.get_logger().warn(f'Echo filter: dropped "{sentence[:60]}"')
                return True
            echo_prefix = False
            return False

        def _flush_chunk(force: bool = False):
            """Sends the accumulated block as a single Speak goal.
            By default waits for _TTS_CHUNK_MIN_CHARS or _TTS_CHUNK_MAX_SENTENCES —
            OmniVoice sounds less stable on very short isolated
            phrases, so we don't send one
            sentence at a time. force=True — end of the response, send the rest as-is."""
            nonlocal chunk_buf, chunk_chars
            if not chunk_buf:
                return
            if not force and chunk_chars < _TTS_CHUNK_MIN_CHARS \
                    and len(chunk_buf) < _TTS_CHUNK_MAX_SENTENCES:
                return
            clean = _clean_llm_text(' '.join(chunk_buf))
            chunk_buf   = []
            chunk_chars = 0
            if clean:
                self._send_tts_chunk(clean, voice_style)

        for delta, done, tc in self._stream_llm(payload, self.timeout_sec):
            buf += delta
            content_parts.append(delta)

            if tc:
                api_tool_calls.extend(tc)

            if done:
                break

            if not tool_call_detected:
                detected = any(marker in buf for marker in _TOOL_START_TOKENS)
                if not detected:
                    detected = bool(_TOOL_CALL_AE_RE.search(buf))
                if detected:
                    tool_call_detected = True
                    # Already-collected but not-yet-sent sentences — discard them,
                    # they must not be spoken (they belong to the part of the response with the tool call).
                    chunk_buf   = []
                    chunk_chars = 0
                    self.get_logger().debug('Streaming: tool call — TTS stopped')

            # The [ignore] answer (see _build_addressing_block) must never reach TTS:
            # while the start of the response could still turn into the marker — wait.
            head = buf.lstrip()
            if not first_chunk_sent and not not_addressed:
                if _is_not_addressed(head):
                    not_addressed = True
                elif _NOT_ADDRESSED_MARKER.startswith(head):
                    continue

            if not tool_call_detected and not not_addressed:
                while True:
                    m = _SENT_SPLIT_RE.search(buf, _MIN_SENT_CHARS)
                    if not m and len(buf) >= _MAX_CHUNK_CHARS:
                        m = _COMMA_SPLIT_RE.search(buf, _MIN_SENT_CHARS)
                    if m:
                        sentence = buf[:m.end()].strip()
                        buf      = buf[m.end():]
                        if sentence:
                            if not first_chunk_sent and _ECHO_QUESTION_RE.match(sentence):
                                self.get_logger().warn(
                                    f'Echo-question filter: dropped "{sentence[:60]}"')
                            elif _is_echo(sentence):
                                pass
                            elif tg_req_id:
                                self._tg_stream_partial(sentence, tg_req_id)
                            else:
                                chunk_buf.append(sentence)
                                chunk_chars += len(sentence)
                                _flush_chunk()
                            first_chunk_sent = True
                    else:
                        break

        # Tail: send the remainder if there are no tool calls.
        # NOTE: with stream:False the loop above breaks immediately (done=True), so
        #   tool_call_detected is always False and api_tool_calls is the only signal of
        #   API tool calls. Text tool calls (ᐈ/xml format) end up here too.
        if not first_chunk_sent and _is_not_addressed(buf):
            not_addressed = True
        if buf.strip() and (not tool_call_detected or tg_req_id) and not api_tool_calls \
                and not not_addressed:
            tail = buf.strip()
            has_text_tool_call = bool(
                _TEXT_TOOL_CALL_RE.search(tail) or _TOOLS_BLOCK_RE.search(tail)
            )
            if has_text_tool_call:
                self.get_logger().debug('Tail contains a text tool call — not sending to TTS/TG')
            else:
                if not first_chunk_sent and _ECHO_QUESTION_RE.match(tail):
                    m_end = _ECHO_QUESTION_RE.match(tail).end()
                    tail = tail[m_end:].strip()
                    self.get_logger().warn(
                        f'Echo-question filter: dropped "{buf.strip()[:m_end][:60]}"')
                if tail and not _is_echo(tail):
                    if tg_req_id:
                        self._tg_stream_partial(tail, tg_req_id)
                    else:
                        chunk_buf.append(tail)
                        chunk_chars += len(tail)

        # Final flush: whatever is left in the buffer is sent as-is,
        # even if it's shorter than the usual threshold (this is the end of the response, nothing to wait for).
        _flush_chunk(force=True)

        return ''.join(content_parts), api_tool_calls

    # ── Receiving a voice command ───────────────────────────────────────

    def _memory_context_cb(self, msg: String):
        """Receives working memory + episodes from memory_node for insertion into the system prompt.

        Always saved (working memory — time/place/mode — is needed in the context
        regardless of whether the interlocutor is known). Personal episodes
        are cut off in _query_llm if person_id is not yet determined.
        """
        with self._lock:
            self._memory_context = msg.data

    def _introducing_cb(self, msg: Bool):
        """Gate: when True — identity_manager is collecting the name, we don't process commands."""
        with self._lock:
            self._introducing = msg.data
        if msg.data:
            self.get_logger().info('LLM: introduction mode — voice_command blocked')
        else:
            self.get_logger().info('LLM: introduction mode finished — ready')

    def _go_idle_cb(self, msg: Bool):
        """Explicit farewell: a full reset of the LLM context.

        _person_context_callback will also publish conversation_end when it receives
        an empty context from identity_manager, but by then the history is already
        empty — there won't be a double publish. We don't touch _memory_context — working
        memory (time/place/mode) must stay in context even without an interlocutor;
        personal episodes are cut off in _query_llm by person_id.
        """
        if not msg.data:
            return
        self._pending.clear()   # the person said goodbye — a queued phrase is moot
        with self._lock:
            history_snapshot      = self.history[:]
            person_ctx            = self._person_context
            self.history          = []
            self._dialogue_lines  = []
            self._voice_style     = {'emotion': ''}
            self._person_context  = None
        if history_snapshot:
            self._publish_conversation_end(history_snapshot, person_ctx)
        self.get_logger().info('go_idle: LLM context cleared (history, person_context, memory)')

    def _robot_sleep_cb(self, msg: Bool):
        """Resets the dialogue history on entering/leaving sleep mode."""
        self._pending.clear()
        with self._lock:
            history_snapshot = self.history[:]
            person_ctx       = self._person_context
            if self.history:
                self.get_logger().info(
                    f'robot_sleep={msg.data} — history reset ({len(self.history)} msgs)')
            self.history          = []
            self._dialogue_lines  = []
            self._voice_style     = {'emotion': ''}
            # Safety net: if /introducing got stuck True (identity_manager went
            # to sleep from State.INTRODUCING before its fix, or was restarted and
            # didn't resend False) — sleep must lift the gate itself, otherwise telegram_ask
            # and voice_command would forever reply "busy", even while asleep.
            if msg.data and self._introducing:
                self.get_logger().warn(
                    'robot_sleep=True while /introducing=True — gate forcibly lifted')
                self._introducing = False
        # On falling asleep — publish the end of the dialogue (if there was one)
        if msg.data and history_snapshot:
            self._publish_conversation_end(history_snapshot, person_ctx)

    # Robot-name forms (lowercase), anywhere in the phrase ("Ясно, Лёня, спасибо").
    # Besides the real forms — what Parakeet actually writes for "Лёня" from a
    # far-field mic (live logs + a noisy TTS test 2026-09-26).
    _ROBOT_NAMES = frozenset({
        'лёня', 'леня', 'лёне', 'лёню', 'леню', 'лёной', 'леной',
        'лёнечка', 'ленечка', 'лёнь', 'лёней', 'леней',
        'лен', 'лён', 'ленин', 'леоня', 'леона', 'леля', 'лёля', 'лёль', 'лення',
        'люня', 'ляня', 'легин', 'кленечка', 'лёночка', 'юленя',
        'лена', 'лене', 'лену',   # no Лена in the household
    })
    # Real words/names STT also writes for "Лёня" — only count in the vocative
    # position (first word, or set off by a comma: "Да, Лень, это я"), otherwise
    # "мне лень" or a phone chat about a Юля would pass.
    _ROBOT_NAMES_FIRST_WORD = frozenset({'лень', 'юля', 'юлі'})
    # Parakeet has no token for some "ё" contexts and emits <unk>: "Л<unk>ня".
    # A short word starting with "л" around an <unk> is taken as the name.
    _UNK_NAME_RE = re.compile(r'^л[а-яё]?\*[а-яё]{0,3}$')
    _GAZE_CTX_STALE_SEC = 2.0   # /social_context comes @ 2 Hz

    def _has_robot_name(self, text: str) -> bool:
        low   = text.lower().replace('<unk>', '*')
        words = re.findall(r'[а-яёі*]+', low)
        vocative = set(words[:1]) | set(re.findall(r'[,.!?]\s*([а-яёі]+)\s*[,.!?]', low))
        return bool(self._ROBOT_NAMES.intersection(words)
                    or self._ROBOT_NAMES_FIRST_WORD.intersection(vocative)
                    or any(self._UNK_NAME_RE.match(w) for w in words))

    def _split_at_name(self, text: str) -> str:
        """A recording that glued table talk to an address by name ("…чипсы. Это на
        следующий. Лёня, ты знаешь…", live 2026-10-03): marks the talk before the
        sentence with the name as background, so the LLM answers the address.
        Text unchanged if the name is in the first sentence (or absent)."""
        sentences = [x for x in re.split(r'(?<=[.!?…])\s+', text.strip()) if x]
        for i, sent in enumerate(sentences):
            if self._has_robot_name(sent):
                if i == 0:
                    return text
                before = ' '.join(sentences[:i])
                return (f'[Перед обращением звучал разговор, возможно не тебе: «{before}»]\n'
                        f'{" ".join(sentences[i:])}')
        return text

    def _addressing_reason(self, text: str) -> str | None:
        """Why a voice utterance counts as addressed to the robot, None if it doesn't.

        'name' — the robot's name anywhere in the phrase;
        'gaze' — the interlocutor is looking the robot in the eye right now;
        'wake' — no face in view yet and the wake word fired < wake_grace_sec ago
                 (the start of a dialogue, before the head found the face).
        Everything else — a face looking away, or no face and no recent wake word
        (person walked off mid-dialogue) — is not the robot's business.
        """
        if self._has_robot_name(text):
            return 'name'
        now = time.monotonic()
        with self._lock:
            looking = self._looking_at_robot
            if now - self._social_ctx_ts > self._GAZE_CTX_STALE_SEC:
                looking = None      # identity_manager asleep/silent — no gaze data
            since_wake = now - self._wake_ts
        if looking:
            return 'gaze'
        if looking is None and since_wake < self.wake_grace_sec:
            return 'wake'
        return None

    def _wake_cb(self, msg: Bool):
        if msg.data:
            with self._lock:
                self._wake_ts = time.monotonic()

    def other_command_callback(self, msg: String):
        """A phrase by someone other than the current interlocutor (voice_detector
        SV rejected the voice → STT → voice_command_other). Multi-person talk:
        answered only when it calls the robot by name."""
        self.command_callback(msg, other_speaker=True)

    def command_callback(self, msg: String, other_speaker: bool = False):
        text = msg.data.strip()
        if not text:
            return  # voice_detector publishes an empty string on silence — ignore it

        # Addressee filter: no name, not looking at the robot, not right after the
        # wake word — ignore.
        # Runs before the lock: _addressing_reason takes it itself.
        addressing = self._addressing_reason(text)
        if other_speaker:
            # SV says: not the interlocutor's voice. The lips may say otherwise —
            # checked off the executor thread (the evidence arrives via a callback)
            threading.Thread(target=self._other_speaker_check,
                             args=(text, time.time(), addressing), daemon=True).start()
            return
        if addressing is None:
            with self._lock:
                looking = self._looking_at_robot
            self.get_logger().info(
                f'LLM: speech not addressed to the robot (gaze={looking}, no name, '
                f'no recent wake word) — skipping: "{text[:60]}"')
            return
        else:
            self.get_logger().info(f'LLM: addressed to the robot ({addressing})')

        if addressing == 'gaze' and not other_speaker:
            # Gaze alone lets in speech by someone off-screen while the face in view
            # just looks this way (live 2026-10-04: "You change one year." — lips of
            # the face in view still). Check the lips first; the wait is off the
            # executor thread (the evidence arrives through a callback).
            threading.Thread(target=self._gaze_lips_check, args=(text, time.time()),
                             daemon=True).start()
            return
        self._accept_command(text, addressing, other_speaker)

    # /speaker_evidence (identity_manager) waited for after the STT text: it comes
    # ~0.5 s after the phrase's end + lip analysis, usually within STT time
    _LIPS_WAIT_SEC   = 1.5
    _LIPS_MATCH_SEC  = 5.0   # evidence of a phrase that ended this long before the text

    def _speaker_evidence_cb(self, msg: String):
        try:
            ev = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if ev.get('t_end') is None:
            return
        with self._lips_cond:
            self._lips_evidence.append(ev)
            self._lips_cond.notify_all()

    def _await_lips(self, t_text: float, sv_rejected: bool = False) -> dict | None:
        """Lip evidence of the phrase whose STT text arrived at t_text (wall clock),
        None if it doesn't come in _LIPS_WAIT_SEC (no face / lips off). sv_rejected
        picks the evidence of an SV-rejected phrase (voice_command_other) or not."""
        deadline = time.monotonic() + self._LIPS_WAIT_SEC
        with self._lips_cond:
            while True:
                for ev in reversed(self._lips_evidence):
                    if (t_text - self._LIPS_MATCH_SEC <= ev['t_end'] <= t_text + 0.1
                            and bool(ev.get('sv_rejected')) == sv_rejected
                            and ev.get('segment_id') != self._lips_used_seg):
                        self._lips_used_seg = ev.get('segment_id')
                        return ev
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._lips_cond.wait(left)

    def _other_speaker_check(self, text: str, t_text: float, addressing: str | None):
        """A phrase SV rejected (not the anchor's voice). If the face in view spoke it
        — lips moved with the speech, nobody else's did, it looked at the robot —
        it IS the interlocutor and SV was wrong (live 2026-10-05: the anchor stayed
        on the previous person). Then it's handled as theirs and its voice joins
        the live SV gallery (/voice/sv_confirm). Otherwise another person: answered
        only when calling the robot by name."""
        ev = self._await_lips(t_text, sv_rejected=True)
        if ev and ev.get('who') == 'primary' and ev.get('lips') == 'speaking' and ev.get('gaze'):
            self.get_logger().info(
                f'LLM: SV said another voice, but the face in view spoke it '
                f'(lips excess={ev.get("excess")}, gaze) — the interlocutor')
            self._sv_confirm_pub.publish(String(data=json.dumps(
                {'segment_id': ev.get('segment_id')})))
            self._accept_command(text, addressing if addressing == 'name' else 'gaze',
                                 False, lips_who='primary')
            return
        # The gaze / wake word belong to the interlocutor, not to this voice
        if addressing != 'name':
            self.get_logger().info(
                f'LLM: another person, no robot name — skipping: "{text[:60]}"')
            return
        self.get_logger().info('LLM: another person called the robot by name')
        self._accept_command(text, addressing, True)

    def _gaze_lips_check(self, text: str, t_text: float):
        """Gaze passed the gate — veto it if the lips show someone else spoke:
        the face in view stayed silent ('offscreen') or another face talked
        ('other_face'). 'unknown' lips don't veto (shadow run 2026-10-03/04: this
        rule blocked 3 of ~50 gaze phrases, all 3 not addressed; requiring a positive
        'primary' would have cut 2 real questions). An unknown face is stricter:
        gaze introduces only with lips positively matching ('primary')."""
        ev  = self._await_lips(t_text)
        who = ev.get('who') if ev else None
        if who in ('offscreen', 'other_face'):
            self.get_logger().info(
                f'LLM: gaze vetoed by lips (who={who}, lips={ev.get("lips")}, '
                f'excess={ev.get("excess")}) — skipping: "{text[:60]}"')
            return
        self.get_logger().info(f'LLM: gaze confirmed by lips (who={who or "no data"})')
        self._accept_command(text, 'gaze', False, lips_who=who)

    def _accept_command(self, text: str, addressing: str, other_speaker: bool,
                        lips_who: str | None = None):
        """An utterance that passed the addressee gate → introduction or the LLM."""
        # Parsing the voice direction hint — independent of, and before, the LLM request
        # (doesn't block/slow down the response). Published on EVERY addressed
        # utterance (even direction='none') — this is both a hint and the
        # "an utterance happened" trigger for face-search retry in behavior_manager_node.
        direction, phrase = _parse_direction_hint(text)
        hint_msg = String()
        hint_msg.data = json.dumps(
            {'direction': direction, 'phrase': phrase, 'text': text[:80]}, ensure_ascii=False)
        self._direction_hint_pub.publish(hint_msg)

        if addressing == 'name':
            split = self._split_at_name(text)
            if split != text:
                self.get_logger().info('LLM: name mid-recording — earlier talk marked as background')
                text = split

        with self._lock:
            if self._introducing:
                self.get_logger().debug('LLM: /introducing=True — command ignored')
                return
            # The introduction is for the unknown face in view — another voice is
            # not necessarily that face, so it just gets an LLM answer
            intro_on_address = (not other_speaker and self._introduce_on_address and
                                time.monotonic() - self._social_ctx_ts <= self._GAZE_CTX_STALE_SEC)
        if intro_on_address:
            # Gaze alone is too weak for a stranger — they face the robot while talking
            # to someone else, or another person speaks (live 2026-10-03: "Не мерить."
            # with the watcher's lips still). Gaze introduces only when the lips show
            # that very face spoke; otherwise the name or the wake word is needed.
            if addressing == 'gaze' and lips_who != 'primary':
                self.get_logger().info(
                    f'LLM: unknown person, gaze without their lips speaking '
                    f'(who={lips_who or "no data"}) — skipping: "{text[:60]}"')
                return
            # An unknown person spoke to the robot: identity_manager answers with the
            # introduction ("Привет! Мы ещё не знакомы...") — not the LLM.
            self.get_logger().info('LLM: unknown person addressed the robot → introduction')
            self._speech_addressed_pub.publish(String(data=text))
            return

        tag = self._speaker_tag(other_speaker)
        with self._lock:
            if self._processing:
                replaced = self._pending.put((text, tag, other_speaker))
                self.get_logger().info(
                    f'LLM is busy — command queued{" (replaced the previous one)" if replaced else ""}: '
                    f'"{text[:60]}"')
                return
            self._processing = True
        # Cancel any TTS chunks of the previous response still playing/pending
        _cancel = Bool()
        _cancel.data = True
        self._tts_cancel_pub.publish(_cancel)
        threading.Thread(
            target=self._query_llm, args=(text,),
            kwargs={'addressing': addressing, 'speaker_tag': tag,
                    'other_speaker': other_speaker},
            daemon=True,
        ).start()

    def _speaker_tag(self, other_speaker: bool) -> str | None:
        """History prefix naming who said this turn. Another person's phrase is
        always tagged; the interlocutor's — only once someone else has spoken in
        this history (until then it's a plain one-on-one dialogue)."""
        with self._lock:
            sp = self._speaker
            if (time.time() - sp.get('ts', 0.0) >= _SPEAKER_FRESH_SEC
                    or bool(sp.get('other_speaker')) != other_speaker):
                sp = {}   # stale, or the voice-id of a different phrase
            multi = any(m.get('role') == 'user'
                        and str(m.get('content', '')).startswith(_SPEAKER_TAG_PREFIX)
                        for m in self.history)
            face_name = (self._person_context or {}).get('name') or ''
        name = sp.get('name', '') if sp.get('confidence') == 'high' else ''
        if other_speaker:
            return _speaker_tag(name or 'другой человек, голос не опознан')
        if multi:
            return _speaker_tag(name or face_name or 'собеседник')
        return None

    def _telegram_ask_cb(self, msg: String):
        """Request from telegram_bridge_node: JSON {request_id, text, person_ctx?, image_base64?}.

        Uses the same _query_llm pipeline as voice_command — all tool calls,
        memory and SmartHome context work. person_ctx is injected from the Telegram profile.
        image_base64 (optional) — a photo sent by the user in Telegram;
        it only goes to the LLM on the current turn (see _build_user_content/_query_llm),
        self.history keeps a text placeholder.
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'telegram_ask: invalid JSON: {e}')
            return
        req_id     = data.get('request_id', '').strip()
        text       = data.get('text', '').strip()
        person_ctx = data.get('person_ctx') or None
        image_b64  = data.get('image_base64') or None
        if not req_id or not text:
            return
        if person_ctx and not isinstance(person_ctx, dict):
            person_ctx = None
        if image_b64 and not isinstance(image_b64, str):
            image_b64 = None

        with self._lock:
            if self._introducing or self._processing:
                busy = String()
                busy.data = json.dumps(
                    {'request_id': req_id, 'text': '__busy__'}, ensure_ascii=False)
                self._tg_resp_pub.publish(busy)
                self.get_logger().info(
                    f'telegram_ask: LLM is busy — req_id={req_id[:8]}')
                return
            self._processing = True
            self._tg_req_id  = req_id

        threading.Thread(
            target=self._query_llm,
            args=(text,),
            kwargs={'person_ctx_override': person_ctx, 'image_b64': image_b64,
                    'source': 'telegram'},
            daemon=True,
        ).start()
        self.get_logger().info(
            f'telegram_ask: req_id={req_id[:8]}, text="{text[:60]}"'
            + (f', +photo ({len(image_b64)} b64 chars)' if image_b64 else ''))

    # ── Main LLM request ────────────────────────────────────────────────

    def _query_llm(self, user_text: str, person_ctx_override: dict | None = None,
                   image_b64: str | None = None, source: str = 'voice',
                   addressing: str = '', speaker_tag: str | None = None,
                   other_speaker: bool = False):
        _t0 = time.time()
        if source != 'voice':
            addressing = None   # no addressee question outside voice
        self._queried_since_activate = True
        denied = self._tg_denied_tools if source == 'telegram' else frozenset()
        tools  = filter_tools(TOOLS, denied)
        try:
            with self._lock:
                person_ctx = (
                    person_ctx_override
                    if person_ctx_override is not None
                    else self._person_context
                )
                oh_schema         = self._oh_schema
                # Working memory (time/place/mode) — always in context.
                # Episodic (the interlocutor's personal history) — only when
                # person_id is known, otherwise it would reveal name/facts before identification.
                _pid = (person_ctx or {}).get('person_id')
                memory_context    = (self._memory_context if _pid is not None
                                     else _strip_episodic_memory(self._memory_context))
                scene_ctx         = self._scene_context
                face_search_ctx   = self._face_search_status
                # Only a fresh voice-id belongs to this utterance (it arrives before
                # the STT text: same recording, lookup is faster than STT)
                # …and only the voice-id of the same kind of phrase: another
                # person's lookup must not describe the interlocutor's turn
                speaker = (self._speaker
                           if addressing is not None
                           and time.time() - self._speaker.get('ts', 0.0) < _SPEAKER_FRESH_SEC
                           and bool(self._speaker.get('other_speaker')) == other_speaker
                           else None)

            system_prompt = build_system_prompt(
                oh_schema, person_ctx, memory_context, scene_ctx, face_search_ctx,
                addressing, speaker)
            if speaker:
                self.get_logger().info(
                    f'Speaker for this turn: {speaker.get("name") or "?"} '
                    f'({speaker.get("confidence")}, sim={speaker.get("similarity")}), '
                    f'in frame: {speaker.get("face_name") or ("face" if speaker.get("face_visible") else "nobody")}')
            if scene_ctx:
                _scene_age = time.time() - scene_ctx.get('updated_at', 0)
                _scene_labels = [o['label'] for o in scene_ctx.get('objects', [])]
                self.get_logger().info(
                    f'Scene context for this turn: person_count={scene_ctx.get("person_count")}, '
                    f'objects={_scene_labels}, age={_scene_age:.1f}s')
            else:
                self.get_logger().info(
                    'Scene context for this turn: empty (scene_manager_node not responding?)')

            # Who said it, when several people talk (see _speaker_tag)
            turn_text = f'{speaker_tag}{user_text}' if speaker_tag else user_text
            self.history.append({'role': 'user', 'content': turn_text})
            # Trim the history: count user messages as turns (not raw entries).
            # One turn with a tool call = 3-4 entries, so a raw count would be wrong.
            while sum(1 for m in self.history if m['role'] == 'user') > self.history_max:
                self.history.pop(0)
                while self.history and self.history[0]['role'] != 'user':
                    self.history.pop(0)

            messages = [{'role': 'system', 'content': system_prompt}]
            messages += self.history if self.keep_history else \
                [{'role': 'user', 'content': turn_text}]
            if image_b64:
                # self.history holds a text placeholder (see the append above) —
                # the image is only substituted into the outgoing messages; the last
                # user message is always this very turn.
                messages[-1] = {**messages[-1],
                                'content': _build_user_content(turn_text, image_b64)}

            payload = {
                'model':       self.model,
                'messages':    messages,
                'tools':       tools,
                'stream':      False,
                'temperature': self.temperature,
                'max_tokens':  self.max_tokens,
                'chat_template_kwargs': {'enable_thinking': False},
            }

            full_content, api_tool_calls_r1 = self._stream_with_tts(payload, echo_of=user_text)
            if not api_tool_calls_r1 and _is_user_echo(full_content.strip(), user_text):
                # Parroted the user (not spoken — see echo_of). One retry with a nudge,
                # through the normal R1 path: with it the model does what was asked
                # (e.g. calls look_direction). The Qwen template only allows a system
                # message first — hence a user-role note.
                self.get_logger().warn(
                    f'LLM parroted the user: "{full_content.strip()[:60]}" — retrying R1')
                payload = dict(payload)
                payload['messages'] = messages + [{
                    'role': 'user',
                    'content': '[Системная заметка]: ты только что дословно повторил реплику '
                               'пользователя — так нельзя. Не повторяй его слова, ответь ему '
                               'по существу или выполни просьбу инструментом.'}]
                full_content, api_tool_calls_r1 = self._stream_with_tts(payload, echo_of=user_text)
            response_msg = {
                'role':       'assistant',
                'content':    full_content,
                'tool_calls': api_tool_calls_r1,
            }

            # ── Handling tool calls ──────────────────────────────────────────
            tool_calls = response_msg.get('tool_calls', [])
            # Fallback: Qwen3 sometimes emits tool calls as text in content
            if not tool_calls:
                content_text = response_msg.get('content', '')
                tool_calls = _extract_text_tool_calls(content_text)
                if tool_calls:
                    self.get_logger().info(
                        f'Fallback: extracted {len(tool_calls)} tool call(s) from text')
            if tool_calls:
                # id/type are mandatory on every tool call once this message
                # ends up in self.history and goes out in the next request (see
                # the _normalize_tool_calls docstring). The same call also fixes
                # response_msg['tool_calls'] in case of the text fallback above —
                # otherwise an assistant with empty tool_calls would end up in
                # history right before the tool results.
                tool_calls = _normalize_tool_calls(tool_calls)
                response_msg['tool_calls'] = tool_calls
                self.history.append(response_msg)

                tool_results    = []
                speak_texts     = []   # speak_text from the arguments of action tools
                needs_llm_reply = False  # True if a tool returns data (a query)
                any_tool_failed = False  # True if at least one tool returned success:False

                # ── Running tool calls in parallel ──────────────────────────────
                # The meta tool (set_voice_style) is instant;
                # HTTP tools (items_control × N, get_weather) can run
                # simultaneously — saves time when there is more than 1 action tool.
                _QUERY_FNS = frozenset(('get_openhab_states', 'search_openhab_items',
                                        'web_search', 'get_weather', 'search_memory',
                                        'look_and_describe', 'look_direction'))

                def _exec_one(tc):
                    fn   = tc['function']['name']
                    args = tc['function'].get('arguments', {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    args = dict(args)
                    st   = args.pop('speak_text', None)  # TTS meta-parameter
                    res, allowed = self._run_tool(fn, args, source, denied)
                    if not allowed:
                        st = None
                    return fn, st, res, tc['id']

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(len(tool_calls), 4),
                    thread_name_prefix='llm_tc',
                ) as _pool:
                    # Results in the original order of tool_calls
                    _tc_results = list(_pool.map(_exec_one, tool_calls))

                for fn_name, speak_text, result, tc_id in _tc_results:
                    if speak_text and speak_text.strip():
                        speak_texts.append(speak_text.strip())
                    if fn_name in _QUERY_FNS:
                        needs_llm_reply = True
                    if isinstance(result, dict) and result.get('success') is False:
                        any_tool_failed = True
                    tool_results.append({
                        'role':         'tool',
                        'tool_call_id': tc_id,
                        'content':      json.dumps(result, ensure_ascii=False),
                    })

                self.history.extend(tool_results)

                # Deduplication: the LLM may give the same speak_text to several tool calls
                # (e.g. items_control for a Dimmer + Color of the same room).
                _seen: set = set()
                speak_texts = [st for st in speak_texts
                               if not (_seen.__contains__(st) or _seen.add(st))]

                # Tools for which the BT itself speaks the text (via robot_events).
                # R2 isn't needed for them — it creates duplicate speech and repeated Speak goals.
                _BT_SPEECH_TOOLS = frozenset({'robot_control', 'broadcast_message'})
                any_bt_speech_tool = any(
                    tc['function']['name'] in _BT_SPEECH_TOOLS for tc in tool_calls
                )

                if speak_texts and not needs_llm_reply and not any_tool_failed:
                    # All tools succeeded, the LLM already wrote a reply — no second request needed
                    final_text = ' '.join(speak_texts)
                    self.history.append({'role': 'assistant', 'content': final_text})
                    self.get_logger().info(
                        f'Reply from speak_text in {time.time()-_t0:.1f}s: "{final_text}"')
                    with self._lock:
                        _vs = self._voice_style.get('emotion', '')
                        _tg = self._tg_req_id
                    if _tg:
                        for st in speak_texts:
                            self._tg_stream_partial(st, _tg)
                    else:
                        self._send_tts_chunk(final_text, _vs)
                    self._publish_response('', streamed=True)
                elif any_bt_speech_tool and not needs_llm_reply and not any_tool_failed:
                    # The BT handles speech via robot_events — R2 not needed.
                    # We do NOT call _publish_response: otherwise person_present=True would trigger
                    # another BT tick and a second Speak goal.
                    # If a tool failed (any_tool_failed=True) — fall through to else → R2 with the error.
                    self.get_logger().info(
                        f'robot_control/broadcast_message: skipping R2 — '
                        f'the BT handles speech ({time.time()-_t0:.1f}s)')
                    # Exception: goodbye. The BT has no Speak for it (unlike sleep →
                    # /robot/sleep_text) — it only tears the session down, and after
                    # go_idle person_present=False would kill a BT Speak anyway. The
                    # farewell phrase goes straight to TTS, past the BT gate.
                    for tc in tool_calls:
                        if tc['function']['name'] != 'robot_control':
                            continue
                        _args = tc['function'].get('arguments', {})
                        if isinstance(_args, str):
                            _args = json.loads(_args)
                        if _args.get('action') == 'goodbye':
                            bye = (_args.get('text') or _args.get('speak_text') or '').strip()
                            if bye:
                                with self._lock:
                                    _vs = self._voice_style.get('emotion', '')
                                self.get_logger().info(f'Goodbye phrase → TTS: "{bye}"')
                                self._send_tts_chunk(bye, _vs)
                else:
                    # The tools returned data — the LLM is needed to form the reply.
                    # web_search removed from tools: if the search already ran (successfully or
                    # not), retrying is not needed — the LLM should form a text reply.
                    # Only meta tools in R2: setting emotion/voice.
                    # Action tools (save_memory, items_control, robot_control,
                    # get_weather, web_search) are not allowed in R2 — there is no new data
                    # from the user there, the LLM would hallucinate them.
                    # items_control is included: after search_openhab_items in R1 the LLM must
                    # be able to call it in R2 via a proper tool-call API.
                    tools_r2 = [t for t in tools
                                if t['function']['name'] in (
                                    'set_voice_style', 'items_control')]
                    # Injecting a brief reminder right before the final answer:
                    # the LLM must answer the user's specific question,
                    # not recount every field from the tool result.
                    _r2_reminder = (
                        'ОБЯЗАТЕЛЬНО дай короткий разговорный ответ пользователю — 1-2 предложения. '
                        'Не начинай ответ с повтора или перефразировки вопроса — отвечай сразу по существу. '
                        'Если действие уже выполнено — продолжи диалог естественно, не упоминая факт сохранения. '
                        'Используй только те данные из результата инструмента, '
                        'которые отвечают на конкретный вопрос пользователя. '
                        'Молчать нельзя — нужно что-то сказать.'
                    )
                    payload2 = {
                        'model':    self.model,
                        'messages': (
                            [{'role': 'system', 'content': system_prompt}]
                            + _flatten_tool_history(self.history, _r2_reminder)
                        ),
                        'tools':       tools_r2,
                        'stream':      False,
                        'temperature': self.temperature,
                        'max_tokens':  128,
                        'chat_template_kwargs': {'enable_thinking': False},
                    }
                    r2_content, api_tc_r2 = self._stream_with_tts(payload2)
                    resp2 = {'role': 'assistant',
                             'content': r2_content, 'tool_calls': api_tc_r2}

                    # Check for tool calls in the second response (proper API or text format)
                    tool_calls2 = resp2.get('tool_calls', [])
                    if not tool_calls2:
                        c2 = resp2.get('content', '')
                        tool_calls2 = _extract_text_tool_calls(c2)
                        if tool_calls2:
                            self.get_logger().info(
                                f'Fallback R2: {len(tool_calls2)} tool call(s) from text')

                    _R2_ALLOWED = frozenset(
                        ('set_voice_style', 'items_control'))

                    if tool_calls2:
                        # Only run the meta tools and items_control.
                        # Collect speak_text: if R2 called set_voice_style with only speak_text
                        # and no content — use speak_text directly, without R3.
                        # id/type + a pair of 'tool' messages for every tool_call are mandatory —
                        # see the _normalize_tool_calls docstring and R1 above (same class of bug).
                        tool_calls2 = _normalize_tool_calls(tool_calls2)
                        resp2['tool_calls'] = tool_calls2
                        self.history.append(resp2)
                        _r2_speak: list[str] = []
                        _tool_results2: list[dict] = []
                        for tc in tool_calls2:
                            fn2   = tc['function']['name']
                            if fn2 not in _R2_ALLOWED:
                                self.get_logger().warn(
                                    f'R2 text-fallback: skipping disallowed tool {fn2}')
                                _tool_results2.append({
                                    'role': 'tool', 'tool_call_id': tc['id'],
                                    'content': json.dumps(
                                        {'success': False, 'error': 'not allowed in R2'}),
                                })
                                continue
                            args2 = tc['function'].get('arguments', {})
                            if isinstance(args2, str):
                                args2 = json.loads(args2)
                            args2 = dict(args2)
                            st2 = args2.pop('speak_text', None)
                            res2, allowed2 = self._run_tool(fn2, args2, source, denied, 'R2')
                            if allowed2 and st2 and st2.strip():
                                _r2_speak.append(st2.strip())
                            _tool_results2.append({
                                'role': 'tool', 'tool_call_id': tc['id'],
                                'content': json.dumps(res2, ensure_ascii=False),
                            })
                        self.history.extend(_tool_results2)
                        # Text response: content first, fallback — speak_text of the meta tools
                        final_text = _strip_tool_blocks(resp2.get('content', '').strip())
                        _r2_from_speak = False
                        if not final_text and _r2_speak:
                            final_text = ' '.join(_r2_speak)
                            _r2_from_speak = True
                            self.get_logger().info(
                                f'R2 speak_text from meta tool → "{final_text[:60]}"')
                        if final_text:
                            self.history.append({'role': 'assistant', 'content': final_text})
                            self.get_logger().info(
                                f'Final R2 response in {time.time()-_t0:.1f}s: "{final_text}"')
                            if _r2_from_speak:
                                # The text hasn't been sent to TTS yet — send it
                                with self._lock:
                                    _vs2 = self._voice_style.get('emotion', '')
                                    _tg2 = self._tg_req_id
                                if _tg2:
                                    self._tg_stream_partial(final_text, _tg2)
                                else:
                                    self._send_tts_chunk(final_text, _vs2)
                        else:
                            # The LLM returned an empty R2 with no speak_text either — a rare case, R3 needed
                            self.get_logger().warn(
                                'R2: empty content and no speak_text — running R3')
                            payload3 = {
                                'model':    self.model,
                                'messages': (
                                    [{'role': 'system', 'content': system_prompt}]
                                    + _flatten_tool_history(
                                        self.history,
                                        'Обязательно ответь пользователю одним коротким '
                                        'разговорным предложением — молчать нельзя.')
                                ),
                                'tools':       [],
                                'stream':      True,
                                'temperature': self.temperature,
                                'max_tokens':  128,
                                'chat_template_kwargs': {'enable_thinking': False},
                            }
                            try:
                                r3_content, _ = self._stream_with_tts(payload3)
                                final_text = _strip_tool_blocks(r3_content.strip())
                                if final_text:
                                    self.history.append(
                                        {'role': 'assistant', 'content': final_text})
                                    self.get_logger().info(
                                        f'Final R3 response in {time.time()-_t0:.1f}s: '
                                        f'"{final_text}"')
                                else:
                                    self.get_logger().warn('R3 empty after R2 set_voice_style — staying silent')
                            except Exception as _e3:
                                self.get_logger().warn(f'R3 error (after R2 set_voice_style): {_e3}')
                        # The text has already been streamed to TTS via _stream_with_tts; the BT will handle the gesture
                        self._publish_response('', streamed=True)
                    else:
                        final_text = _strip_tool_blocks(resp2.get('content', '').strip())
                        if final_text:
                            self.history.append({'role': 'assistant', 'content': final_text})
                            self.get_logger().info(
                                f'Final response in {time.time()-_t0:.1f}s: "{final_text}"')
                            self._publish_response('', streamed=True)
                        else:
                            # R2 is empty — if there were only action tools (not query),
                            # do a minimal R3 requiring a reply
                            if not needs_llm_reply:
                                self.get_logger().warn('R2 empty after an action tool — trying R3')
                                payload3 = {
                                    'model':    self.model,
                                    'messages': (
                                        [{'role': 'system', 'content': system_prompt}]
                                        + _flatten_tool_history(self.history,
                                                                'Действие выполнено. Ответь пользователю одним коротким предложением — '
                                                                'продолжи разговор естественно, не упоминая факт сохранения.')
                                    ),
                                    'tools':       [],
                                    'stream':      False,
                                    'temperature': self.temperature,
                                    'max_tokens':  64,
                                    'chat_template_kwargs': {'enable_thinking': False},
                                }
                                try:
                                    r3_content, _ = self._stream_with_tts(payload3)
                                    final_text = _strip_tool_blocks(r3_content.strip())
                                    if final_text:
                                        self.history.append(
                                            {'role': 'assistant', 'content': final_text})
                                        self.get_logger().info(
                                            f'Final R3 response in {time.time()-_t0:.1f}s: "{final_text}"')
                                        self._publish_response('', streamed=True)
                                    else:
                                        self.get_logger().warn('Empty final LLM response (R3) — staying silent')
                                except Exception as _e3:
                                    self.get_logger().warn(f'R3 error: {_e3}')
                            else:
                                self.get_logger().warn('Empty final LLM response — staying silent')

            elif addressing is not None and _is_not_addressed(response_msg.get('content', '')):
                # The LLM judged the utterance as not meant for the robot: say nothing,
                # drop the user turn from history and don't publish /llm_response
                # (it would refresh the BM's voice presence and keep the dialogue alive).
                self.history.pop()
                self.get_logger().info(
                    f'LLM: {_NOT_ADDRESSED_MARKER} — utterance judged not addressed to the '
                    f'robot ({time.time()-_t0:.1f}s): "{user_text[:60]}"')
            else:
                text = response_msg.get('content', '').strip()
                if _is_user_echo(text, user_text):
                    # Parroted again after the retry above: stay silent and keep the
                    # parrot out of the history — otherwise the model copies it on
                    # every next turn.
                    self.history.pop()
                    self.get_logger().warn(f'LLM parroted the user again: "{text[:60]}" — staying silent')
                    return
                self.history.append({'role': 'assistant', 'content': text})
                self.get_logger().info(f'Text response in {time.time()-_t0:.1f}s: "{text}"')
                # The text has already been streamed to TTS; the BT only handles the emotion/gesture
                self._publish_response('', streamed=True)

        except requests.exceptions.ConnectionError as e:
            self.get_logger().error(f'LLM unavailable (both servers): {e}')
            self._publish_response('Извини, не могу связаться с сервером обработки. Попробуй позже.')
        except requests.exceptions.Timeout:
            self.get_logger().error(f'LLM timeout after {time.time()-_t0:.1f}s (limit={self.timeout_sec}s)')
            self._publish_response('Извини, сервер слишком долго не отвечает. Попробуй задать вопрос покороче.')
        except Exception as e:
            self.get_logger().error(f'LLM error: {e}')
            self._publish_response('Извини, произошла ошибка. Попробуй ещё раз.')
        finally:
            with self._lock:
                self._processing = False
            self._dispatch_pending()

    def _dispatch_pending(self):
        """Runs the voice command that arrived while we were busy (if still fresh).

        No /tts_cancel_queue here, unlike command_callback: the previous reply's
        TTS may still be playing and the user should hear it — Speak goals queue.
        """
        item = self._pending.take()
        if not item:
            return
        text, tag, other = item if isinstance(item, tuple) else (item, None, False)
        with self._lock:
            if self._introducing:
                self.get_logger().info(f'Queued command dropped (/introducing): "{text[:60]}"')
                return
            if self._processing:          # a Telegram request slipped in — keep waiting
                self._pending.put(text)
                return
            self._processing = True
        self.get_logger().info(f'Running queued command: "{text[:60]}"')
        threading.Thread(target=self._query_llm, args=(text,),
                         kwargs={'speaker_tag': tag, 'other_speaker': other},
                         daemon=True).start()

    # ── Executing tool calls ────────────────────────────────────────────

    def _run_tool(self, fn: str, args: dict, source: str, denied: frozenset,
                  stage: str = 'R1') -> tuple[dict, bool]:
        """The ONLY way tools are executed: per-source policy, execution, audit.

        Returns (result, allowed). A tool denied for this source (not offered to
        the model — e.g. Telegram can't move the robot) is refused, not run.
        """
        self.get_logger().info(f'Tool call {stage} [{source}]: {fn}({args})')
        t0 = time.time()
        allowed = fn not in denied
        if allowed:
            res = self._execute_tool(fn, args)
        else:
            res = {'success': False, 'error': f'{fn} недоступен для запросов из {source}'}
        self._audit.record(source, fn, args, res, time.time() - t0)
        self.get_logger().info(f'Tool result {stage}: {res}')
        return res, allowed

    def _execute_tool(self, fn_name: str, args: dict) -> dict:
        if fn_name == 'get_weather':
            return self._tool_get_weather(args)
        elif fn_name == 'items_control':
            return self._tool_items_control(args)
        elif fn_name == 'get_openhab_states':
            return self._tool_get_openhab_states(args)
        elif fn_name == 'search_openhab_items':
            return self._tool_search_openhab_items(args)
        elif fn_name == 'robot_control':
            return self._tool_robot_control(args)
        elif fn_name == 'web_search':
            return self._tool_web_search(args)
        elif fn_name == 'set_voice_style':
            return self._tool_set_voice_style(args)
        elif fn_name == 'save_memory':
            return self._tool_save_memory(args)
        elif fn_name == 'search_memory':
            return self._tool_search_memory(args)
        elif fn_name == 'set_reminder':
            return self._tool_set_reminder(args)
        elif fn_name == 'confirm_reminder':
            return self._tool_confirm_reminder(args)
        elif fn_name == 'broadcast_message':
            return self._tool_broadcast_message(args)
        elif fn_name == 'merge_persons':
            return self._tool_merge_persons(args)
        elif fn_name == 'look_and_describe':
            return self._tool_look_and_describe(args)
        elif fn_name == 'look_direction':
            return self._tool_look_direction(args)
        else:
            return {'error': f'Unknown function: {fn_name}'}

    def _call_vision_model(self, images_b64: list[str], query: str,
                           system_content: str, max_tokens: int = 200) -> dict:
        """A single request to a SEPARATE vision model (self.vision_llm_url, not self.llm_url —
        that one is limited to 1 image per prompt, see the 2026-08-26 history). This model
        accepts 2-4 images at once (verified by the user on the server). A plain
        blocking POST, no streaming — a short self-contained request outside self.history."""
        if not images_b64:
            return {'success': False, 'error': 'Нет ни одного кадра с камер'}
        content = [{'type': 'text', 'text': query}]
        for b64 in images_b64:
            content.append(
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}})
        payload = {
            'model':    self.vision_model,
            'messages': [
                {'role': 'system', 'content': system_content},
                {'role': 'user',   'content': content},
            ],
            'stream':      False,
            'temperature': 0.3,
            'max_tokens':  max_tokens,
        }
        headers = ({'Authorization': f'Bearer {self.vision_bearer_token}'}
                   if self.vision_bearer_token else {})
        try:
            r = requests.post(self.vision_llm_url, json=payload, headers=headers,
                              timeout=(self.connect_timeout, 25.0))
            r.raise_for_status()
            data = r.json()
            description = (data['choices'][0]['message']['content'] or '').strip()
        except Exception as e:
            self.get_logger().warn(f'vision model ({len(images_b64)} frame(s)): error {e}')
            return {'success': False, 'error': str(e)}
        if not description:
            return {'success': False, 'error': 'Vision-модель не дала ответа'}
        return {'success': True, 'description': description}

    _VISION_SYS_HINT = (
        'Отвечай кратко и по делу, простым текстом без markdown-разметки, '
        'списков и заголовков — 2-3 предложения максимум. Тебе может быть показано '
        'несколько кадров одной и той же сцены с разных камер или в разные моменты '
        'времени — используй все вместе, один кадр может быть смазан или не в фокусе.'
    )

    # The eye cameras run at 1280x960 for lip reading (face_tracker); the vision
    # model gets 640 px wide frames — 4x fewer image tokens, same as before.
    _LLM_IMAGE_MAX_W = 640

    @classmethod
    def _jpeg_b64_for_llm(cls, jpeg: bytes) -> str:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None and img.shape[1] > cls._LLM_IMAGE_MAX_W:
            scale = cls._LLM_IMAGE_MAX_W / img.shape[1]
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                jpeg = buf.tobytes()
        return base64.b64encode(jpeg).decode()

    def _tool_look_and_describe(self, args: dict) -> dict:
        """A snapshot RIGHT NOW from BOTH eye cameras (left and right — in case
        one is out of focus/blurry) → analysis via a separate vision model."""
        query = (args.get('query') or '').strip() or 'Опиши коротко и по делу, что видишь.'
        with self._lock:
            left, right = self._latest_eye_jpeg, self._latest_eye_jpeg_right
        images_b64 = [self._jpeg_b64_for_llm(j) for j in (left, right) if j]
        if not images_b64:
            return {'success': False, 'error': 'Камера недоступна или кадр ещё не пришёл'}
        return self._call_vision_model(images_b64, query, self._VISION_SYS_HINT)

    # How long the physical turn actually takes (see ExecuteRobotCommand._do_head/
    # _send_head_cmd in behavior_manager_node.py) — the head starts 0.2s
    # after the event, the turn itself takes ~1.2-1.8s. With the torso (scope=partial/full) a bit longer.
    _LOOK_SETTLE_HEAD_SEC  = 2.0
    _LOOK_SETTLE_TORSO_SEC = 2.3
    _LOOK_FRAME_GAP_SEC    = 1.0   # interval between 2 captures along the turn

    def _tool_look_direction(self, args: dict) -> dict:
        """Turns the head/torso (via the same /robot_events as robot_control) +
        WAITS for the turn to actually finish + 2 shots from each eye (from each
        camera — before and at the end of the turn) + analysis of all frames by one vision model.
        In a single tool call, synchronously in this same thread — this guarantees the correct
        order (turn → shots), unlike separate robot_control+look_and_describe calls,
        which the ThreadPoolExecutor runs in parallel (see _exec_one/_pool.map above)."""
        pan   = float(args.get('pan', 0) or 0)
        tilt  = float(args.get('tilt', 0) or 0)
        scope = str(args.get('scope') or 'head').strip().lower()
        if scope not in ('head', 'partial', 'full'):
            scope = 'head'
        query = (args.get('query') or '').strip() or 'Опиши коротко и по делу, что видишь.'

        event = {'action': 'head', 'pan': pan, 'tilt': tilt, 'scope': scope, 'priority': 10}
        msg = String()
        msg.data = json.dumps(event, ensure_ascii=False)
        self.event_pub.publish(msg)
        self.get_logger().info(f'look_direction: turn pan={pan:+.0f}° tilt={tilt:+.0f}° scope={scope}')

        settle = self._LOOK_SETTLE_HEAD_SEC if scope == 'head' else self._LOOK_SETTLE_TORSO_SEC
        # 2 captures from both cameras ~1s apart — the first shortly before the end
        # of the turn (in case of overshoot), the second right after settling.
        # A separate vision model accepts 2-4 images per request (unlike
        # self.llm_url, limited to 1 image — see the 2026-08-26 history).
        t1 = max(0.0, settle - self._LOOK_FRAME_GAP_SEC)
        time.sleep(t1)
        with self._lock:
            left_1, right_1 = self._latest_eye_jpeg, self._latest_eye_jpeg_right
        time.sleep(settle - t1)
        with self._lock:
            left_2, right_2 = self._latest_eye_jpeg, self._latest_eye_jpeg_right

        images_b64 = [self._jpeg_b64_for_llm(j)
                      for j in (left_1, right_1, left_2, right_2) if j]
        if not images_b64:
            return {'success': False, 'error': 'Камера недоступна или кадр ещё не пришёл'}

        sys_hint = (
            'Тебе показаны кадры (с обеих камер, в конце поворота головы робота '
            'в нужную сторону), снятые с интервалом около секунды. ' + self._VISION_SYS_HINT
        )
        result = self._call_vision_model(images_b64, query, sys_hint)
        result.update({'pan': pan, 'scope': scope})
        return result

    # Default coordinates — Bødalen, Asker, Norway
    _DEFAULT_LAT  = 59.835
    _DEFAULT_LON  = 10.440
    _DEFAULT_LOC  = 'Bødalen, Asker'
    _YR_UA        = 'InMoov-Robot/1.0 (fedjukevitsh@gmail.com)'

    # yr.no symbols → Russian descriptions (spoken to the user, kept in Russian)
    _YR_SYMBOLS = {
        'clearsky':           'ясно',
        'fair':               'малооблачно',
        'partlycloudy':       'переменная облачность',
        'cloudy':             'пасмурно',
        'fog':                'туман',
        'lightrainshowers':   'небольшой ливень',
        'rainshowers':        'ливень',
        'heavyrainshowers':   'сильный ливень',
        'lightrain':          'небольшой дождь',
        'rain':               'дождь',
        'heavyrain':          'сильный дождь',
        'lightsleet':         'лёгкий мокрый снег',
        'sleet':              'мокрый снег',
        'heavysleet':         'сильный мокрый снег',
        'lightsnow':          'небольшой снег',
        'snow':               'снег',
        'heavysnow':          'сильный снег',
        'lightsnowshowers':   'небольшой снегопад',
        'snowshowers':        'снегопад',
        'thunder':            'гроза',
        'lightrainandthunder': 'дождь с грозой',
    }

    def _tool_get_weather(self, args: dict) -> dict:
        from collections import Counter

        location = (args.get('location') or '').strip() or self._DEFAULT_LOC
        date_arg  = (args.get('date') or '').strip().lower()

        # Determine the target date
        today = datetime.date.today()
        if not date_arg or date_arg in ('today', 'сегодня'):
            target = today
        elif date_arg in ('tomorrow', 'завтра'):
            target = today + datetime.timedelta(days=1)
        else:
            try:
                target = datetime.date.fromisoformat(date_arg)
            except ValueError:
                target = today

        # Geocoding via Nominatim (if not the default location)
        lat, lon = self._DEFAULT_LAT, self._DEFAULT_LON
        resolved_loc = location
        if location.lower() not in ('bødalen, asker', 'bødalen', 'бодален'):
            try:
                geo = requests.get(
                    'https://nominatim.openstreetmap.org/search',
                    params={'q': location, 'format': 'json', 'limit': 1},
                    headers={'User-Agent': self._YR_UA},
                    timeout=5.0,
                )
                if geo.ok and geo.json():
                    g = geo.json()[0]
                    lat = float(g['lat'])
                    lon = float(g['lon'])
                    resolved_loc = g.get('display_name', location).split(',')[0]
            except Exception as e:
                self.get_logger().warn(f'Geocoding failed ({e}), using Bødalen')
                resolved_loc = self._DEFAULT_LOC

        # Request to api.met.no
        try:
            wr = requests.get(
                'https://api.met.no/weatherapi/locationforecast/2.0/compact',
                params={'lat': round(lat, 4), 'lon': round(lon, 4)},
                headers={'User-Agent': self._YR_UA},
                timeout=10.0,
            )
            wr.raise_for_status()
            ts_list = wr.json().get('properties', {}).get('timeseries', [])
        except Exception as e:
            return {'error': f'Ошибка yr.no API: {e}'}

        # Filter by date
        entries = []
        for entry in ts_list:
            if entry['time'][:10] == str(target):
                hour = int(entry['time'][11:13])
                det  = entry['data']['instant']['details']
                n1   = entry['data'].get('next_1_hours', {})
                n6   = entry['data'].get('next_6_hours', {})
                sym  = (n1.get('summary', {}).get('symbol_code')
                        or n6.get('summary', {}).get('symbol_code', ''))
                pre  = (n1.get('details', {}).get('precipitation_amount')
                        or n6.get('details', {}).get('precipitation_amount', 0))
                entries.append({
                    'hour':   hour,
                    'temp':   det.get('air_temperature'),
                    'wind':   det.get('wind_speed'),
                    'symbol': sym,
                    'precip': pre or 0,
                })

        if not entries:
            return {'error': f'Нет данных прогноза для {target} ({resolved_loc})'}

        temps   = [e['temp'] for e in entries if e['temp'] is not None]
        winds   = [e['wind'] for e in entries if e['wind'] is not None]
        precips = [e['precip'] for e in entries]

        # Main description — the most frequent symbol (without the _day/_night/_polartwilight suffix)
        symbols = [e['symbol'].split('_')[0] for e in entries if e['symbol']]
        main_sym  = Counter(symbols).most_common(1)[0][0] if symbols else ''
        main_desc = self._YR_SYMBOLS.get(main_sym, main_sym)

        return {
            'location':    resolved_loc,
            'date':        str(target),
            'description': main_desc,
            'temp_min':    round(min(temps), 1) if temps else None,
            'temp_max':    round(max(temps), 1) if temps else None,
            'wind_max_ms': round(max(winds), 1) if winds else None,
            'precip_mm':   round(sum(precips), 1),
            'source':      'yr.no (MET Norway)',
        }

    def _tool_items_control(self, args: dict) -> dict:
        name  = args.get('name', '')
        state = str(args.get('state', ''))

        # Validate the name against the cache — before calling the API
        with self._lock:
            known = {it['name'] for it in self._oh_items}
        if known and name not in known:
            # Look for similar names (common prefix by '_')
            prefix = name.rsplit('_', 1)[0] if '_' in name else name
            suggestions = sorted(n for n in known if prefix.lower() in n.lower())[:5]
            return {
                'success': False,
                'error': f'Item "{name}" не существует в OpenHAB. '
                         f'Похожие устройства: {suggestions}. '
                         f'Используй точное имя из схемы.',
            }

        try:
            r = requests.post(
                f'{self.openhab_url}/rest/items/{name}',
                data=state,
                headers={'Content-Type': 'text/plain'},
                timeout=5.0,
            )
            if r.ok:
                return {'success': True, 'item': name, 'new_state': state}
            else:
                return {'success': False, 'error': f'HTTP {r.status_code}: {r.text}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _tool_save_memory(self, args: dict) -> dict:
        """Save a note about the person or general knowledge to the DB via /memory/query."""
        key       = args.get('key', '')
        value     = args.get('value', '')
        person_id = args.get('person_id')

        if not key or not value:
            return {'error': 'key and value are required'}

        if person_id is not None:
            req = {'op': 'set_note', 'person_id': int(person_id), 'key': key, 'value': value}
        else:
            req = {'op': 'set_knowledge', 'key': key, 'value': value}

        # Call synchronously from the background thread (the LLM is already in a thread)
        if not self._mem_client.wait_for_service(timeout_sec=2.0):
            return {'error': '/memory/query недоступен'}
        request = MemoryQuery.Request()
        request.request_json = json.dumps(req)
        future = self._mem_client.call_async(request)
        done_event = threading.Event()
        future.add_done_callback(lambda _: done_event.set())
        if not done_event.wait(timeout=5.0):
            return {'error': 'таймаут memory service'}
        try:
            result = json.loads(future.result().response_json)
            self.get_logger().info(
                f'save_memory: {"person_id=" + str(person_id) if person_id else "knowledge"}'
                f' {key}="{value}" → {result}')
            return result
        except Exception as e:
            return {'error': str(e)}

    def _tool_robot_control(self, args: dict) -> dict:
        event    = dict(args)
        event.setdefault('priority', 10)   # voice commands — high priority
        msg      = String()
        msg.data = json.dumps(event, ensure_ascii=False)
        self.event_pub.publish(msg)
        return {'success': True, 'event': args}

    def _tool_web_search(self, args: dict) -> dict:
        query = args.get('query', '')
        event = {'action': 'search', 'query': query, 'priority': 10}

        with self._lock:
            self._search_result_data = None
            self._waiting_for_search = True
            self._search_event.clear()

        msg      = String()
        msg.data = json.dumps(event, ensure_ascii=False)
        self.event_pub.publish(msg)

        # Block the background thread until we get a real result from BM (Tavily)
        got = self._search_event.wait(timeout=30.0)

        with self._lock:
            self._waiting_for_search = False
            result_text = self._search_result_data

        if got and result_text:
            return {'success': True, 'result': result_text, 'query': query}
        return {'success': False, 'error': 'поиск не вернул результат за 30 секунд', 'query': query}

    def _tool_set_voice_style(self, args: dict) -> dict:
        """Buffers a voice preset. tts_node itself will show the same emotion on
        the face for the whole duration of the phrase (see /face_expression_hold) —
        there is no separate mimicry tool call anymore."""
        style = (args.get('style') or 'neutral').lower()
        if style not in ('neutral', 'happy', 'sad', 'surprise', ''):
            self.get_logger().warn(f'set_voice_style: unknown preset "{style}" → neutral')
            style = 'neutral'
        with self._lock:
            self._voice_style = {'emotion': style or 'neutral'}
        self.get_logger().info(f'Voice preset: "{style or "neutral"}"')
        return {'success': True, 'style': style or 'neutral'}

    def _tool_get_openhab_states(self, args: dict) -> dict:
        names = args.get('names', [])
        with self._lock:
            items = self._oh_items
        if not items:
            return {'error': 'OpenHAB bridge недоступен'}
        cache = {it['name']: it for it in items}
        result = []
        for name in names:
            if name in cache:
                it = cache[name]
                result.append({
                    'name':  name,
                    'label': it.get('label', name),
                    'type':  it['type'],
                    'state': it.get('state', 'NULL'),
                })
            else:
                result.append({'name': name, 'error': 'not found'})
        return {'items': result}

    # Russian room names → English substrings for searching in OpenHAB
    _RU_ROOM_MAP = {
        'гостиная':  'living',
        'гостевая':  'guest',
        'спальня':   'bedroom',
        'кухня':     'kitchen',
        'ванная':    'bathroom',
        'туалет':    'wc',
        'прихожая':  'entrance',
        'холл':      'hall',
        'коридор':   'hall',
        'лестница':  'stairs',
        'улица':     'outside',
        'терраса':   'terrace',
        'гардероб':  'wardrobe',
    }

    def _tool_search_openhab_items(self, args: dict) -> dict:
        group_filter  = args.get('group_filter',  '').strip()
        state_filter  = args.get('state_filter',  '').strip()
        name_contains = args.get('name_contains', '').strip().lower()
        # Translate the Russian room name into English for searching
        name_contains = self._RU_ROOM_MAP.get(name_contains, name_contains)

        if not group_filter and not state_filter and not name_contains:
            return {
                'error': (
                    'At least one filter is required. '
                    'Provide group_filter (e.g. "HumiditySensors") and/or '
                    'name_contains (e.g. "Bathroom"). '
                    'Calling without filters is not allowed.'
                )
            }

        with self._lock:
            items = self._oh_items
        if not items:
            return {'error': 'OpenHAB bridge недоступен'}

        result = []
        for it in items:
            # filter by semantic group (exact group-name match)
            if group_filter and group_filter not in it.get('groups', []):
                continue
            # filter by name/label
            if name_contains:
                if (name_contains not in it['name'].lower() and
                        name_contains not in it.get('label', '').lower()):
                    continue
            # filter by state
            if state_filter:
                state = it.get('state', '')
                sf    = state_filter.upper()
                if sf == 'ON':
                    # Switch: state='ON'; Dimmer/Color: state=numeric >0
                    is_on = state.upper() == 'ON'
                    if not is_on:
                        try:
                            is_on = float(state.split()[0]) > 0
                        except (ValueError, IndexError):
                            pass
                    if not is_on:
                        continue
                elif sf == 'OFF':
                    # Switch: state='OFF'; Dimmer: state='0' or '0.0'
                    is_off = state.upper() == 'OFF'
                    if not is_off:
                        try:
                            is_off = float(state.split()[0]) == 0
                        except (ValueError, IndexError):
                            pass
                    if not is_off:
                        continue
                elif sf in ('NULL', 'UNDEF'):
                    if state.upper() != sf:
                        continue
                elif state_filter == '>0':
                    try:
                        if float(state.split()[0]) <= 0:
                            continue
                    except (ValueError, IndexError):
                        continue
            result.append({
                'name':  it['name'],
                'label': it.get('label', it['name']),
                'type':  it['type'],
                'state': it.get('state', 'NULL'),
            })
        return {'count': len(result), 'items': result}

    # ── Callbacks for the OpenHAB bridge ───────────────────────────────────

    def _oh_schema_callback(self, msg: String):
        with self._lock:
            self._oh_schema = msg.data
        self.get_logger().debug('OpenHAB schema received')

    def _oh_items_callback(self, msg: String):
        try:
            items = json.loads(msg.data)
            with self._lock:
                self._oh_items = items
            self.get_logger().debug(f'OpenHAB items updated: {len(items)} devices')
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'Invalid openhab_items JSON: {e}')

    def _eye_camera_cb(self, msg: CompressedImage):
        """Caches the latest frame from the left eye camera for look_and_describe/
        look_direction (a snapshot on request, not a stream — just keep the freshest JPEG)."""
        with self._lock:
            self._latest_eye_jpeg = bytes(msg.data)

    def _eye_camera_right_cb(self, msg: CompressedImage):
        """Same for the right eye camera — see _eye_camera_cb."""
        with self._lock:
            self._latest_eye_jpeg_right = bytes(msg.data)

    def _social_context_cb(self, msg: String):
        """Intercepts looking_at_robot from social_context."""
        try:
            data = json.loads(msg.data)
            with self._lock:
                # looking_at_robot: True/False from identity_manager, None if no face in
                # view right now. None must overwrite: a stale True would keep the gate
                # open after the person turned away or left.
                raw = data.get('looking_at_robot')
                self._looking_at_robot = None if raw is None else bool(raw)
                self._introduce_on_address = bool(data.get('introduce_on_address', False))
                self._social_ctx_ts    = time.monotonic()
        except Exception:
            pass

    def _speaker_cb(self, msg: String):
        """Voice-id of the last utterance from identity_manager (see _build_speaker_block)."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._lock:
            self._speaker = data

    def _scene_context_cb(self, msg: String):
        """Scene summary (objects + people) from scene_manager_node → cache for the system prompt."""
        try:
            ctx = json.loads(msg.data)
            with self._lock:
                self._scene_context = ctx
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'Invalid /scene/objects JSON: {e}')

    def _face_search_status_cb(self, msg: String):
        """Face-search status from behavior_manager_node → cache for the system prompt
        (see _build_face_search_block)."""
        try:
            ctx = json.loads(msg.data)
            with self._lock:
                self._face_search_status = ctx
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'Invalid /behavior/face_search_status JSON: {e}')

    def _person_context_callback(self, msg: String):
        try:
            ctx = json.loads(msg.data)
            with self._lock:
                old_ctx          = self._person_context
                history_snapshot = self.history[:]
            old_name = old_ctx.get('name') if old_ctx else None
            new_name = ctx.get('name')
            # The person changed → publish conversation_end for the previous one
            if old_name and new_name != old_name and history_snapshot:
                self._publish_conversation_end(history_snapshot, old_ctx)
                with self._lock:
                    self.history         = []
                    self._dialogue_lines = []
            with self._lock:
                # Keep the context only when the person is known; otherwise explicitly None.
                # This guarantees person_block is empty in system_prompt until INTERACTING.
                # We don't touch _memory_context: working memory stays in context,
                # personal episodes are cut off in _query_llm by person_id.
                self._person_context = ctx if ctx.get('person_id') else None
            name = ctx.get('name') or 'Незнакомец'
            self.get_logger().debug(f'Person context: {name}')
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'Invalid person_context JSON: {e}')

    def _search_result_callback(self, msg: String):
        search_text = msg.data.strip()
        if not search_text:
            return
        with self._lock:
            if not self._waiting_for_search:
                self.get_logger().warn('Search result received with no active web_search — ignoring')
                return
            self._search_result_data = search_text
            self._waiting_for_search = False
        self._search_event.set()

    def _publish_response(self, text: str, user_text: str = '', streamed: bool = False):
        """Publishes the LLM's response to /llm_response → the BT reads it from the Blackboard and orchestrates speech.

        Format: {text, voice_instruct, streamed, telegram}
        If streamed=True: the text has already been sent to TTS directly;
        the BT only starts the Gesticulation (SpeakBehaviour gets empty text).
        The facial expression no longer goes through the BT/ExpressEmotion — tts_node
        holds it for the whole duration of speech, in sync with voice_instruct (see /face_expression_hold).
        telegram=True: the request came via Telegram — the BM must not set person_present=True.
        """
        if not streamed:
            if not text or not text.strip():
                return
            # Defensive cleanup: strip any leftover tool-call blocks if the regex above didn't catch them
            text = _strip_tool_blocks(text)
            text = _clean_llm_text(text)
            if not text:
                return

        with self._lock:
            voice_preset      = self._voice_style.get('emotion', '')
            self._voice_style     = {'emotion': ''}
            tg_req_id         = self._tg_req_id
            self._tg_req_id   = ''

        payload = {
            'text':           '' if streamed else text,
            # The field kept its old name (voice_instruct) for compatibility with
            # behavior_manager_node.py: it used to hold the instruct phrase for
            # CosyVoice3, now it's the OmniVoice voice-preset name
            # (neutral/happy/sad/surprise).
            'voice_instruct': voice_preset,
            'streamed':       streamed,
            'telegram':       bool(tg_req_id),
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._response_pub.publish(msg)
        if streamed:
            self.get_logger().info(
                '→ BT: (streamed)'
                + (' [TG]' if tg_req_id else '')
                + (f' voice="{voice_preset}"' if voice_preset else '')
            )
        else:
            self.get_logger().info(
                f'→ BT: "{text[:70]}{"..." if len(text) > 70 else ""}"'
                + (' [TG]' if tg_req_id else '')
                + (f' voice="{voice_preset}"' if voice_preset else '')
            )

        # Forward the response to Telegram if the request came via /telegram_ask
        if tg_req_id:
            # streamed=True: the text was already delivered via partial chunks, send an empty "stop" signal
            # streamed=False: the text is an error string (LLM unavailable, etc.)
            tg_text = '' if streamed else text
            tg_resp = String()
            tg_resp.data = json.dumps(
                {'request_id': tg_req_id, 'text': tg_text, 'partial': False},
                ensure_ascii=False,
            )
            self._tg_resp_pub.publish(tg_resp)
            self.get_logger().info(
                f'→ TG stop req={tg_req_id[:8]}'
                + (f': error="{tg_text[:60]}"' if tg_text else '')
            )

    def _publish_conversation_end(self, history: list, person_ctx: dict | None):
        """Publishes the transcript of a finished dialogue → memory_node saves the episode."""
        lines = []
        for m in history:
            role    = m.get('role', '')
            content = m.get('content', '')
            if not content or not isinstance(content, str):
                continue
            if role == 'user':
                lines.append(f'Пользователь: {content}')
            elif role == 'assistant':
                lines.append(f'Лёня: {content}')
        if not lines:
            return
        participants = []
        if person_ctx and person_ctx.get('name'):
            participants = [person_ctx['name']]
        payload = {
            'transcript':   '\n'.join(lines),
            'participants': participants,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._conv_end_pub.publish(msg)
        self.get_logger().info(
            f'conversation_end: {len(lines)} lines, participants: {participants}')

    def _tool_search_memory(self, args: dict) -> dict:
        """Search long-term semantic memory via memory_node."""
        query    = args.get('query', '')
        category = args.get('category', '') or ''
        req      = {'op': 'search_semantic', 'query': query,
                    'category': category, 'limit': 5}
        if not self._mem_client.wait_for_service(timeout_sec=2.0):
            return {'error': '/memory/query недоступен'}
        request = MemoryQuery.Request()
        request.request_json = json.dumps(req)
        future     = self._mem_client.call_async(request)
        done_event = threading.Event()
        future.add_done_callback(lambda _: done_event.set())
        if not done_event.wait(timeout=5.0):
            return {'error': 'таймаут memory service'}
        try:
            result = json.loads(future.result().response_json)
            facts  = result.get('facts', [])
            self.get_logger().info(f'search_memory: "{query}" → {len(facts)} facts')
            return result
        except Exception as e:
            return {'error': str(e)}

    def _tool_set_reminder(self, args: dict) -> dict:
        """Save a reminder for the user via /memory/query."""
        person_name = args.get('person_name', '')
        message     = args.get('message', '')
        date        = _resolve_reminder_date(args.get('date', '') or None)
        time_val    = _resolve_reminder_time(args.get('time', '') or None)
        person_id   = args.get('person_id')

        if not message:
            return {'error': 'message обязателен'}

        if not person_name or person_id is None:
            with self._lock:
                if self._person_context:
                    if not person_name:
                        person_name = self._person_context.get('name', '')
                    if person_id is None:
                        person_id = self._person_context.get('person_id')
        if person_id is None:
            return {'error': 'person_id не известен — пользователь не распознан'}

        req = {
            'op':           'add_reminder',
            'person_id':    int(person_id),
            'person_name':  person_name,
            'message':      message,
            'trigger_date': date,
            'trigger_time': time_val,
            'source':       'manual',
        }
        if not self._mem_client.wait_for_service(timeout_sec=2.0):
            return {'error': '/memory/query недоступен'}
        request = MemoryQuery.Request()
        request.request_json = json.dumps(req)
        future = self._mem_client.call_async(request)
        done_event = threading.Event()
        future.add_done_callback(lambda _: done_event.set())
        if not done_event.wait(timeout=5.0):
            return {'error': 'таймаут memory service'}
        try:
            result = json.loads(future.result().response_json)
            self.get_logger().info(
                f'set_reminder: {person_name} date={date} time={time_val or "default"} → {result}')
            return result
        except Exception as e:
            return {'error': str(e)}

    def _tool_confirm_reminder(self, args: dict) -> dict:
        """Delete shown reminders after the user confirms."""
        person_id = args.get('person_id')
        if person_id is None:
            with self._lock:
                if self._person_context:
                    person_id = self._person_context.get('person_id')
        if person_id is None:
            return {'error': 'person_id не известен'}

        req = {'op': 'confirm_reminders', 'person_id': int(person_id)}
        if not self._mem_client.wait_for_service(timeout_sec=2.0):
            return {'error': '/memory/query недоступен'}
        request = MemoryQuery.Request()
        request.request_json = json.dumps(req)
        future = self._mem_client.call_async(request)
        done_event = threading.Event()
        future.add_done_callback(lambda _: done_event.set())
        if not done_event.wait(timeout=5.0):
            return {'error': 'таймаут memory service'}
        try:
            result = json.loads(future.result().response_json)
            self.get_logger().info(
                f'confirm_reminder: person_id={person_id} → {result}')
            return result
        except Exception as e:
            return {'error': str(e)}

    # ── Chromecast broadcast ─────────────────────────────────────────────

    def _tool_broadcast_message(self, args: dict) -> dict:
        """
        1. POST /tts/to_file to the TTS server (192.168.10.118:8000):
           the server synthesizes a WAV, writes it to /etc/openhab/html/, returns a URL.
        2. Sets the LivingRoom_Chromecast_volume.
        3. Sends the URL to LivingRoom_Chromecast_uri → the Chromecast plays it.
        """
        text   = (args.get('text') or '').strip()
        volume = int(args.get('volume') or self._cast_volume)

        if not text:
            return {'success': False, 'error': 'text обязателен'}

        # ── The TTS server synthesizes and saves the file ─────────────────
        try:
            r = requests.post(
                self._cast_to_file_url,
                json={'text': text},
                timeout=(5.0, 30.0),
            )
            r.raise_for_status()
            data     = r.json()
            file_url = data['url']
            self.get_logger().info(
                f'Cast to_file: {data.get("file")} '
                f'({data.get("bytes", "?")} bytes) → {file_url}')
        except Exception as e:
            self.get_logger().error(f'Cast to_file error: {e}')
            return {'success': False, 'error': str(e)}

        # ── Setting the volume ──────────────────────────────────────────────
        try:
            requests.post(
                f'{self.openhab_url}/rest/items/LivingRoom_Chromecast_volume',
                data=str(volume),
                headers={'Content-Type': 'text/plain'},
                timeout=5.0,
            )
            time.sleep(0.3)
        except Exception as e:
            self.get_logger().warn(f'Cast: volume not set: {e}')

        # ── Sending the URI to the Chromecast ───────────────────────────────
        try:
            r = requests.post(
                f'{self.openhab_url}/rest/items/LivingRoom_Chromecast_uri',
                data=file_url,
                headers={'Content-Type': 'text/plain'},
                timeout=5.0,
            )
            if r.ok:
                self.get_logger().info(f'Cast: → Chromecast {file_url} (vol={volume})')
                return {'success': True, 'url': file_url, 'volume': volume}
            return {'success': False,
                    'error': f'OpenHAB HTTP {r.status_code}: {r.text}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ── Lifecycle callbacks ────────────────────────────────────────────────

    def _dp(self, name, default=None):
        """Safe declare_parameter: ignores a repeated declaration on re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        # llm_url — the primary backend, an OpenAI-compatible chat.completions endpoint
        # (currently vLLM). bearer_token is required for it. llm_fallback_url — optional
        # OpenAI-compatible backup; empty = no fallback. (The local Ollama qwen2.5:7b on the
        # NUC was dropped: ~375 s CPU prefill for the ~8.7k-token prompt + 4096 ctx truncation.)
        self._dp('llm_url',             'http://192.168.10.118:18020/v1/chat/completions')
        self._dp('llm_fallback_url',    '')
        self._dp('bearer_token',          '')  # for llm_url (vLLM)
        self._dp('bearer_token_fallback', '')  # for llm_fallback_url, if ever needed
        self._dp('model',               'qwen3.8-27b')
        self._dp('model_fallback',      'qwen2.5:7b')
        self._dp('temperature',         0.1)
        self._dp('max_tokens',          512)
        self._dp('connect_timeout_sec', 5.0)
        self._dp('timeout_sec',         120.0)
        self._dp('keep_history',        True)
        self._dp('history_max_turns',   8)
        self._dp('openhab_url',         'http://192.168.10.118:8080')
        self._dp('tts_server_url',      'http://192.168.10.118:8000')
        self._dp('tts_fallback_url',    '')
        self._dp('cast_volume',         80)
        self._dp('cast_to_file_url',    'http://192.168.10.118:8000/tts/to_file')
        # Tools the LLM may NOT use for Telegram requests (comma-separated): nobody
        # is necessarily next to the robot, so no motion and no destructive merges.
        self._dp('telegram_tool_denylist', 'robot_control,look_direction,merge_persons')
        # JSONL audit of every tool call ('' = off)
        self._dp('tool_audit_log',      '~/.ros/inmoov_tool_audit.jsonl')
        # Addressee gate: how long after the wake word speech counts as addressed
        # while no face is in view (see _addressing_reason). Covers the first
        # phrase (recording alone can take up to max_phrase_sec=20s) plus a quick follow-up.
        self._dp('wake_grace_sec',      30.0)

        # A separate vision model (look_and_describe/look_direction) — accepts
        # 2-4 base64 images in one request, an OpenAI-compatible /v1/chat/completions.
        # Added 2026-08-28: images used to go to self.llm_url (the text model),
        # but that one is limited to 1 image per prompt ("At most 1 image(s) may be provided").
        # vision_bearer_token is empty by default → falls back to the same bearer_token as
        # llm_url (server behind the same proxy/token), unless set separately.
        self._dp('vision_llm_url',      'http://192.168.10.118:18090/v1/chat/completions')
        self._dp('vision_model',        'qwen3vl')
        self._dp('vision_bearer_token', '')

        self.llm_url            = self.get_parameter('llm_url').value
        self.llm_fallback_url   = self.get_parameter('llm_fallback_url').value
        self.bearer_token          = self.get_parameter('bearer_token').value
        self.bearer_token_fallback = self.get_parameter('bearer_token_fallback').value
        self.vision_llm_url     = self.get_parameter('vision_llm_url').value
        self.vision_model       = self.get_parameter('vision_model').value
        self.vision_bearer_token = (self.get_parameter('vision_bearer_token').value
                                    or self.bearer_token)
        self.model_primary     = self.get_parameter('model').value
        self.model_fallback    = self.get_parameter('model_fallback').value
        self.temperature       = self.get_parameter('temperature').value
        self.max_tokens        = self.get_parameter('max_tokens').value
        self.connect_timeout   = self.get_parameter('connect_timeout_sec').value
        self.timeout_sec       = self.get_parameter('timeout_sec').value
        self.keep_history      = self.get_parameter('keep_history').value
        self.history_max       = self.get_parameter('history_max_turns').value
        self.openhab_url       = self.get_parameter('openhab_url').value
        self._tts_url          = self.get_parameter('tts_server_url').value
        self._tts_fallback_url = self.get_parameter('tts_fallback_url').value
        self._cast_volume      = self.get_parameter('cast_volume').value
        self._cast_to_file_url = self.get_parameter('cast_to_file_url').value
        self._tg_denied_tools  = parse_tool_list(
            self.get_parameter('telegram_tool_denylist').value)
        self._audit            = ToolAuditLog(self.get_parameter('tool_audit_log').value)
        self.wake_grace_sec    = self.get_parameter('wake_grace_sec').value
        self._active_url       = self.llm_url

        _latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(String, 'voice_command',   self.command_callback,         10)
        self.create_subscription(String, 'voice_command_other', self.other_command_callback, 10)
        self.create_subscription(String, '/speaker_evidence', self._speaker_evidence_cb, 10)
        self.create_subscription(String, '/telegram_ask',   self._telegram_ask_cb,         10)
        self.create_subscription(String, 'search_result',   self._search_result_callback,  10)
        self.create_subscription(String, 'person_context',  self._person_context_callback, 10)
        self.create_subscription(String, '/social_context', self._social_context_cb,       10)
        self.create_subscription(String, '/scene/objects',  self._scene_context_cb,        10)
        self.create_subscription(String, '/voice/speaker',  self._speaker_cb,              10)
        self.create_subscription(String, '/behavior/face_search_status',
                                 self._face_search_status_cb,                             10)
        self.create_subscription(String, 'openhab_schema',  self._oh_schema_callback,      10)
        self.create_subscription(String, 'openhab_items',   self._oh_items_callback,       10)
        self.create_subscription(CompressedImage, '/camera/eye_left/compressed',
                                 self._eye_camera_cb, _CAMERA_QOS)
        self.create_subscription(CompressedImage, '/camera/eye_right/compressed',
                                 self._eye_camera_right_cb, _CAMERA_QOS)
        self.create_subscription(Bool,   'wake_detected',   self._wake_cb,                 10)
        self.create_subscription(Bool,   '/introducing',    self._introducing_cb,          10)
        self.create_subscription(Bool,   '/go_idle',        self._go_idle_cb,              10)
        self.create_subscription(Bool,   '/robot_sleep',    self._robot_sleep_cb,          _latched)
        self.create_subscription(String, '/memory/context', self._memory_context_cb,       10)

        self.event_pub           = self.create_lifecycle_publisher(String, 'robot_events',       10)
        self._response_pub       = self.create_lifecycle_publisher(String, '/llm_response',      10)
        self._tg_resp_pub        = self.create_lifecycle_publisher(String, '/telegram_response', 10)
        self._tts_cancel_pub     = self.create_lifecycle_publisher(Bool,   '/tts_cancel_queue',  10)
        self._conv_end_pub       = self.create_lifecycle_publisher(String, '/conversation_end',  10)
        # Parsed voice direction hint ("I'm on the right" etc.) — on
        # every addressed utterance, consumer: behavior_manager_node
        # (face-search retry). See _parse_direction_hint/command_callback.
        self._direction_hint_pub = self.create_lifecycle_publisher(String, '/voice/direction_hint', 10)
        self._speech_addressed_pub = self.create_lifecycle_publisher(String, '/speech_addressed', 10)
        self._sv_confirm_pub = self.create_lifecycle_publisher(String, '/voice/sv_confirm', 10)

        self._mem_client         = self.create_client(MemoryQuery, '/memory/query')
        self._tts_direct_client  = ActionClient(self, Speak, 'speak')
        self.get_logger().info('LLM node configured')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.event_pub.on_activate(state)
        self._response_pub.on_activate(state)
        self._tg_resp_pub.on_activate(state)
        self._tts_cancel_pub.on_activate(state)
        self._conv_end_pub.on_activate(state)
        self._direction_hint_pub.on_activate(state)
        self._speech_addressed_pub.on_activate(state)
        self._sv_confirm_pub.on_activate(state)
        self._check_servers()
        # Background warmup: wait for the schema from openhab_bridge (10-15s), then
        # send a minimal request — loads the model and fills the KV cache.
        self._warmup_cancel.clear()
        self._queried_since_activate = False
        threading.Thread(target=self._warmup_llm, daemon=True).start()
        self.get_logger().info(f'LLM node ready. Model: {self.model}')
        return TransitionCallbackReturn.SUCCESS

    def _warmup_llm(self):
        """Warmup: load the model onto the GPU and fill the system-prompt KV cache."""
        # wait for openhab_bridge_node to publish the schema (every 10s)
        if self._warmup_cancel.wait(15.0):
            return   # deactivated
        if self._queried_since_activate:
            self.get_logger().info('LLM warmup skipped — a real request already warmed it')
            return
        _t = time.time()
        try:
            with self._lock:
                oh_schema = self._oh_schema
            sys_prompt = build_system_prompt(oh_schema, None, '')
            payload = {
                'model':    self.model,
                'messages': [
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user',   'content': 'Привет'},
                ],
                'tools':       TOOLS,
                'stream':      False,
                'temperature': 0.0,
                'max_tokens':  3,
                'chat_template_kwargs': {'enable_thinking': False},
            }
            for _ in self._stream_llm(payload, read_timeout=120.0):
                pass
            self.get_logger().info(f'LLM warmup finished in {time.time()-_t:.1f}s')
        except Exception as e:
            self.get_logger().warn(f'LLM warmup: error {e}')

    def on_deactivate(self, state):
        self._warmup_cancel.set()
        self._pending.clear()
        self.event_pub.on_deactivate(state)
        self._response_pub.on_deactivate(state)
        self._tg_resp_pub.on_deactivate(state)
        self._tts_cancel_pub.on_deactivate(state)
        self._conv_end_pub.on_deactivate(state)
        self._direction_hint_pub.on_deactivate(state)
        self._speech_addressed_pub.on_deactivate(state)
        self._sv_confirm_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def _tool_merge_persons(self, args: dict) -> dict:
        """Merges a duplicate person_id → target person_id.

        Steps:
        1. lookup_by_name for both names
        2. merge_persons with check_similarity=True
        3. If similarity_too_low — tell the user
        """
        dup_name    = (args.get('duplicate_name') or '').strip()
        target_name = (args.get('target_name') or '').strip()

        if not dup_name or not target_name:
            return {'success': False, 'error': 'duplicate_name и target_name обязательны'}
        if dup_name.lower() == target_name.lower():
            return {'success': False, 'error': 'Имена совпадают'}

        def _call(req):
            if not self._mem_client.wait_for_service(timeout_sec=2.0):
                return None
            request = MemoryQuery.Request()
            request.request_json = json.dumps(req, ensure_ascii=False)
            future = self._mem_client.call_async(request)
            ev = threading.Event()
            future.add_done_callback(lambda _: ev.set())
            if not ev.wait(timeout=6.0):
                return None
            try:
                return json.loads(future.result().response_json)
            except Exception:
                return None

        dup_r    = _call({'op': 'lookup_by_name', 'name': dup_name})
        target_r = _call({'op': 'lookup_by_name', 'name': target_name})

        if not dup_r or not dup_r.get('person_id'):
            return {'success': False, 'error': f'Не нашёл "{dup_name}" в памяти'}
        if not target_r or not target_r.get('person_id'):
            return {'success': False, 'error': f'Не нашёл "{target_name}" в памяти'}

        dup_id    = dup_r['person_id']
        target_id = target_r['person_id']

        result = _call({
            'op':               'merge_persons',
            'from_id':          dup_id,
            'to_id':            target_id,
            'check_similarity': True,
        })

        if not result:
            return {'success': False, 'error': 'memory service не ответил'}

        if result.get('merged'):
            self.get_logger().info(
                f'merge_persons: "{dup_name}"({dup_id}) → "{target_name}"({target_id})')
            return {
                'success':   True,
                'from_name': dup_name,
                'to_name':   target_name,
            }
        else:
            reason = result.get('reason', 'unknown')
            msg    = result.get('message', '')
            self.get_logger().warn(f'merge_persons rejected: {reason} — {msg}')
            return {'success': False, 'reason': reason, 'message': msg}

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS


def main():
    rclpy.init()
    node = LLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
