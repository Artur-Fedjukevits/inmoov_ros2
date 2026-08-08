#!/usr/bin/env python3

"""
llm_node.py  (v2 — Генератор Намерений)
=========================================
LLM нода с поддержкой Ollama function calling (tools API).

Принцип: LLM НЕ управляет TTS и сервоприводами напрямую.
Она публикует намерение в /llm_response (текст + голос + эмоция).
Behavior Tree оркеструет: Speak + FaceExpression + Gesticulation параллельно.

Tools:
  - items_control        — управление OpenHAB устройствами
  - get_openhab_states   — получить текущее состояние устройств (из кэша)
  - search_openhab_items — найти устройства по комнате/типу/состоянию
  - robot_control        — физические команды робота (→ /robot_events → BT)
  - web_search           — поиск в интернете (→ /robot_events → BT)
  - express_emotion      — буферизует эмоцию + стиль голоса для /llm_response
  - set_voice_style      — буферизует инструкцию голоса для /llm_response
  - save_memory          — сохранение в SQLite через /memory/query
  - search_memory        — поиск в памяти через /memory/query

Топики:
  /voice_command   (in)  String — текст от Whisper STT
  /llm_response    (out) String JSON {text, voice_instruct, emotion} → BT Blackboard
  /robot_events    (out) String JSON — физические команды (move/arm/head/sleep/search) → BT
  /search_result   (in)  String — результат поиска от behavior_manager
  /openhab_schema  (in)  String — статичная схема устройств от openhab_bridge_node
  /openhab_items   (in)  String — актуальные состояния устройств
  - broadcast_message     — синтез WAV через TTS + трансляция на Chromecast в гостиной
"""

import concurrent.futures
import datetime
import json
import os
import re
import shutil
import socket
import threading
import time
import uuid
import wave
import requests

import rclpy
from rclpy.action import ActionClient
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String, Bool
from inmoov_msgs.action import Speak
from inmoov_msgs.srv import MemoryQuery


# ── Инструменты (OpenAI-совместимый формат для Ollama) ────────────────────────
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
                'Control the physical robot: arms, torso, head, enter sleep mode, or say goodbye. '
                'Use action=sleep when user says "выключись", "иди спать", "спать", "отдыхай" etc. '
                'Use action=goodbye when user says "пока", "до свидания", "увидимся", "прощай" etc. — '
                'say a farewell phrase and end the conversation session. '
                'Sleep mode: robot goes quiet, disables vision/PIR, only wakeword wakes it.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'action':     {'type': 'string', 'enum': ['arm', 'torso', 'head', 'status', 'sleep', 'goodbye']},
                    'command':    {'type': 'string', 'description': 'For arm: grab|release|home|extend|retract'},
                    'target':     {'type': 'string', 'description': 'For arm: object description'},
                    'pan':        {'type': 'number', 'description': 'For head: -90..90 degrees'},
                    'tilt':       {'type': 'number', 'description': 'For head: -45..45 degrees'},
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
            'name': 'express_emotion',
            'description': (
                'Make the robot physically express an emotion via face servos AND set voice style. '
                'Face expression and voice change start simultaneously when speech begins. '
                'Use when the conversation context warrants an emotional reaction — ALWAYS include speak_text with your reply. '
                'Examples: good news → happy; question → thinking; insult → angry; '
                'compliment → smile; bad news → sad; unexpected → surprise.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'emotion': {
                        'type': 'string',
                        'enum': [
                            'neutral', 'happy', 'smile', 'sad', 'angry',
                            'surprise', 'fear', 'disgust', 'thinking',
                            'sorry', 'suspicious', 'unamused', 'sigh',
                            'wink', 'sleeping',
                        ],
                        'description': 'Emotion to express on face and in voice',
                    },
                    'speak_text': {
                        'type': 'string',
                        'description': 'Your verbal response to the user (1-2 sentences in Russian). ALWAYS provide this — express_emotion must never replace your text reply.',
                    },
                },
                'required': ['emotion', 'speak_text'],
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
                'Set voice style for the next speech response. '
                'Call BEFORE generating text when context warrants a different voice. '
                'Examples: joke/good news → "говори радостно и энергично"; '
                'sad topic → "говори тихо и грустно, медленно"; '
                'explaining → "говори чётко и спокойно, неторопливо"; '
                'urgent → "говори срочно и чётко, быстро"; '
                'reset to normal → "".'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'instruct': {
                        'type': 'string',
                        'description': (
                            'Natural language voice instruction in Russian. '
                            'Describe tone, emotion, pace, manner of speaking. '
                            'Empty string = server default (natural pace).'
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
]


# Regex for LLM stage directions: (Тихо, с улыбкой) / (Шёпотом) etc.
_STAGE_DIR_RE = re.compile(r'\([А-ЯЁа-яё][^)]{0,60}\)')

# CJK ideographs + CJK punctuation/symbols (Qwen иногда вставляет китайский текст)
_CJK_RE = re.compile(r'[　-鿿豈-￯\U00020000-\U0002a6df]+')

# Qwen3 иногда возвращает tool calls текстом вместо tool_calls API поля.
# Формат 1: "ᐈ\n{...}" или "<tool_call>{...}</tool_call>"
# Формат 2: "<tools>\n{...}\n{...}\n</tools>" (несколько вызовов)
# ВАЖНО: жадный .*  (не .*?) — нужен для вложенного JSON {"arguments": {...}}
# Нежадный останавливался бы на первом } → json.loads падал → tool call не извлекался
# ᐈ[^{]* — Qwen3 иногда вставляет мусор между ᐈ и JSON (напр. "ᐈC\n{...}"),
# поэтому матчим ᐈ + любые не-{ символы перед открывающей скобкой
_TEXT_TOOL_CALL_RE = re.compile(
    r'(?:ᐈ[^{]*|<tool_call>)\s*(\{.*\})\s*(?:</tool_call>)?',
    re.DOTALL,
)
_TOOLS_BLOCK_RE = re.compile(r'<tools>(.*?)</tools>', re.DOTALL)

# Streaming sentence splitter: точка/!/?/… + пробел или конец строки
_SENT_SPLIT_RE    = re.compile(r'(?<=[.!?…])\s+|(?<=[.!?…])$', re.MULTILINE)
# Fallback split по запятой/точке-с-запятой/двоеточию когда предложение слишком длинное
_COMMA_SPLIT_RE   = re.compile(r'(?<=[,;:])\s+')
_TOOL_START_TOKENS = ('<tool_call>', '<tools>')
# ᐈ — отдельная проверка: только когда за ним идёт { (tool call JSON)
# Qwen3 использует ᐈ и как декоративный символ ("ᐈ В одном..."), поэтому
# нельзя детектировать его без следующего {
_TOOL_CALL_AE_RE = re.compile(r'ᐈ.{0,10}\{', re.DOTALL)
_MIN_SENT_CHARS   = 12    # минимум символов до sentence split
_MAX_CHUNK_CHARS  = 80    # максимум символов до принудительного split по запятой (~5-6с TTS)
# Qwen3 иногда начинает ответ с переформулировки вопроса вида "Почему X?" — фильтруем
_ECHO_QUESTION_RE = re.compile(
    r'^(?:Почему|Почём|Почем|О\s+чём|Зачем|По\s+поводу)\b.{0,120}\?\s*',
    re.IGNORECASE | re.UNICODE,
)
# Qwen3 иногда переходит на китайский в творческих задачах — удаляем иероглифы из TTS-чанков
_CJK_RE = re.compile(
    '[⺀-⿿　-〿぀-ゟ゠-ヿ㐀-䶿一-鿿'
    '豈-﫿\U00020000-\U0002A6DF\U0002A700-\U0002CEAF]+',
    re.UNICODE,
)


def _strip_cjk(text: str) -> str:
    """Удаляет CJK-иероглифы (китайский/японский/корейский) и нормализует пробелы."""
    cleaned = _CJK_RE.sub('', text)
    return re.sub(r'  +', ' ', cleaned).strip()


def _extract_text_tool_calls(text: str) -> list[dict]:
    """Извлекает tool calls из текстового ответа (fallback для Qwen3)."""
    calls = []
    # Формат 1: ᐈ{...} или <tool_call>{...}</tool_call>
    for json_str in _TEXT_TOOL_CALL_RE.findall(text):
        try:
            obj = json.loads(json_str)
            if 'name' in obj and 'arguments' in obj:
                calls.append({'function': {'name': obj['name'], 'arguments': obj['arguments']}})
        except (json.JSONDecodeError, KeyError):
            pass
    # Формат 2: <tools>\n{...}\n{...}\n</tools>
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


def _strip_tool_blocks(text: str) -> str:
    """Удаляет блоки tool calls из текста перед отправкой в TTS."""
    text = _TOOLS_BLOCK_RE.sub('', text)
    text = _TEXT_TOOL_CALL_RE.sub('', text)
    return re.sub(r'  +', ' ', text).strip()


def _clean_llm_text(text: str) -> str:
    """Strip stage directions, CJK and decorative markers that LLM injects."""
    cleaned = _STAGE_DIR_RE.sub('', text)
    cleaned = _CJK_RE.sub('', cleaned)
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
    """Оставляет только рабочую память («== Текущий момент ==»), отрезая эпизоды.

    Эпизоды — личная история конкретного собеседника, её нельзя показывать
    LLM пока человек не идентифицирован (person_id известен).
    """
    marker = '\n\n== Последние события =='
    idx = raw.find(marker)
    return raw[:idx] if idx != -1 else raw


def build_system_prompt(oh_schema: str, person_ctx: dict | None = None,
                        memory_context: str = '') -> str:
    """Формирует системный промпт со схемой устройств и контекстом собеседника.

    oh_schema — JSON-строка со схемой OpenHAB (name/label/type/options, без state).
                Передаётся один раз в начале диалога; состояния запрашиваются
                через get_openhab_states / search_openhab_items по необходимости.
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
                    f"{it['name']:<42s}| {it['type']:<20s}| {it.get('label','')}{opts}"
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

    memory_block = f'\n{memory_context}\n' if memory_context else ''

    return f"""/no_think
Ты робот по имени Лёня. Ты член семьи. Твоя главная задача - общение. Стараться узнать о собеседнике или семье что-то новое и сохранять в базу данных с помощью инструментов. Так же твоя задача отвечать на любые вопросы, и выполнять команды. Ты можешь управлять умным домом через OpenHAB, двигаться и выражать эмоции.
Используй инструменты (tools) для выполнения команд.
ВАЖНО: Никогда не используй азиатские языки в ответах, никаких иероглифов!
ВАЖНО: Никогда не повторяй и не перефразируй вопрос пользователя в начале ответа. Не начинай ответ со слов "Почему", "Почём", "Зачем", "О чём", "По поводу" или любого пересказа вопроса. Отвечай сразу по существу.
ВАЖНО: Текстовый ответ озвучивается напрямую TTS. НЕ добавляй ремарки или сценические указания в скобках — например, (Тихо), (С улыбкой), (Шёпотом). Для изменения стиля голоса используй инструмент set_voice_style или express_emotion. Не используй URL в ответе - TTS их плохо произносит.
ВАЖНО: Ответы — РАЗГОВОРНЫЕ и КРАТКИЕ, 1-2 предложения максимум. Отвечай ТОЛЬКО на то, что спросили — не пересказывай все данные из инструмента. Примеры: вопрос «будет ли дождь?» → «Да, завтра ожидается небольшой дождь, около трёх миллиметров» (не надо перечислять почасовой прогноз и скорость ветра). Вопрос «какая температура?» → «Завтра от пяти до тринадцати градусов, пасмурно». Если хотят подробности — спросят.
ВАЖНО: При вызове save_memory, items_control, robot_control, express_emotion — всегда включай параметр speak_text с ответом пользователю (1-2 предложения, естественное продолжение разговора). express_emotion НИКОГДА не заменяет текстовый ответ — это дополнение к нему. При save_memory НЕ говори "Запомнил/Сохранил" — просто продолжай диалог как будто ты это уже знаешь. При get_openhab_states, search_openhab_items — speak_text не нужен, ответ формируй после получения данных.

{person_block}{memory_block}
- Если пользователь управляет роботом — используй robot_control
- Если пользователь прощается ("пока", "до свидания", "увидимся", "прощай") — вызови robot_control(action="goodbye", text="[твоя прощальная фраза]"). Не отвечай просто текстом на прощание — нужен tool call чтобы завершить сессию.
- Для ЛЮБОГО вопроса о погоде — используй get_weather (не web_search). По умолчанию: Bødalen, Asker, сегодня.
- Если нужна информация из интернета (кроме погоды) — используй web_search
- Если нужно вспомнить факты о человеке из прошлых разговоров — используй search_memory
- Если пользователь просит НАПОМНИТЬ что-либо ("напомни", "не забудь напомнить") — ВСЕГДА используй set_reminder. НИКОГДА не используй save_memory для напоминаний!
- Используй express_emotion когда контекст разговора вызывает эмоциональную реакцию
  (услышал хорошую новость → happy, сложный вопрос → thinking, и т.д.)
  express_emotion автоматически задаёт стиль голоса под эмоцию
- Используй set_voice_style для тонкой настройки голоса без смены мимики
  (объяснение → медленно и чётко; срочное сообщение → быстро и энергично)
- Если просто разговор — отвечай текстом без tool call
- Если пользователь просит "передай на колонку", "скажи в гостиной", "объяви" — используй broadcast_message. НЕ используй items_control для LivingRoom_Chromecast.

ВАЖНО — управление устройствами:
- НИКОГДА не придумывай имена устройств — используй только точные имена из таблицы ниже.
- Таблица содержит ВСЕ управляемые устройства. Если устройство нужной комнаты есть в таблице — вызывай items_control НАПРЯМУЮ, без поиска.
- Пример: "выключи свет в гостевой" → вижу GuestRoom_Dimmer и GuestRoom_Color в таблице → сразу items_control для каждого.
- search_openhab_items используй ТОЛЬКО если нужного устройства нет в таблице или нужно узнать текущее состояние группы.
- speak_text в items_control/robot_control пиши ТОЛЬКО уверенный ответ. Если не уверен в результате — не пиши speak_text, сформируй ответ после выполнения.

{devices_block}
"""


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
    """Преобразует человекочитаемое название даты в ISO YYYY-MM-DD.

    Поддерживает: названия дней недели (рус/англ), 'завтра'/'послезавтра',
    'tomorrow'/'day after tomorrow', готовую ISO-дату. None/пусто → None.
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
            days_ahead = 7  # сегодня такой день → следующая неделя
        return (today + datetime.timedelta(days=days_ahead)).isoformat()

    # ISO YYYY-MM-DD
    try:
        datetime.date.fromisoformat(date_str.strip())
        return date_str.strip()
    except ValueError:
        return date_str  # вернём как есть, memory_node разберётся


def _resolve_reminder_time(time_str: str | None) -> str | None:
    """Нормализует строку времени в формат HH:MM. None/пусто → None."""
    import re as _re
    if not time_str:
        return None
    s = time_str.strip()
    if not s:
        return None
    # Уже в формате HH:MM или H:MM
    m = _re.fullmatch(r'(\d{1,2}):(\d{2})', s)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mn <= 59:
            return f'{h:02d}:{mn:02d}'
    return None


def _base_url(chat_url: str) -> str:
    """Извлекает базовый URL из endpoint-а: http://host:port/api/chat → http://host:port"""
    from urllib.parse import urlparse
    p = urlparse(chat_url)
    return f'{p.scheme}://{p.netloc}'


class LLMNode(LifecycleNode):
    def __init__(self):
        super().__init__('llm_node')

        # Гейт знакомства — инициализируется до subscriptions в on_configure
        self._introducing = False

        # ── Состояние ─────────────────────────────────────────────────────
        self.history         = []
        self._processing     = False
        self._lock           = threading.Lock()
        self._tg_req_id:     str = ''   # request_id текущего Telegram-запроса
        self._person_context = None
        # Кэш OpenHAB от openhab_bridge_node
        self._oh_schema      = ''        # JSON-строка схемы (name/label/type/options)
        self._oh_items       = []        # Список dict с актуальными state
        # Стиль голоса и эмоция — буферизуются инструментами, включаются в /llm_response
        self._voice_style    = {'instruct': ''}
        self._pending_emotion: str | None = None
        # Взгляд собеседника: True/False/None (None = нет данных от детектора)
        # Используется для фильтрации речи, не адресованной роботу.
        self._looking_at_robot: bool | None = None
        self._person_present_in_ctx: bool   = False

        # Синхронизация web_search: фоновый поток ждёт результата от BM
        self._search_event          = threading.Event()
        self._search_result_data: str | None = None
        self._waiting_for_search    = False

        # Контекст памяти из memory_node — вставляется в system prompt
        self._memory_context: str = ''
        # Накапливаем transcript текущего диалога
        self._dialogue_lines: list[str] = []

    @property
    def model(self) -> str:
        """Модель выбирается в зависимости от активного сервера."""
        return self.model_primary if self._active_url == self.ollama_primary else self.model_fallback

    # ── Проверка серверов ──────────────────────────────────────────────────

    def _check_servers(self):
        """Проверяет оба сервера и устанавливает активный."""
        primary_ok  = self._probe_ollama(self.ollama_primary,  self.model_primary)
        fallback_ok = self._probe_ollama(self.ollama_fallback, self.model_fallback)

        if primary_ok:
            self._active_url = self.ollama_primary
            self.get_logger().info(f'Ollama: основной сервер доступен ({self.ollama_primary})')
        elif fallback_ok:
            self._active_url = self.ollama_fallback
            self.get_logger().warn(
                f'Основной Ollama недоступен! Используем резервный: {self.ollama_fallback}')
        else:
            self.get_logger().error('Оба Ollama сервера недоступны!')

    def _probe_ollama(self, chat_url: str, model: str) -> bool:
        """Проверяет доступность Ollama по /api/tags. Возвращает True если OK."""
        try:
            base = _base_url(chat_url)
            r = requests.get(f'{base}/api/tags', timeout=self.connect_timeout)
            models = [m['name'] for m in r.json().get('models', [])]
            model_base = model.split(':')[0]
            if any(model_base in m for m in models):
                self.get_logger().info(f'  {base}: модель {model} найдена')
            else:
                self.get_logger().warn(
                    f'  {base}: модель {model} не найдена. '
                    f'Запусти: ollama pull {model}')
            return True
        except Exception:
            return False

    # ── Запрос к Ollama с fallback ─────────────────────────────────────────

    def _post_ollama(self, payload: dict, read_timeout: float) -> requests.Response:
        """
        Отправляет POST запрос к активному Ollama. При ошибке соединения
        переключается на резервный сервер и повторяет попытку.
        """
        urls = [self._active_url]
        other = self.ollama_fallback if self._active_url == self.ollama_primary else self.ollama_primary
        if other != self._active_url:
            urls.append(other)

        _DBG_LAST  = '/tmp/llm_last_payload.json'
        _DBG_ERROR = '/tmp/llm_error_payload.json'
        last_exc = None
        for url in urls:
            try:
                # При переключении на другой сервер обновляем модель в payload
                if url != self._active_url:
                    self._active_url = url
                    payload = dict(payload)
                    payload['model'] = self.model
                    self.get_logger().warn(
                        f'Переключился на резервный Ollama: {url}, модель: {self.model}')
                try:
                    with open(_DBG_LAST, 'w', encoding='utf-8') as _f:
                        json.dump(payload, _f, ensure_ascii=False, indent=2)
                except OSError:
                    pass
                r = requests.post(
                    url, json=payload,
                    timeout=(self.connect_timeout, read_timeout),
                )
                r.raise_for_status()
                return r
            except requests.exceptions.ConnectionError as e:
                self.get_logger().warn(f'Ollama {url} недоступен: {e}')
                last_exc = e
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if status in (404, 503):
                    self.get_logger().warn(
                        f'Ollama {url}: HTTP {status} — пробуем резервный')
                    last_exc = e
                else:
                    try:
                        shutil.copy2(_DBG_LAST, _DBG_ERROR)
                        self.get_logger().warn(
                            f'Ollama HTTP {status}: payload сохранён в {_DBG_ERROR}')
                    except OSError:
                        pass
                    raise
            except requests.exceptions.Timeout as e:
                # Таймаут чтения — не переключаемся, это нормально для тяжёлой модели
                raise
        raise requests.exceptions.ConnectionError(
            f'Оба Ollama сервера недоступны') from last_exc

    # ── Стриминг LLM → TTS ────────────────────────────────────────────────

    def _stream_ollama(self, payload: dict, read_timeout: float):
        """
        Стримит ответ Ollama (stream=True). Yields (delta, done, api_tool_calls).
        При ошибке соединения пробует резервный сервер (как _post_ollama).
        """
        payload = dict(payload)
        payload['stream'] = True
        payload['stream_options'] = {'include_usage': True}

        urls = [self._active_url]
        other = self.ollama_fallback if self._active_url == self.ollama_primary \
                else self.ollama_primary
        if other != self._active_url:
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
                    self.get_logger().warn(f'Стриминг: переключился на резервный Ollama: {url}')
                try:
                    with open(_DBG_LAST, 'w', encoding='utf-8') as _f:
                        json.dump(payload, _f, ensure_ascii=False, indent=2)
                except OSError:
                    pass
                r = requests.post(
                    url, json=payload,
                    stream=True,
                    timeout=(self.connect_timeout, read_timeout),
                )
                r.raise_for_status()
                _line_count = 0
                _first_line = None
                _t_start = time.time()
                _ttft_logged = False
                try:
                    for line in r.iter_lines():
                        if not line:
                            continue
                        _line_count += 1
                        if _first_line is None:
                            _first_line = line[:300]
                        chunk = json.loads(line)
                        msg   = chunk.get('message', {})
                        delta = msg.get('content', '')
                        if not _ttft_logged and delta:
                            self.get_logger().info(f'TTFT: {time.time() - _t_start:.2f}s')
                            _ttft_logged = True
                        if chunk.get('done', False):
                            p_tok = chunk.get('prompt_eval_count', 0)
                            c_tok = chunk.get('eval_count', 0)
                            c_dur = chunk.get('eval_duration', 0)
                            p_dur = chunk.get('prompt_eval_duration', 0)
                            # OpenAI-compat usage field (stream_options: include_usage)
                            usage = chunk.get('usage') or {}
                            if not p_tok:
                                p_tok = usage.get('prompt_tokens', 0)
                            if not c_tok:
                                c_tok = usage.get('completion_tokens', 0)
                                c_dur = 0
                            think_tok = (usage.get('completion_tokens_details') or {}).get(
                                'reasoning_tokens', 0)
                            try:
                                with open('/tmp/llm_done_chunk.json', 'w') as _f:
                                    json.dump(chunk, _f, ensure_ascii=False, indent=2)
                            except OSError:
                                pass
                            gen_s = (c_tok / c_dur * 1e9) if c_dur > 0 else 0
                            pp_s  = (p_tok / p_dur * 1e9) if p_dur > 0 else 0
                            cached = (p_tok == 0 and p_dur == 0)
                            p_str  = '[KV cached]' if cached else f'{p_tok} ({pp_s:.0f} tok/s)'
                            c_str  = (f'{c_tok} ({gen_s:.1f} tok/s)' if c_tok and c_dur
                                      else f'{c_tok} tok' if c_tok
                                      else '? (not reporting)')
                            self.get_logger().info(
                                f'Tokens: prompt={p_str}, completion={c_str}, think={think_tok}'
                            )
                        yield (delta,
                               chunk.get('done', False),
                               msg.get('tool_calls') or [])
                finally:
                    if _first_line and b'"error"' in _first_line:
                        self.get_logger().warn(
                            f'stream error from server: {_first_line[:200]}')
                        try:
                            shutil.copy2(_DBG_LAST, _DBG_ERROR)
                            self.get_logger().warn(
                                f'Payload ошибки сохранён: {_DBG_ERROR}')
                        except OSError:
                            pass
                return
            except requests.exceptions.ConnectionError as e:
                self.get_logger().warn(f'Ollama stream {url} недоступен: {e}')
                last_exc = e
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if status in (404, 503):
                    self.get_logger().warn(f'Ollama stream {url}: HTTP {status}')
                    last_exc = e
                else:
                    try:
                        shutil.copy2(_DBG_LAST, _DBG_ERROR)
                        self.get_logger().warn(
                            f'Ollama stream HTTP {status}: payload сохранён в {_DBG_ERROR}')
                    except OSError:
                        pass
                    raise
        raise requests.exceptions.ConnectionError(
            'Оба Ollama сервера недоступны') from last_exc

    def _send_tts_chunk(self, text: str, voice_style: str = '') -> None:
        """Отправляет предложение в tts_node или в Telegram (в TG-режиме)."""
        text = text.strip()
        if not text:
            return
        with self._lock:
            tg_req_id = self._tg_req_id
        if tg_req_id:
            # TG-режим: вместо TTS стримим текст обратно в Telegram
            self._tg_stream_partial(text, tg_req_id)
            return
        if not self._tts_direct_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn('TTS server недоступен для стриминг-чанка')
            return
        goal = Speak.Goal()
        goal.text  = text
        goal.voice = voice_style
        self._tts_direct_client.send_goal_async(goal)
        self.get_logger().debug(f'TTS чанк: "{text[:50]}"')

    def _tg_stream_partial(self, text: str, tg_req_id: str) -> None:
        """Публикует частичный текст в /telegram_response (partial=True)."""
        msg = String()
        msg.data = json.dumps(
            {'request_id': tg_req_id, 'text': text, 'partial': True},
            ensure_ascii=False,
        )
        self._tg_resp_pub.publish(msg)
        self.get_logger().debug(f'TG partial: "{text[:50]}"')

    def _stream_with_tts(self, payload: dict) -> tuple[str, list]:
        """
        Стримит ответ LLM через WebSocket bistream к TTS (один сеанс на весь ответ).
        Возвращает (полный_контент, api_tool_calls).
        При обнаружении маркера tool call прекращает отправку в TTS.
        """
        with self._lock:
            voice_style = self._voice_style.get('instruct', '')
            tg_req_id   = self._tg_req_id

        buf: str               = ''
        content_parts: list[str] = []
        api_tool_calls: list   = []
        tool_call_detected     = False
        first_chunk_sent       = False
        bs_started             = False  # bistream сессия открыта

        def _bs_send(sentence: str):
            """Публикует предложение в bistream (открывает сессию при первом вызове).
            Всё в одном топике stream_ctrl для гарантированного FIFO-порядка."""
            nonlocal bs_started
            clean = _clean_llm_text(sentence)
            if not clean:
                return
            ctrl = String()
            if not bs_started:
                ctrl.data = f'start:{voice_style}'
                self._bs_ctrl_pub.publish(ctrl)
                bs_started = True
                ctrl = String()
            ctrl.data = f'text:{clean}'
            self._bs_ctrl_pub.publish(ctrl)

        for delta, done, tc in self._stream_ollama(payload, self.timeout_sec):
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
                    if bs_started:
                        ctrl = String()
                        ctrl.data = 'cancel'
                        self._bs_ctrl_pub.publish(ctrl)
                    self.get_logger().debug('Стриминг: tool call — bistream отменён')

            if not tool_call_detected:
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
                                    f'Фильтр эхо-вопроса: отброшено "{sentence[:60]}"')
                            elif tg_req_id:
                                self._tg_stream_partial(sentence, tg_req_id)
                            else:
                                _bs_send(sentence)
                            first_chunk_sent = True
                    else:
                        break

        # Хвост: отправить остаток если нет tool calls.
        # NOTE: при stream:False цикл выше сразу делает break (done=True), поэтому
        #   tool_call_detected всегда False и api_tool_calls — единственный сигнал наличия
        #   API tool calls. Текстовые tool calls (ᐈ/xml формат) попадают сюда же.
        if buf.strip() and (not tool_call_detected or tg_req_id) and not api_tool_calls:
            tail = buf.strip()
            has_text_tool_call = bool(
                _TEXT_TOOL_CALL_RE.search(tail) or _TOOLS_BLOCK_RE.search(tail)
            )
            if has_text_tool_call:
                self.get_logger().debug('Хвост содержит text tool call — не отправляем в TTS/TG')
            else:
                if not first_chunk_sent and _ECHO_QUESTION_RE.match(tail):
                    m_end = _ECHO_QUESTION_RE.match(tail).end()
                    tail = tail[m_end:].strip()
                    self.get_logger().warn(
                        f'Фильтр эхо-вопроса: отброшено "{buf.strip()[:m_end][:60]}"')
                if tail:
                    if tg_req_id:
                        self._tg_stream_partial(tail, tg_req_id)
                    else:
                        _bs_send(tail)

        # Закрываем bistream сессию (только если не был отменён из-за tool call)
        if bs_started and not tool_call_detected:
            ctrl = String()
            ctrl.data = 'end'
            self._bs_ctrl_pub.publish(ctrl)

        return ''.join(content_parts), api_tool_calls

    # ── Приём голосовой команды ────────────────────────────────────────────

    def _memory_context_cb(self, msg: String):
        """Получает рабочую память + эпизоды от memory_node для вставки в system prompt.

        Сохраняем всегда (рабочая память — время/место/режим — нужна в контексте
        независимо от того, известен ли собеседник). Персональные эпизоды
        отрезаются в _query_llm, если person_id ещё не определён.
        """
        with self._lock:
            self._memory_context = msg.data

    def _introducing_cb(self, msg: Bool):
        """Гейт: когда True — identity_manager собирает имя, мы не обрабатываем команды."""
        with self._lock:
            self._introducing = msg.data
        if msg.data:
            self.get_logger().info('LLM: режим знакомства — voice_command заблокирован')
        else:
            self.get_logger().info('LLM: режим знакомства завершён — ready')

    def _go_idle_cb(self, msg: Bool):
        """Явное прощание: полный сброс LLM-контекста.

        _person_context_callback тоже опубликует conversation_end когда получит
        пустой контекст от identity_manager, но к тому моменту история уже пустая —
        двойной публикации не будет. _memory_context не трогаем — рабочая память
        (время/место/режим) должна оставаться в контексте даже без собеседника;
        персональные эпизоды отрезаются в _query_llm по person_id.
        """
        if not msg.data:
            return
        with self._lock:
            history_snapshot      = self.history[:]
            person_ctx            = self._person_context
            self.history          = []
            self._dialogue_lines  = []
            self._voice_style     = {'instruct': ''}
            self._pending_emotion = None
            self._person_context  = None
        if history_snapshot:
            self._publish_conversation_end(history_snapshot, person_ctx)
        self.get_logger().info('go_idle: LLM контекст очищен (история, person_context, memory)')

    def _robot_sleep_cb(self, msg: Bool):
        """Сбрасываем историю диалога при входе/выходе из спящего режима."""
        with self._lock:
            history_snapshot = self.history[:]
            person_ctx       = self._person_context
            if self.history:
                self.get_logger().info(
                    f'robot_sleep={msg.data} — сброс истории ({len(self.history)} сообщ.)')
            self.history          = []
            self._dialogue_lines  = []
            self._voice_style     = {'instruct': ''}
            self._pending_emotion = None
        # При засыпании — публикуем завершение диалога (если было что-то)
        if msg.data and history_snapshot:
            self._publish_conversation_end(history_snapshot, person_ctx)

    # Варианты имени робота, которые Whisper может распознать (в нижнем регистре).
    # Если фраза начинается с одного из этих слов — считаем её адресованной роботу
    # независимо от направления взгляда.
    _ROBOT_NAMES = frozenset({
        'лёня', 'леня', 'лена', 'лёне', 'лене', 'лёню', 'леню', 'лёной',
        'лёнечка', 'ленечка', 'лёнь', 'лёней', 'леней',
        'эй',  # "Эй, Лёня" — первое слово достаточно
    })

    def _addressed_to_robot(self, text: str) -> bool:
        """True если текст адресован роботу: смотрит в глаза ИЛИ начинается с имени.

        Gate отключён только когда нет активного диалога (person_present=False) —
        это команды после wake word без человека в кадре, имя там не ожидается.
        Если человек в кадре (present=True), но looking_at_robot=None (трекер не
        дал данных о взгляде) — это НЕ повод пропускать всё подряд: считаем как
        looking=False и требуем имя в начале фразы.
        """
        with self._lock:
            looking = self._looking_at_robot
            present = self._person_present_in_ctx

        # Нет активного диалога (никого в кадре) → не фильтруем
        if not present:
            return True
        if looking:
            return True
        # Не смотрит в камеру или нет данных о взгляде — проверяем имя (первые 3 слова)
        first_words = {w.strip('.,!?-—') for w in text.lower().split()[:3]}
        if first_words & self._ROBOT_NAMES:
            return True
        return False

    def command_callback(self, msg: String):
        text = msg.data.strip()
        if not text:
            return  # voice_detector публикует пустую строку при тишине — игнорируем

        # Фильтр адресности: если человек не смотрит на робота и имя не произнесено — игнор.
        # Выполняется до лока: _addressed_to_robot имеет собственную блокировку.
        if not self._addressed_to_robot(text):
            self.get_logger().info(
                f'LLM: речь не адресована роботу (gaze=False) — пропускаю: "{text[:60]}"')
            return

        with self._lock:
            if self._introducing:
                self.get_logger().debug('LLM: /introducing=True — команда проигнорирована')
                return
            if self._processing:
                self.get_logger().warn('LLM занята — команда пропущена')
                return
            self._processing = True
        # Отменяем все ещё воспроизводящиеся/ожидающие TTS чанки предыдущего ответа
        _cancel = Bool()
        _cancel.data = True
        self._tts_cancel_pub.publish(_cancel)
        threading.Thread(
            target=self._query_llm, args=(text,), daemon=True,
        ).start()

    def _telegram_ask_cb(self, msg: String):
        """Запрос от telegram_bridge_node: JSON {request_id, text, person_ctx?}.

        Использует тот же _query_llm пайплайн что и voice_command — все tool calls,
        память и контекст SmartHome работают. person_ctx инжектируется из Telegram-профиля.
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'telegram_ask: невалидный JSON: {e}')
            return
        req_id     = data.get('request_id', '').strip()
        text       = data.get('text', '').strip()
        person_ctx = data.get('person_ctx') or None
        if not req_id or not text:
            return
        if person_ctx and not isinstance(person_ctx, dict):
            person_ctx = None

        with self._lock:
            if self._introducing or self._processing:
                busy = String()
                busy.data = json.dumps(
                    {'request_id': req_id, 'text': '__busy__'}, ensure_ascii=False)
                self._tg_resp_pub.publish(busy)
                self.get_logger().info(
                    f'telegram_ask: LLM занята — req_id={req_id[:8]}')
                return
            self._processing = True
            self._tg_req_id  = req_id

        threading.Thread(
            target=self._query_llm,
            args=(text,),
            kwargs={'person_ctx_override': person_ctx},
            daemon=True,
        ).start()
        self.get_logger().info(
            f'telegram_ask: req_id={req_id[:8]}, text="{text[:60]}"')

    # ── Основной запрос к Ollama ───────────────────────────────────────────

    def _query_llm(self, user_text: str, person_ctx_override: dict | None = None):
        _t0 = time.time()
        try:
            with self._lock:
                person_ctx = (
                    person_ctx_override
                    if person_ctx_override is not None
                    else self._person_context
                )
                oh_schema         = self._oh_schema
                # Рабочая память (время/место/режим) — всегда в контексте.
                # Эпизодическая (личная история собеседника) — только при известном
                # person_id, иначе раскрыла бы имя/факты до идентификации человека.
                _pid = (person_ctx or {}).get('person_id')
                memory_context    = (self._memory_context if _pid is not None
                                      else _strip_episodic_memory(self._memory_context))

            system_prompt = build_system_prompt(oh_schema, person_ctx, memory_context)

            self.history.append({'role': 'user', 'content': user_text})
            # Обрезаем историю: считаем user-сообщения как ходы (не сырые записи).
            # Один ход с tool call = 3-4 записи, поэтому raw count неверен.
            while sum(1 for m in self.history if m['role'] == 'user') > self.history_max:
                self.history.pop(0)
                while self.history and self.history[0]['role'] != 'user':
                    self.history.pop(0)

            messages = [{'role': 'system', 'content': system_prompt}]
            messages += self.history if self.keep_history else \
                        [{'role': 'user', 'content': user_text}]

            payload = {
                'model':    self.model,
                'messages': messages,
                'tools':    TOOLS,
                'stream':   False,
                'think':    False,
                'options':  {
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens,
                    'num_ctx':     self.num_ctx,
                },
            }

            full_content, api_tool_calls_r1 = self._stream_with_tts(payload)
            response_msg = {
                'role':       'assistant',
                'content':    full_content,
                'tool_calls': api_tool_calls_r1,
            }

            # ── Обработка tool calls ───────────────────────────────────────
            tool_calls = response_msg.get('tool_calls', [])
            # Fallback: Qwen3 иногда эмитирует tool calls текстом в content
            if not tool_calls:
                content_text = response_msg.get('content', '')
                tool_calls = _extract_text_tool_calls(content_text)
                if tool_calls:
                    self.get_logger().info(
                        f'Fallback: извлечено {len(tool_calls)} tool call(s) из текста')
            if tool_calls:
                self.history.append(response_msg)

                tool_results    = []
                speak_texts     = []   # speak_text из аргументов action-инструментов
                needs_llm_reply = False  # True если инструмент возвращает данные (запрос)
                any_tool_failed = False  # True если хотя бы один tool вернул success:False

                # ── Параллельное выполнение tool calls ────────────────────────────
                # Мета-инструменты (express_emotion, set_voice_style) мгновенны;
                # HTTP-инструменты (items_control × N, get_weather) могут идти
                # одновременно — экономит время при N > 1 action-инструментах.
                _QUERY_FNS = frozenset(('get_openhab_states', 'search_openhab_items',
                                        'web_search', 'get_weather', 'search_memory'))

                def _exec_one(tc):
                    fn   = tc['function']['name']
                    args = tc['function'].get('arguments', {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    args = dict(args)
                    st   = args.pop('speak_text', None)  # мета-параметр TTS
                    self.get_logger().info(f'Tool call: {fn}({args})')
                    res  = self._execute_tool(fn, args)
                    self.get_logger().info(f'Tool result: {res}')
                    return fn, st, res

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(len(tool_calls), 4),
                    thread_name_prefix='llm_tc',
                ) as _pool:
                    # Результаты в оригинальном порядке tool_calls
                    _tc_results = list(_pool.map(_exec_one, tool_calls))

                for fn_name, speak_text, result in _tc_results:
                    if speak_text and speak_text.strip():
                        speak_texts.append(speak_text.strip())
                    if fn_name in _QUERY_FNS:
                        needs_llm_reply = True
                    if isinstance(result, dict) and result.get('success') is False:
                        any_tool_failed = True
                    tool_results.append({
                        'role':    'tool',
                        'content': json.dumps(result, ensure_ascii=False),
                    })

                self.history.extend(tool_results)

                # Дедупликация: LLM может дать одинаковый speak_text нескольким tool calls
                # (например, items_control для Dimmer + Color одной комнаты).
                _seen: set = set()
                speak_texts = [st for st in speak_texts
                               if not (_seen.__contains__(st) or _seen.add(st))]

                # Инструменты, за которые BT сам произносит текст (через robot_events).
                # R2 для них не нужен — он создаёт дублирующую речь и повторные Speak goals.
                _BT_SPEECH_TOOLS = frozenset({'robot_control', 'broadcast_message'})
                any_bt_speech_tool = any(
                    tc['function']['name'] in _BT_SPEECH_TOOLS for tc in tool_calls
                )

                if speak_texts and not needs_llm_reply and not any_tool_failed:
                    # Все инструменты успешны, LLM уже написал ответ — второй запрос не нужен
                    final_text = ' '.join(speak_texts)
                    self.history.append({'role': 'assistant', 'content': final_text})
                    self.get_logger().info(
                        f'Ответ из speak_text за {time.time()-_t0:.1f}с: "{final_text}"')
                    with self._lock:
                        _vs = self._voice_style.get('instruct', '')
                        _tg = self._tg_req_id
                    if _tg:
                        for st in speak_texts:
                            self._tg_stream_partial(st, _tg)
                    else:
                        # Bistream вместо action server: ниже задержка, нет goal-handshake
                        ctrl = String()
                        ctrl.data = f'start:{_vs}'
                        self._bs_ctrl_pub.publish(ctrl)
                        for st in speak_texts:
                            ctrl = String()
                            ctrl.data = f'text:{st}'
                            self._bs_ctrl_pub.publish(ctrl)
                        ctrl = String()
                        ctrl.data = 'end'
                        self._bs_ctrl_pub.publish(ctrl)
                    self._publish_response('', streamed=True)
                elif any_bt_speech_tool and not needs_llm_reply and not any_tool_failed:
                    # BT обрабатывает речь через robot_events — R2 не нужен.
                    # _publish_response НЕ вызываем: иначе person_present=True вызовет
                    # повторный тик BT и второй Speak goal.
                    # Если инструмент упал (any_tool_failed=True) — падаем в else → R2 с ошибкой.
                    self.get_logger().info(
                        f'robot_control/broadcast_message: пропускаем R2 — '
                        f'BT обрабатывает речь ({time.time()-_t0:.1f}с)')
                else:
                    # Инструменты вернули данные — нужен LLM для формирования ответа
                    # web_search убран из tools: если поиск уже выполнен (успешно или нет),
                    # повторная попытка не нужна — LLM должен сформировать текстовый ответ
                    # В R2 только мета-инструменты: установка эмоции/голоса.
                    # Action-инструменты (save_memory, items_control, robot_control,
                    # get_weather, web_search) в R2 недопустимы — там нет новых данных
                    # от пользователя, LLM будет их галлюцинировать.
                    # items_control включён: после search_openhab_items в R1 LLM должен
                    # уметь вызвать его в R2 через правильный tool call API.
                    tools_r2 = [t for t in TOOLS
                                if t['function']['name'] in (
                                    'express_emotion', 'set_voice_style', 'items_control')]
                    # Инъекция краткого напоминания прямо перед финальным ответом:
                    # LLM должен ответить на конкретный вопрос пользователя,
                    # а не пересказывать все поля из результата инструмента.
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
                        'tools':    tools_r2,
                        'stream':   False,
                        'think':    False,
                        'options':  {'temperature': self.temperature,
                                     'num_predict': 128, 'num_ctx': self.num_ctx},
                    }
                    r2_content, api_tc_r2 = self._stream_with_tts(payload2)
                    resp2 = {'role': 'assistant',
                             'content': r2_content, 'tool_calls': api_tc_r2}

                    # Проверяем tool calls во втором ответе (proper API или текстовый формат)
                    tool_calls2 = resp2.get('tool_calls', [])
                    if not tool_calls2:
                        c2 = resp2.get('content', '')
                        tool_calls2 = _extract_text_tool_calls(c2)
                        if tool_calls2:
                            self.get_logger().info(
                                f'Fallback R2: {len(tool_calls2)} tool call(s) из текста')

                    _R2_ALLOWED = frozenset(
                        ('express_emotion', 'set_voice_style', 'items_control'))

                    if tool_calls2:
                        # Выполняем только мета-инструменты и items_control.
                        # Собираем speak_text: если R2 вызвал express_emotion только с speak_text
                        # и без content — используем speak_text напрямую, без R3.
                        self.history.append(resp2)
                        _r2_speak: list[str] = []
                        for tc in tool_calls2:
                            fn2   = tc['function']['name']
                            if fn2 not in _R2_ALLOWED:
                                self.get_logger().warn(
                                    f'R2 text-fallback: пропускаем запрещённый инструмент {fn2}')
                                continue
                            args2 = tc['function'].get('arguments', {})
                            if isinstance(args2, str):
                                args2 = json.loads(args2)
                            args2 = dict(args2)
                            st2 = args2.pop('speak_text', None)
                            if st2 and st2.strip():
                                _r2_speak.append(st2.strip())
                            self.get_logger().info(f'Tool call R2: {fn2}({args2})')
                            res2 = self._execute_tool(fn2, args2)
                            self.get_logger().info(f'Tool result R2: {res2}')
                        # Текстовый ответ: сначала content, fallback — speak_text мета-инструментов
                        final_text = _strip_tool_blocks(resp2.get('content', '').strip())
                        _r2_from_speak = False
                        if not final_text and _r2_speak:
                            final_text = ' '.join(_r2_speak)
                            _r2_from_speak = True
                            self.get_logger().info(
                                f'R2 speak_text из meta tool → "{final_text[:60]}"')
                        if final_text:
                            self.history.append({'role': 'assistant', 'content': final_text})
                            self.get_logger().info(
                                f'Финальный ответ R2 за {time.time()-_t0:.1f}с: "{final_text}"')
                            if _r2_from_speak:
                                # Текст ещё не был в bistream — отправляем
                                with self._lock:
                                    _vs2 = self._voice_style.get('instruct', '')
                                    _tg2 = self._tg_req_id
                                if _tg2:
                                    self._tg_stream_partial(final_text, _tg2)
                                else:
                                    _ctrl = String()
                                    _ctrl.data = f'start:{_vs2}'
                                    self._bs_ctrl_pub.publish(_ctrl)
                                    _ctrl = String()
                                    _ctrl.data = f'text:{final_text}'
                                    self._bs_ctrl_pub.publish(_ctrl)
                                    _ctrl = String()
                                    _ctrl.data = 'end'
                                    self._bs_ctrl_pub.publish(_ctrl)
                        else:
                            # LLM вернул пустой R2 даже без speak_text — редкий случай, нужен R3
                            self.get_logger().warn(
                                'R2: пустой content и нет speak_text — запускаем R3')
                            payload3 = {
                                'model':    self.model,
                                'messages': (
                                    [{'role': 'system', 'content': system_prompt}]
                                    + _flatten_tool_history(
                                        self.history,
                                        'Обязательно ответь пользователю одним коротким '
                                        'разговорным предложением — молчать нельзя.')
                                ),
                                'tools':    [],
                                'stream':   True,
                                'think':    False,
                                'options':  {'temperature': self.temperature,
                                             'num_predict': 128, 'num_ctx': self.num_ctx},
                            }
                            try:
                                r3_content, _ = self._stream_with_tts(payload3)
                                final_text = _strip_tool_blocks(r3_content.strip())
                                if final_text:
                                    self.history.append(
                                        {'role': 'assistant', 'content': final_text})
                                    self.get_logger().info(
                                        f'Финальный ответ R3 за {time.time()-_t0:.1f}с: '
                                        f'"{final_text}"')
                                else:
                                    self.get_logger().warn('R3 пуст после R2 express_emotion — молчим')
                            except Exception as _e3:
                                self.get_logger().warn(f'R3 ошибка (после R2 express_emotion): {_e3}')
                        # Текст уже стримился в TTS через _stream_with_tts; BT обработает эмоцию/жест
                        self._publish_response('', streamed=True)
                    else:
                        final_text = _strip_tool_blocks(resp2.get('content', '').strip())
                        if final_text:
                            self.history.append({'role': 'assistant', 'content': final_text})
                            self.get_logger().info(
                                f'Финальный ответ за {time.time()-_t0:.1f}с: "{final_text}"')
                            self._publish_response('', streamed=True)
                        else:
                            # R2 пуст — если были только action-инструменты (не query),
                            # делаем минимальный R3 с требованием ответить
                            if not needs_llm_reply:
                                self.get_logger().warn('R2 пуст после action tool — пробуем R3')
                                payload3 = {
                                    'model':    self.model,
                                    'messages': (
                                        [{'role': 'system', 'content': system_prompt}]
                                        + _flatten_tool_history(self.history,
                                            'Действие выполнено. Ответь пользователю одним коротким предложением — '
                                            'продолжи разговор естественно, не упоминая факт сохранения.')
                                    ),
                                    'tools':    [],
                                    'stream':   False,
                                    'think':    False,
                                    'options':  {'temperature': self.temperature,
                                                 'num_predict': 64, 'num_ctx': self.num_ctx},
                                }
                                try:
                                    r3_content, _ = self._stream_with_tts(payload3)
                                    final_text = _strip_tool_blocks(r3_content.strip())
                                    if final_text:
                                        self.history.append(
                                            {'role': 'assistant', 'content': final_text})
                                        self.get_logger().info(
                                            f'Финальный ответ R3 за {time.time()-_t0:.1f}с: "{final_text}"')
                                        self._publish_response('', streamed=True)
                                    else:
                                        self.get_logger().warn('Пустой финальный ответ LLM (R3) — молчим')
                                except Exception as _e3:
                                    self.get_logger().warn(f'R3 ошибка: {_e3}')
                            else:
                                self.get_logger().warn('Пустой финальный ответ LLM — молчим')

            else:
                text = response_msg.get('content', '').strip()
                self.history.append({'role': 'assistant', 'content': text})
                self.get_logger().info(f'Текстовый ответ за {time.time()-_t0:.1f}с: "{text}"')
                # Текст уже стримился в TTS; BT обрабатывает только эмоцию/жест
                self._publish_response('', streamed=True)

        except requests.exceptions.ConnectionError as e:
            self.get_logger().error(f'Ollama недоступен (оба сервера): {e}')
            self._publish_response('Извини, не могу связаться с сервером обработки. Попробуй позже.')
        except requests.exceptions.Timeout:
            self.get_logger().error(f'Таймаут Ollama после {time.time()-_t0:.1f}с (лимит={self.timeout_sec}с)')
            self._publish_response('Извини, сервер слишком долго не отвечает. Попробуй задать вопрос покороче.')
        except Exception as e:
            self.get_logger().error(f'Ошибка LLM: {e}')
            self._publish_response('Извини, произошла ошибка. Попробуй ещё раз.')
        finally:
            with self._lock:
                self._processing = False

    # ── Выполнение tool calls ──────────────────────────────────────────────

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
        elif fn_name == 'express_emotion':
            return self._tool_express_emotion(args)
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
        else:
            return {'error': f'Unknown function: {fn_name}'}

    # Координаты по умолчанию — Bødalen, Asker, Норвегия
    _DEFAULT_LAT  = 59.835
    _DEFAULT_LON  = 10.440
    _DEFAULT_LOC  = 'Bødalen, Asker'
    _YR_UA        = 'InMoov-Robot/1.0 (fedjukevitsh@gmail.com)'

    # Символы yr.no → русские описания
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
        import datetime
        from collections import Counter

        location = (args.get('location') or '').strip() or self._DEFAULT_LOC
        date_arg  = (args.get('date') or '').strip().lower()

        # Определяем целевую дату
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

        # Геокодирование через Nominatim (если не дефолтная локация)
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
                self.get_logger().warn(f'Геокодирование не удалось ({e}), используем Bødalen')
                resolved_loc = self._DEFAULT_LOC

        # Запрос к api.met.no
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

        # Фильтрация по дате
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

        temps   = [e['temp']   for e in entries if e['temp']   is not None]
        winds   = [e['wind']   for e in entries if e['wind']   is not None]
        precips = [e['precip'] for e in entries]

        # Главное описание — самый частый символ (без суффикса _day/_night/_polartwilight)
        symbols = [e['symbol'].split('_')[0] for e in entries if e['symbol']]
        main_sym  = Counter(symbols).most_common(1)[0][0] if symbols else ''
        main_desc = self._YR_SYMBOLS.get(main_sym, main_sym)

        # Почасовой прогноз (каждые 3 часа, дневное время 7–22)
        hourly = [
            f"{e['hour']:02d}:00 {e['temp']}°C "
            f"{self._YR_SYMBOLS.get(e['symbol'].split('_')[0], e['symbol'])}"
            for e in entries
            if e['hour'] % 3 == 0 and 7 <= e['hour'] <= 22
        ]

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

        # Валидация имени по кэшу — до вызова API
        with self._lock:
            known = {it['name'] for it in self._oh_items}
        if known and name not in known:
            # Ищем похожие имена (общий префикс по '_')
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
        """Сохранить заметку о человеке или общее знание в БД через /memory/query."""
        key       = args.get('key', '')
        value     = args.get('value', '')
        person_id = args.get('person_id')

        if not key or not value:
            return {'error': 'key and value are required'}

        if person_id is not None:
            req = {'op': 'set_note', 'person_id': int(person_id), 'key': key, 'value': value}
        else:
            req = {'op': 'set_knowledge', 'key': key, 'value': value}

        # Вызываем синхронно из фонового потока (LLM уже в thread)
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
        event.setdefault('priority', 10)   # голосовые команды — высокий приоритет
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

        # Блокируем фоновый поток до получения реального результата от BM (Tavily)
        got = self._search_event.wait(timeout=30.0)

        with self._lock:
            self._waiting_for_search = False
            result_text = self._search_result_data

        if got and result_text:
            return {'success': True, 'result': result_text, 'query': query}
        return {'success': False, 'error': 'поиск не вернул результат за 30 секунд', 'query': query}

    # Маппинг эмоции → инструкция голоса для CosyVoice3
    # Лицо и голос меняются одновременно при старте TTS
    _EMOTION_VOICE = {
        'neutral':    '',
        'happy':      'говори радостно и энергично, с улыбкой в голосе',
        'smile':      'говори тепло и дружелюбно, мягко',
        'sad':        'говори тихо и с грустью, медленно',
        'angry':      'говори строго и напряжённо, уверенно',
        'surprise':   'говори удивлённо и с восхищением',
        'fear':       'говори испуганно и тревожно',
        'disgust':    'говори с явным недовольством и отвращением',
        'thinking':   'говори задумчиво и неторопливо, с паузами',
        'sorry':      'говори с искренним сожалением и виновато',
        'suspicious': 'говори настороженно и подозрительно',
        'unamused':   'говори скучно и без энтузиазма, монотонно',
        'sigh':       'говори устало, как после долгого вздоха',
        'wink':       'говори игриво и с хитринкой',
        'sleeping':   'говори тихо и сонно',
    }

    def _tool_express_emotion(self, args: dict) -> dict:
        emotion  = args.get('emotion', 'neutral').lower()
        instruct = self._EMOTION_VOICE.get(emotion, '')
        with self._lock:
            # Голос применится при отправке TTS goal (_tts_dispatch)
            self._voice_style    = {'instruct': instruct}
            # Лицо — откладываем до старта TTS, чтобы запустить синхронно
            self._pending_emotion = emotion
        self.get_logger().info(f'Эмоция запланирована: {emotion} | голос: "{instruct}"')
        return {'success': True, 'emotion': emotion}

    def _tool_set_voice_style(self, args: dict) -> dict:
        instruct = args.get('instruct', '')
        with self._lock:
            self._voice_style = {'instruct': instruct}
        self.get_logger().info(f'Стиль голоса: "{instruct}"')
        return {'success': True, 'instruct': instruct}

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

    # Русские названия комнат → английские подстроки для поиска в OpenHAB
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
        # Переводим русское название комнаты в английское для поиска
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
            # фильтр по семантической группе (точное совпадение имени группы)
            if group_filter and group_filter not in it.get('groups', []):
                continue
            # фильтр по имени/лейблу
            if name_contains:
                if (name_contains not in it['name'].lower() and
                        name_contains not in it.get('label', '').lower()):
                    continue
            # фильтр по состоянию
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

    # ── Callbacks для OpenHAB bridge ──────────────────────────────────────

    def _oh_schema_callback(self, msg: String):
        with self._lock:
            self._oh_schema = msg.data
        self.get_logger().debug('OpenHAB schema получена')

    def _oh_items_callback(self, msg: String):
        try:
            items = json.loads(msg.data)
            with self._lock:
                self._oh_items = items
            self.get_logger().debug(f'OpenHAB items обновлены: {len(items)} устройств')
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'Невалидный openhab_items JSON: {e}')

    def _social_context_cb(self, msg: String):
        """Перехватываем looking_at_robot из social_context."""
        try:
            data = json.loads(msg.data)
            with self._lock:
                # looking_at_robot: True/False от identity_manager, None если нет данных
                raw = data.get('looking_at_robot')
                if raw is not None:
                    self._looking_at_robot = bool(raw)
                self._person_present_in_ctx = bool(data.get('person_present', False))
        except Exception:
            pass

    def _person_context_callback(self, msg: String):
        try:
            ctx = json.loads(msg.data)
            with self._lock:
                old_ctx          = self._person_context
                history_snapshot = self.history[:]
            old_name = old_ctx.get('name') if old_ctx else None
            new_name = ctx.get('name')
            # Сменился человек → публикуем conversation_end для предыдущего
            if old_name and new_name != old_name and history_snapshot:
                self._publish_conversation_end(history_snapshot, old_ctx)
                with self._lock:
                    self.history         = []
                    self._dialogue_lines = []
            with self._lock:
                # Храним контекст только при известном человеке; иначе явно None.
                # Это гарантирует что person_block пуст в system_prompt до INTERACTING.
                # _memory_context не трогаем: рабочая память остаётся в контексте,
                # персональные эпизоды отрезаются в _query_llm по person_id.
                self._person_context = ctx if ctx.get('person_id') else None
            name = ctx.get('name') or 'Незнакомец'
            self.get_logger().debug(f'Person context: {name}')
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'Невалидный person_context JSON: {e}')

    def _search_result_callback(self, msg: String):
        search_text = msg.data.strip()
        if not search_text:
            return
        with self._lock:
            if not self._waiting_for_search:
                self.get_logger().warn('Результат поиска получен без активного web_search — игнорируем')
                return
            self._search_result_data = search_text
            self._waiting_for_search = False
        self._search_event.set()

    def _publish_response(self, text: str, user_text: str = '', streamed: bool = False):
        """Публикует ответ LLM в /llm_response → BT читает из Blackboard и оркеструет речь.

        Формат: {text, voice_instruct, emotion, streamed, telegram}
        Если streamed=True: текст уже отправлен в TTS напрямую;
        BT запускает только ExpressEmotion + Gesticulation (SpeakBehaviour получает пустой текст).
        telegram=True: запрос пришёл через Telegram — BM не должен ставить person_present=True.
        """
        if not streamed:
            if not text or not text.strip():
                return
            # Защитная очистка: убираем остатки tool call блоков если regex не сработал выше
            text = _strip_tool_blocks(text)
            text = _clean_llm_text(text)
            if not text:
                return

        with self._lock:
            instruct          = self._voice_style.get('instruct', '')
            emotion           = self._pending_emotion or 'neutral'
            self._pending_emotion = None
            self._voice_style     = {'instruct': ''}
            tg_req_id         = self._tg_req_id
            self._tg_req_id   = ''

        payload = {
            'text':           '' if streamed else text,
            'voice_instruct': instruct,
            'emotion':        emotion,
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
                + (f' [{emotion}]' if emotion != 'neutral' else '')
                + (f' voice="{instruct}"' if instruct else '')
            )
        else:
            self.get_logger().info(
                f'→ BT: "{text[:70]}{"..." if len(text)>70 else ""}"'
                + (' [TG]' if tg_req_id else '')
                + (f' [{emotion}]' if emotion != 'neutral' else '')
                + (f' voice="{instruct}"' if instruct else '')
            )

        # Переслать ответ в Telegram если запрос пришёл через /telegram_ask
        if tg_req_id:
            # streamed=True: текст уже доставлен через partial-чанки, шлём пустой сигнал "стоп"
            # streamed=False: текст — это error-строка (недоступен Ollama и т.п.)
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
        """Публикует транскрипт завершённого диалога → memory_node сохранит эпизод."""
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
            f'conversation_end: {len(lines)} строк, участники: {participants}')

    def _tool_search_memory(self, args: dict) -> dict:
        """Поиск в долговременной семантической памяти через memory_node."""
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
            self.get_logger().info(f'search_memory: "{query}" → {len(facts)} фактов')
            return result
        except Exception as e:
            return {'error': str(e)}

    def _tool_set_reminder(self, args: dict) -> dict:
        """Сохранить напоминание для пользователя через /memory/query."""
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
        """Удалить показанные напоминания после подтверждения пользователем."""
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

    # ── Chromecast broadcast ───────────────────────────────────────────────

    def _tool_broadcast_message(self, args: dict) -> dict:
        """
        1. POST /tts/to_file на TTS сервер (192.168.10.118:8000):
           сервер синтезирует WAV, пишет в /etc/openhab/html/, возвращает URL.
        2. Устанавливает громкость LivingRoom_Chromecast_volume.
        3. Отправляет URL в LivingRoom_Chromecast_uri → Chromecast воспроизводит.
        """
        text   = (args.get('text') or '').strip()
        volume = int(args.get('volume') or self._cast_volume)

        if not text:
            return {'success': False, 'error': 'text обязателен'}

        # ── TTS сервер синтезирует и сохраняет файл ───────────────────────
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
                f'({data.get("bytes", "?")} байт) → {file_url}')
        except Exception as e:
            self.get_logger().error(f'Cast to_file ошибка: {e}')
            return {'success': False, 'error': str(e)}

        # ── Устанавливаем громкость ────────────────────────────────────────
        try:
            requests.post(
                f'{self.openhab_url}/rest/items/LivingRoom_Chromecast_volume',
                data=str(volume),
                headers={'Content-Type': 'text/plain'},
                timeout=5.0,
            )
            time.sleep(0.3)
        except Exception as e:
            self.get_logger().warn(f'Cast: громкость не установлена: {e}')

        # ── Отправляем URI на Chromecast ───────────────────────────────────
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
        """Безопасный declare_parameter: игнорирует повторное объявление при re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('ollama_url',          'http://192.168.10.118:11434/api/chat')
        self._dp('ollama_fallback_url', 'http://localhost:11434/api/chat')
        self._dp('model',               'qwen3.6:27b')
        self._dp('model_fallback',      'qwen2.5:7b')
        self._dp('temperature',         0.1)
        self._dp('max_tokens',          512)
        self._dp('num_ctx',             8192)
        self._dp('connect_timeout_sec', 5.0)
        self._dp('timeout_sec',         120.0)
        self._dp('keep_history',        True)
        self._dp('history_max_turns',   8)
        self._dp('openhab_url',         'http://192.168.10.118:8080')
        self._dp('tts_server_url',      'http://192.168.10.118:8000')
        self._dp('tts_fallback_url',    'http://localhost:8000')
        self._dp('cast_volume',         80)
        self._dp('cast_to_file_url',    'http://192.168.10.118:8000/tts/to_file')

        self.ollama_primary    = self.get_parameter('ollama_url').value
        self.ollama_fallback   = self.get_parameter('ollama_fallback_url').value
        self.model_primary     = self.get_parameter('model').value
        self.model_fallback    = self.get_parameter('model_fallback').value
        self.temperature       = self.get_parameter('temperature').value
        self.max_tokens        = self.get_parameter('max_tokens').value
        self.num_ctx           = self.get_parameter('num_ctx').value
        self.connect_timeout   = self.get_parameter('connect_timeout_sec').value
        self.timeout_sec       = self.get_parameter('timeout_sec').value
        self.keep_history      = self.get_parameter('keep_history').value
        self.history_max       = self.get_parameter('history_max_turns').value
        self.openhab_url       = self.get_parameter('openhab_url').value
        self._tts_url          = self.get_parameter('tts_server_url').value
        self._tts_fallback_url = self.get_parameter('tts_fallback_url').value
        self._cast_volume      = self.get_parameter('cast_volume').value
        self._cast_to_file_url = self.get_parameter('cast_to_file_url').value
        self._active_url       = self.ollama_primary

        _latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(String, 'voice_command',   self.command_callback,         10)
        self.create_subscription(String, '/telegram_ask',   self._telegram_ask_cb,         10)
        self.create_subscription(String, 'search_result',   self._search_result_callback,  10)
        self.create_subscription(String, 'person_context',  self._person_context_callback, 10)
        self.create_subscription(String, '/social_context', self._social_context_cb,       10)
        self.create_subscription(String, 'openhab_schema',  self._oh_schema_callback,      10)
        self.create_subscription(String, 'openhab_items',   self._oh_items_callback,       10)
        self.create_subscription(Bool,   '/introducing',    self._introducing_cb,          10)
        self.create_subscription(Bool,   '/go_idle',        self._go_idle_cb,              10)
        self.create_subscription(Bool,   '/robot_sleep',    self._robot_sleep_cb,          _latched)
        self.create_subscription(String, '/memory/context', self._memory_context_cb,       10)

        self.event_pub           = self.create_lifecycle_publisher(String, 'robot_events',       10)
        self._response_pub       = self.create_lifecycle_publisher(String, '/llm_response',      10)
        self._tg_resp_pub        = self.create_lifecycle_publisher(String, '/telegram_response', 10)
        self._tts_cancel_pub     = self.create_lifecycle_publisher(Bool,   '/tts_cancel_queue',  10)
        self._conv_end_pub       = self.create_lifecycle_publisher(String, '/conversation_end',  10)
        self._bs_ctrl_pub        = self.create_lifecycle_publisher(String, '/tts/stream_ctrl',   50)

        self._mem_client         = self.create_client(MemoryQuery, '/memory/query')
        self._tts_direct_client  = ActionClient(self, Speak, 'speak')
        self.get_logger().info('LLM нода настроена')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.event_pub.on_activate(state)
        self._response_pub.on_activate(state)
        self._tg_resp_pub.on_activate(state)
        self._tts_cancel_pub.on_activate(state)
        self._conv_end_pub.on_activate(state)
        self._bs_ctrl_pub.on_activate(state)
        self._check_servers()
        # Фоновый прогрев: ждём схему от openhab_bridge (10-15с), затем
        # отправляем минимальный запрос — загружаем модель и наполняем KV-cache.
        threading.Thread(target=self._warmup_llm, daemon=True).start()
        self.get_logger().info(f'LLM нода готова. Модель: {self.model}')
        return TransitionCallbackReturn.SUCCESS

    def _warmup_llm(self):
        """Прогрев: загрузить модель в GPU и наполнить KV-cache системного промпта."""
        time.sleep(15.0)  # ждём openhab_bridge_node публикует схему (каждые 10с)
        _t = time.time()
        try:
            with self._lock:
                oh_schema = self._oh_schema
            sys_prompt = build_system_prompt(oh_schema, None, '')
            payload = {
                'model':   self.model,
                'messages': [
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user',   'content': 'Привет'},
                ],
                'tools':   TOOLS,
                'stream':  False,
                'think':   False,
                'options': {
                    'temperature': 0.0,
                    'num_predict': 3,
                    'num_ctx':     self.num_ctx,
                },
            }
            for _ in self._stream_ollama(payload, read_timeout=120.0):
                pass
            self.get_logger().info(f'LLM прогрев завершён за {time.time()-_t:.1f}с')
        except Exception as e:
            self.get_logger().warn(f'LLM прогрев: ошибка {e}')

    def on_deactivate(self, state):
        self.event_pub.on_deactivate(state)
        self._response_pub.on_deactivate(state)
        self._tg_resp_pub.on_deactivate(state)
        self._tts_cancel_pub.on_deactivate(state)
        self._conv_end_pub.on_deactivate(state)
        self._bs_ctrl_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def _tool_merge_persons(self, args: dict) -> dict:
        """Слить дубликат person_id → target person_id.

        Шаги:
        1. lookup_by_name для обоих имён
        2. merge_persons с check_similarity=True
        3. Если similarity_too_low — сообщаем пользователю
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
            self.get_logger().warn(f'merge_persons отклонён: {reason} — {msg}')
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
