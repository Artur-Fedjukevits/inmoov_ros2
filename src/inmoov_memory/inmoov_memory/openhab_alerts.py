"""
openhab_alerts.py — Shared sensor alert logic for OpenHAB monitoring.

Used by:
  openhab_bridge_node  — creates alerts when thresholds are crossed
  identity_manager_node — verifies env reminders before announcing (fresh state)
"""

import math
import re

# ── Пороги ────────────────────────────────────────────────────────────────────

TEMP_MIN     = 16.0   # °C
TEMP_MAX     = 27.0   # °C
HUMIDITY_MIN = 30.0   # %
HUMIDITY_MAX = 70.0   # %
CO2_MAX      = 1200.0 # ppm
VOC_MAX      = 300.0  # ppb
RADON_ST_MAX = 200.0  # Bq/m³ — краткосрочный (> 200: повышенный риск)
RADON_LT_MAX = 100.0  # Bq/m³ — долгосрочный (> 100: уровень действия ВОЗ)
BATTERY_MIN  = 10.0   # %

# Уличные датчики — не мониторим по температуре/влажности
_OUTDOOR_PREFIXES = {'EntranceOutside', 'Terrace'}

# Названия комнат на русском
_ROOM_RU = {
    'Bathroom':      'Ванная',
    'Bedroom':       'Спальня',
    'ChildrensRoom': 'Детская',
    'Entrance':      'Прихожая',
    'GuestRoom':     'Гостевая',
    'Hall':          'Холл',
    'Kitchen':       'Кухня',
    'LivingRoom':    'Гостиная',
    'Stairs':        'Лестница',
    'Wardrobe':      'Гардероб',
    'WC':            'Туалет',
}

# Regex для числового значения из строк вида "23.5 °C", "84", "34 Bq/m³"
_NUM_RE = re.compile(r'[-+]?\d+(?:\.\d+)?')


# ── Классификация датчика ──────────────────────────────────────────────────────

def classify_sensor(item_name: str, item_type: str) -> str | None:
    """Определяет тип датчика или None если этот item не мониторим."""
    prefix  = item_name.split('_')[0]
    name_lo = item_name.lower()

    if item_type == 'Color':
        return None
    if 'setpoint' in name_lo or 'targettemp' in name_lo:
        return None

    # Батарейки мониторим для ВСЕХ датчиков, включая уличные
    if name_lo.endswith('_batterylow') and item_type == 'Switch':
        return 'battery_low'
    if name_lo.endswith('_battery') and item_type == 'Number':
        return 'battery'

    # Экологические датчики — уличные исключаем
    if prefix in _OUTDOOR_PREFIXES:
        return None

    if item_type == 'Number:Temperature' and name_lo.endswith('_temp'):
        return 'temp'
    if 'humidity' in name_lo and item_type in ('Number', 'Number:Dimensionless'):
        return 'humidity'
    if '_co2' in name_lo and item_type in ('Number', 'Number:Dimensionless'):
        return 'co2'
    if '_voc' in name_lo and item_type in ('Number', 'Number:Dimensionless'):
        return 'voc'
    if name_lo.endswith('_radon_st') and item_type in ('Number', 'Number:Dimensionless'):
        return 'radon_st'
    if name_lo.endswith('_radon_lt') and item_type in ('Number', 'Number:Dimensionless'):
        return 'radon_lt'

    return None


# ── Парсинг значения ───────────────────────────────────────────────────────────

def parse_value(state_str: str) -> float | None:
    """Извлекает число из строки вида '23.5 °C', '84', '34 Bq/m³'."""
    if not state_str or state_str in ('NULL', 'UNDEF', 'None', '-', ''):
        return None
    m = _NUM_RE.search(state_str)
    return float(m.group()) if m else None


# ── Оценка порога ──────────────────────────────────────────────────────────────

def evaluate_threshold(item_name: str, sensor_type: str,
                       value: float) -> tuple[bool, str]:
    """Проверяет значение на критический порог.

    Returns:
        (True, сообщение) — превышение порога, message для напоминания
        (False, '')       — значение в норме
    """
    if sensor_type == 'battery':
        device_raw = item_name[: item_name.lower().rfind('_battery')]
        device = device_raw.replace('Phone_', 'телефон ').replace('_', ' ')
        if value < BATTERY_MIN:
            # Округляем вверх до ближайших 5% для более точного сообщения:
            # 8% → <10%, 4% → <5%, 18% → <20%
            threshold = math.ceil(value / 5) * 5
            return True, f'Разряжен {device} (заряд <{threshold}%)'
        return False, ''

    room = _ROOM_RU.get(item_name.split('_')[0], item_name.split('_')[0])

    if sensor_type == 'temp':
        if value > TEMP_MAX:
            return True, f'Температура в «{room}» слишком высокая: {value:.1f}°C (норма <{TEMP_MAX:.0f}°C)'
        if value < TEMP_MIN:
            return True, f'Температура в «{room}» слишком низкая: {value:.1f}°C (норма >{TEMP_MIN:.0f}°C)'
    elif sensor_type == 'humidity':
        if value > HUMIDITY_MAX:
            return True, f'Влажность в «{room}» слишком высокая: {value:.0f}% (норма <{HUMIDITY_MAX:.0f}%)'
        if value < HUMIDITY_MIN:
            return True, f'Влажность в «{room}» слишком низкая: {value:.0f}% (норма >{HUMIDITY_MIN:.0f}%)'
    elif sensor_type == 'co2':
        if value > CO2_MAX:
            return True, f'Уровень CO₂ в «{room}»: {value:.0f} ppm (норма <{CO2_MAX:.0f} ppm)'
    elif sensor_type == 'voc':
        if value > VOC_MAX:
            return True, f'Уровень VOC в «{room}»: {value:.0f} ppb (норма <{VOC_MAX:.0f} ppb)'
    elif sensor_type == 'radon_st':
        if value > RADON_ST_MAX:
            return True, (f'Повышенный уровень радона в «{room}» '
                          f'(краткосрочный: {value:.0f} Бк/м³, норма <{RADON_ST_MAX:.0f})')
    elif sensor_type == 'radon_lt':
        if value > RADON_LT_MAX:
            return True, (f'Повышенный уровень радона в «{room}» '
                          f'(долгосрочный: {value:.0f} Бк/м³, норма ВОЗ <{RADON_LT_MAX:.0f})')

    return False, ''
