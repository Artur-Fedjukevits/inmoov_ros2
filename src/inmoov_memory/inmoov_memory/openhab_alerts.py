"""
openhab_alerts.py — Shared sensor alert logic for OpenHAB monitoring.

Used by:
  openhab_bridge_node  — creates alerts when thresholds are crossed

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import math
import re

# ── Thresholds ────────────────────────────────────────────────────────────────────

TEMP_MIN     = 16.0   # °C
TEMP_MAX     = 27.0   # °C
HUMIDITY_MIN = 30.0   # %
HUMIDITY_MAX = 70.0   # %
CO2_MAX      = 1200.0 # ppm
VOC_MAX      = 300.0  # ppb
RADON_ST_MAX = 200.0  # Bq/m³ — short-term (> 200: elevated risk)
RADON_LT_MAX = 100.0  # Bq/m³ — long-term (> 100: WHO action level)
BATTERY_MIN  = 10.0   # %

# Outdoor sensors — not monitored for temperature/humidity
_OUTDOOR_PREFIXES = {'EntranceOutside', 'Terrace'}

# Room names in Russian (used in the spoken/pushed alert messages)
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

# Regex for the numeric value in strings such as "23.5 °C", "84", "34 Bq/m³"
_NUM_RE = re.compile(r'[-+]?\d+(?:\.\d+)?')


# ── Sensor classification ──────────────────────────────────────────────────────

def classify_sensor(item_name: str, item_type: str) -> str | None:
    """Determines the sensor type, or None if this item is not monitored."""
    prefix  = item_name.split('_')[0]
    name_lo = item_name.lower()

    if item_type == 'Color':
        return None
    if 'setpoint' in name_lo or 'targettemp' in name_lo:
        return None

    # Batteries are monitored for ALL sensors, including outdoor ones
    if name_lo.endswith('_batterylow') and item_type == 'Switch':
        return 'battery_low'
    if name_lo.endswith('_battery') and item_type == 'Number':
        return 'battery'

    # Environmental sensors — outdoor ones are excluded
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


# ── Value parsing ───────────────────────────────────────────────────────────

def parse_value(state_str: str) -> float | None:
    """Extracts a number from a string such as '23.5 °C', '84', '34 Bq/m³'."""
    if not state_str or state_str in ('NULL', 'UNDEF', 'None', '-', ''):
        return None
    m = _NUM_RE.search(state_str)
    return float(m.group()) if m else None


# ── Threshold evaluation ──────────────────────────────────────────────────────────────

def evaluate_threshold(item_name: str, sensor_type: str,
                       value: float) -> tuple[bool, str]:
    """Checks a value against the critical threshold.

    Returns:
        (True, message) — threshold exceeded, message text for the reminder
        (False, '')       — value is normal
    """
    if sensor_type == 'battery':
        device_raw = item_name[: item_name.lower().rfind('_battery')]
        device = device_raw.replace('Phone_', 'телефон ').replace('_', ' ')
        if value < BATTERY_MIN:
            # Round up to the nearest 5% for a more accurate message:
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
