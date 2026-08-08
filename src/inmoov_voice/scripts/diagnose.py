#!/usr/bin/env python3
"""
diagnose.py — Диагностика всех компонентов InMoov Voice Pipeline.

Запуск:
    python3 src/inmoov_voice/scripts/diagnose.py
    python3 src/inmoov_voice/scripts/diagnose.py --quick   # без inference-тестов

Проверяет:
  - Сетевые сервисы (Ollama, TTS, OpenHAB)
  - Модели (загружены ли в Ollama, LLM inference)
  - Файловую систему (wake word модель)
  - Аудио устройства
  - Python пакеты
  - ROS2 окружение
"""

import argparse
import importlib
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional


# ── ANSI цвета ────────────────────────────────────────────────────────────────

USE_COLOR = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f'\033[{code}m{text}\033[0m' if USE_COLOR else text

OK    = lambda t: _c(t, '32')   # зелёный
WARN  = lambda t: _c(t, '33')   # жёлтый
FAIL  = lambda t: _c(t, '31')   # красный
INFO  = lambda t: _c(t, '36')   # голубой
BOLD  = lambda t: _c(t, '1')    # жирный
DIM   = lambda t: _c(t, '2')    # тёмный


# ── Результат проверки ─────────────────────────────────────────────────────────

@dataclass
class Check:
    name:    str
    status:  str   # 'ok' | 'warn' | 'fail' | 'skip'
    message: str   = ''
    detail:  str   = ''

    def icon(self) -> str:
        return {'ok': OK('✓'), 'warn': WARN('⚠'), 'fail': FAIL('✗'), 'skip': DIM('–')}[self.status]

    def label(self) -> str:
        return {'ok': OK('OK  '), 'warn': WARN('WARN'), 'fail': FAIL('FAIL'), 'skip': DIM('SKIP')}[self.status]


# ── Конфиг ────────────────────────────────────────────────────────────────────

PRIMARY_HOST    = '192.168.10.118'
OLLAMA_PRIMARY  = f'http://{PRIMARY_HOST}:11434'
OLLAMA_LOCAL    = 'http://localhost:11434'
TTS_PRIMARY     = f'http://{PRIMARY_HOST}:8000'
TTS_LOCAL       = 'http://localhost:8000'
OPENHAB_URL     = f'http://{PRIMARY_HOST}:8080'
LLM_MODEL       = 'qwen2.5:14b-instruct-q8_0'
WAKEWORD_MODEL  = '/home/artur/openWakeWord/my_custom_model/ey_lyonya.onnx'
TIMEOUT         = 4.0


# ── Вспомогательные функции ────────────────────────────────────────────────────

def _get(url: str, timeout: float = TIMEOUT, **kwargs):
    import requests
    return requests.get(url, timeout=timeout, **kwargs)


def _post(url: str, **kwargs):
    import requests
    return requests.post(url, timeout=TIMEOUT, **kwargs)


def _probe(url: str, timeout: float = 2.0) -> bool:
    try:
        import requests
        requests.get(url, timeout=timeout)
        return True
    except Exception:
        return False


def _ollama_models(base_url: str) -> list[str]:
    try:
        r = _get(f'{base_url}/api/tags')
        return [m['name'] for m in r.json().get('models', [])]
    except Exception:
        return []


def _tts_health(url: str) -> Optional[dict]:
    try:
        return _get(f'{url}/health').json()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# СЕКЦИИ ПРОВЕРОК
# ══════════════════════════════════════════════════════════════════════════════

def check_ollama(quick: bool) -> list[Check]:
    checks = []

    for label, base_url in [('Основной (RTX 3090)', OLLAMA_PRIMARY),
                              ('Локальный (CPU/ROCm)', OLLAMA_LOCAL)]:
        name = f'Ollama {label}'
        if not _probe(base_url):
            checks.append(Check(name, 'fail', f'Недоступен ({base_url})'))
            continue

        models = _ollama_models(base_url)
        model_base = LLM_MODEL.split(':')[0]
        found = [m for m in models if model_base in m]

        if found:
            checks.append(Check(name, 'ok', f'{base_url}', f'Модель: {found[0]}'))
        else:
            checks.append(Check(
                name, 'warn', f'{base_url} — сервер ОК, но модель не загружена',
                f'Запусти: ollama pull {LLM_MODEL}\nДоступные: {models[:5]}'
            ))

    # Inference тест (только основного, не quick)
    if not quick and _probe(OLLAMA_PRIMARY):
        models = _ollama_models(OLLAMA_PRIMARY)
        model_base = LLM_MODEL.split(':')[0]
        if any(model_base in m for m in models):
            name = 'Ollama inference'
            try:
                t0 = time.time()
                import requests as req
                r = req.post(
                    f'{OLLAMA_PRIMARY}/api/chat',
                    json={
                        'model': LLM_MODEL,
                        'messages': [{'role': 'user', 'content': 'Привет.'}],
                        'stream': False,
                        'options': {'num_predict': 5},
                    },
                    timeout=30.0,
                )
                dt = time.time() - t0
                text = r.json().get('message', {}).get('content', '').strip()
                checks.append(Check(
                    name, 'ok', f'Ответ за {dt:.1f}с', f'Текст: "{text}"'
                ))
            except Exception as e:
                checks.append(Check(name, 'fail', str(e)))

    return checks


def check_tts(quick: bool) -> list[Check]:
    checks = []

    for label, base_url in [('Основной (RTX 3090)', TTS_PRIMARY),
                              ('Локальный (ROCm)', TTS_LOCAL)]:
        name = f'TTS {label}'
        health = _tts_health(base_url)
        if health is None:
            checks.append(Check(name, 'warn' if base_url == TTS_LOCAL else 'fail',
                                f'Недоступен ({base_url})'))
            continue

        gpu  = health.get('gpu', 'N/A')
        vram = f"{health.get('vram_used_mb', '?')}/{health.get('vram_total_mb', '?')} MB"
        checks.append(Check(name, 'ok', base_url, f'GPU: {gpu}, VRAM: {vram}'))

    # Синтез тест (только основного, не quick)
    if not quick and _probe(TTS_PRIMARY):
        name = 'TTS синтез'
        try:
            import requests as req
            t0 = time.time()
            r = req.post(
                f'{TTS_PRIMARY}/tts/stream',
                json={'text': 'Тест.'},
                stream=True,
                timeout=(5.0, 20.0),
            )
            data = b''.join(r.iter_content(4096))
            dt = time.time() - t0
            # Определяем sample rate
            ct = r.headers.get('Content-Type', '')
            sr = 'N/A'
            for part in ct.split(';'):
                if part.strip().startswith('rate='):
                    sr = part.strip()[5:]
            checks.append(Check(
                name, 'ok' if data else 'fail',
                f'{len(data)} байт за {dt:.1f}с',
                f'Sample rate: {sr} Hz'
            ))
        except Exception as e:
            checks.append(Check(name, 'fail', str(e)))

    return checks


def check_openhab() -> list[Check]:
    checks = []
    name = f'OpenHAB ({PRIMARY_HOST}:8080)'

    if not _probe(OPENHAB_URL):
        return [Check(name, 'warn', 'Недоступен — умный дом не будет работать')]

    try:
        all_items = _get(f'{OPENHAB_URL}/rest/items').json()
        llm_items = _get(f'{OPENHAB_URL}/rest/items?tags=ChatGPT').json()
        checks.append(Check(
            name, 'ok' if llm_items else 'warn',
            f'{len(llm_items)} устройств с тегом ChatGPT (из {len(all_items)} всего)',
            f'Без тега ChatGPT LLM не видит устройства' if not llm_items else ''
        ))
    except Exception as e:
        checks.append(Check(name, 'fail', str(e)))

    return checks


def check_filesystem() -> list[Check]:
    checks = []

    # Wake word модель
    name = 'Wake word модель'
    if os.path.exists(WAKEWORD_MODEL):
        size_kb = os.path.getsize(WAKEWORD_MODEL) // 1024
        checks.append(Check(name, 'ok', WAKEWORD_MODEL, f'{size_kb} KB'))
    else:
        checks.append(Check(
            name, 'fail',
            f'Не найдена: {WAKEWORD_MODEL}',
            'Необходима для обнаружения "Эй Лёня"'
        ))

    # ~/.cache/torch (Silero VAD)
    silero_cache = os.path.expanduser('~/.cache/torch/hub/snakers4_silero-vad_master')
    name = 'Silero VAD cache'
    if os.path.isdir(silero_cache):
        checks.append(Check(name, 'ok', silero_cache))
    else:
        checks.append(Check(
            name, 'warn',
            'Не закэширован — будет скачан при первом запуске (~8MB)',
            silero_cache
        ))

    # Whisper cache
    whisper_cache = os.path.expanduser('~/.cache/huggingface')
    name = 'Whisper model cache'
    if os.path.isdir(whisper_cache):
        # ищем medium модель
        found = any(
            'medium' in root
            for root, _, _ in os.walk(whisper_cache)
            if 'faster-whisper' in root
        )
        if found:
            checks.append(Check(name, 'ok', 'faster-whisper medium найдена'))
        else:
            checks.append(Check(
                name, 'warn',
                'faster-whisper medium не найдена — будет скачана при первом запуске (~1.5GB)',
                whisper_cache
            ))
    else:
        checks.append(Check(name, 'warn', 'Hugging Face cache не найден'))

    return checks


def check_audio() -> list[Check]:
    checks = []

    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        count = pa.get_device_count()

        input_devices = []
        for i in range(count):
            info = pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                input_devices.append(f"[{i}] {info['name']} "
                                     f"({int(info['defaultSampleRate'])} Hz)")

        pa.terminate()

        if input_devices:
            checks.append(Check(
                'Аудио входы', 'ok',
                f'{len(input_devices)} устройств найдено',
                '\n    '.join(input_devices)
            ))
        else:
            checks.append(Check('Аудио входы', 'fail', 'Нет входных аудио устройств'))

    except ImportError:
        checks.append(Check('PyAudio', 'fail', 'Не установлен: pip install pyaudio'))
    except Exception as e:
        checks.append(Check('Аудио', 'fail', str(e)))

    try:
        import sounddevice as sd
        devs = sd.query_devices()
        output_devs = [d for d in devs if d['max_output_channels'] > 0]
        if output_devs:
            default_out = sd.query_devices(kind='output')
            checks.append(Check(
                'Аудио выход', 'ok',
                f"По умолчанию: {default_out['name']}",
                f'{int(default_out["default_samplerate"])} Hz'
            ))
        else:
            checks.append(Check('Аудио выход', 'fail', 'Нет выходных устройств'))
    except Exception as e:
        checks.append(Check('sounddevice', 'fail', str(e)))

    return checks


def check_python_packages() -> list[Check]:
    checks = []
    packages = [
        ('rclpy',         'ROS2 Python клиент'),
        ('faster_whisper','Whisper STT'),
        ('torch',         'PyTorch (Silero VAD)'),
        ('openwakeword',  'Wake word детектор'),
        ('pyaudio',       'Захват микрофона'),
        ('sounddevice',   'Воспроизведение аудио'),
        ('requests',      'HTTP клиент'),
        ('numpy',         'Обработка аудио'),
        ('py_trees',      'Behavior Tree'),
    ]
    for pkg, desc in packages:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, '__version__', '?')
            checks.append(Check(f'{pkg}', 'ok', desc, f'v{version}'))
        except ImportError:
            checks.append(Check(f'{pkg}', 'fail', f'{desc} — не установлен'))

    # inmoov_msgs action
    try:
        from inmoov_msgs.action import Speak  # noqa: F401
        checks.append(Check('inmoov_msgs.action.Speak', 'ok', 'Action definition'))
    except ImportError:
        checks.append(Check(
            'inmoov_msgs', 'fail',
            'Не собран: colcon build --packages-select inmoov_msgs'
        ))

    return checks


def check_ros2_env() -> list[Check]:
    checks = []

    # ROS_DISTRO
    distro = os.environ.get('ROS_DISTRO', '')
    if distro:
        ok = distro == 'jazzy'
        checks.append(Check(
            'ROS_DISTRO', 'ok' if ok else 'warn',
            distro,
            '' if ok else 'Ожидается jazzy'
        ))
    else:
        checks.append(Check(
            'ROS_DISTRO', 'fail',
            'Не установлен',
            'Запусти: source /opt/ros/jazzy/setup.bash'
        ))

    # Workspace
    ament_path = os.environ.get('AMENT_PREFIX_PATH', '')
    ws_path = '/home/artur/ros2_ws'
    if ws_path in ament_path:
        checks.append(Check('Workspace sourced', 'ok', ws_path))
    else:
        checks.append(Check(
            'Workspace sourced', 'warn',
            f'Workspace не найден в AMENT_PREFIX_PATH',
            f'Запусти: source {ws_path}/install/setup.bash'
        ))

    # ROS2 topics (если daemon запущен)
    try:
        import subprocess
        result = subprocess.run(
            ['ros2', 'node', 'list'],
            capture_output=True, text=True, timeout=3.0
        )
        nodes = [n for n in result.stdout.strip().split('\n') if n]
        if nodes:
            checks.append(Check(
                'Запущенные ноды', 'ok' if nodes else 'skip',
                f'{len(nodes)} нод активно',
                ', '.join(nodes[:8]) + ('...' if len(nodes) > 8 else '')
            ))
        else:
            checks.append(Check('Запущенные ноды', 'skip', 'Нет активных нод'))
    except Exception:
        checks.append(Check('ros2 CLI', 'warn', 'Недоступен'))

    return checks


# ══════════════════════════════════════════════════════════════════════════════
# ВЫВОД
# ══════════════════════════════════════════════════════════════════════════════

def print_section(title: str, checks: list[Check]):
    print(f'\n{BOLD(title)}')
    print('─' * 60)
    for c in checks:
        msg = f'{c.icon()} {c.label()}  {c.name}'
        if c.message:
            msg += f'  {DIM("—")} {c.message}'
        print(msg)
        if c.detail:
            for line in c.detail.strip().split('\n'):
                print(f'         {DIM(line)}')


def print_summary(all_checks: list[Check]):
    total = len(all_checks)
    ok    = sum(1 for c in all_checks if c.status == 'ok')
    warn  = sum(1 for c in all_checks if c.status == 'warn')
    fail  = sum(1 for c in all_checks if c.status == 'fail')
    skip  = sum(1 for c in all_checks if c.status == 'skip')

    print(f'\n{"─"*60}')
    status = 'ok' if fail == 0 and warn == 0 else ('warn' if fail == 0 else 'fail')
    icon   = {'ok': OK('✓'), 'warn': WARN('⚠'), 'fail': FAIL('✗')}[status]
    print(f'{icon}  {ok} OK  |  {warn} WARN  |  {fail} FAIL  |  {skip} SKIP  (всего {total})')

    if fail > 0:
        print(f'\n{FAIL("Критические проблемы:")}')
        for c in all_checks:
            if c.status == 'fail':
                print(f'  {FAIL("✗")} {c.name}: {c.message}')

    if warn > 0:
        print(f'\n{WARN("Предупреждения:")}')
        for c in all_checks:
            if c.status == 'warn':
                print(f'  {WARN("⚠")} {c.name}: {c.message}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='InMoov Voice Pipeline Diagnostic')
    parser.add_argument('--quick', action='store_true',
                        help='Пропустить тесты inference (быстрый режим)')
    parser.add_argument('--no-audio', action='store_true',
                        help='Не проверять аудио устройства')
    args = parser.parse_args()

    print(BOLD('\n╔══════════════════════════════════════════════════════╗'))
    print(BOLD('║     InMoov Voice Pipeline — Диагностика              ║'))
    print(BOLD('╚══════════════════════════════════════════════════════╝'))
    if args.quick:
        print(DIM('  [быстрый режим — inference тесты пропущены]'))

    all_checks: list[Check] = []

    sections = [
        ('🔌  ROS2 окружение',   check_ros2_env()),
        ('📦  Python пакеты',    check_python_packages()),
        ('📁  Файловая система', check_filesystem()),
        ('🧠  Ollama LLM',       check_ollama(args.quick)),
        ('🔊  TTS Server',       check_tts(args.quick)),
        ('🏠  OpenHAB',          check_openhab()),
    ]
    if not args.no_audio:
        sections.append(('🎤  Аудио устройства', check_audio()))

    for title, checks in sections:
        print_section(title, checks)
        all_checks.extend(checks)

    print_summary(all_checks)
    print()

    # Возвращаем код ошибки если есть fail
    fail_count = sum(1 for c in all_checks if c.status == 'fail')
    sys.exit(1 if fail_count > 0 else 0)


if __name__ == '__main__':
    main()
