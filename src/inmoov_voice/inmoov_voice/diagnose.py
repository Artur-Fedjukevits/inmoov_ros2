#!/usr/bin/env python3
"""
diagnose.py — Diagnostics for all components of the InMoov Voice Pipeline.

Usage:
    ros2 run inmoov_voice diagnose
    ros2 run inmoov_voice diagnose --quick   # without inference tests

Checks:
  - Network services (vLLM, TTS, OpenHAB)
  - Models (whether they are loaded in vLLM, LLM inference)
  - Filesystem (wake word model, Silero VAD / Parakeet STT caches)
  - Audio devices
  - Python packages
  - ROS2 environment

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import argparse
import importlib
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional


# ── ANSI colors ────────────────────────────────────────────────────────────────

USE_COLOR = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f'\033[{code}m{text}\033[0m' if USE_COLOR else text

OK    = lambda t: _c(t, '32')   # green
WARN  = lambda t: _c(t, '33')   # yellow
FAIL  = lambda t: _c(t, '31')   # red
INFO  = lambda t: _c(t, '36')   # cyan
BOLD  = lambda t: _c(t, '1')    # bold
DIM   = lambda t: _c(t, '2')    # dim


# ── Check result ─────────────────────────────────────────────────────────

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


# ── Config ────────────────────────────────────────────────────────────────────

PRIMARY_HOST    = '192.168.10.118'
LLM_PRIMARY     = f'http://{PRIMARY_HOST}:18020'   # vLLM, OpenAI-compatible API
LLM_LOCAL       = ''   # local fallback removed (empty = not checked)
LLM_BEARER      = os.environ.get('VLLM_BEARER_TOKEN', '')
TTS_PRIMARY     = f'http://{PRIMARY_HOST}:8000'
TTS_LOCAL       = ''   # local CosyVoice fallback removed (empty = not checked)
OPENHAB_URL     = f'http://{PRIMARY_HOST}:8080'
LLM_MODEL       = 'qwen3.8-27b'
WAKEWORD_MODEL  = os.path.expanduser('~/openWakeWord/my_custom_model/ey_lyonya.onnx')
TIMEOUT         = 4.0


# ── Helper functions ────────────────────────────────────────────────────

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


def _llm_models(base_url: str) -> list[str]:
    try:
        headers = {'Authorization': f'Bearer {LLM_BEARER}'} if LLM_BEARER else {}
        r = _get(f'{base_url}/v1/models', headers=headers)
        return [m['id'] for m in r.json().get('data', [])]
    except Exception:
        return []


def _tts_health(url: str) -> Optional[dict]:
    try:
        return _get(f'{url}/health').json()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CHECK SECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def check_llm(quick: bool) -> list[Check]:
    checks = []

    for label, base_url in [('Primary (vLLM, RTX 3090)', LLM_PRIMARY),
                              ('Local (NUC)', LLM_LOCAL)]:
        if not base_url:
            continue
        name = f'LLM {label}'
        if not _probe(f'{base_url}/health'):
            checks.append(Check(name, 'fail', f'Unreachable ({base_url})'))
            continue

        models = _llm_models(base_url)
        model_base = LLM_MODEL.split(':')[0]
        found = [m for m in models if model_base in m]

        if found:
            checks.append(Check(name, 'ok', f'{base_url}', f'Model: {found[0]}'))
        else:
            checks.append(Check(
                name, 'warn', f'{base_url} — server OK, but model not found',
                f'Expected: {LLM_MODEL}\nAvailable: {models[:5]}'
            ))

    # Inference test (primary only, not in quick mode)
    if not quick and _probe(f'{LLM_PRIMARY}/health'):
        models = _llm_models(LLM_PRIMARY)
        model_base = LLM_MODEL.split(':')[0]
        if any(model_base in m for m in models):
            name = 'LLM inference'
            try:
                t0 = time.time()
                import requests as req
                headers = {'Authorization': f'Bearer {LLM_BEARER}'} if LLM_BEARER else {}
                r = req.post(
                    f'{LLM_PRIMARY}/v1/chat/completions',
                    headers=headers,
                    json={
                        'model': LLM_MODEL,
                        'messages': [{'role': 'user', 'content': 'Привет.'}],
                        'stream': False,
                        'max_tokens': 5,
                        'chat_template_kwargs': {'enable_thinking': False},
                    },
                    timeout=30.0,
                )
                dt = time.time() - t0
                choices = r.json().get('choices') or []
                text = (choices[0].get('message', {}).get('content', '') if choices else '').strip()
                checks.append(Check(
                    name, 'ok', f'Answered in {dt:.1f}s', f'Text: "{text}"'
                ))
            except Exception as e:
                checks.append(Check(name, 'fail', str(e)))

    return checks


def check_tts(quick: bool) -> list[Check]:
    checks = []

    for label, base_url in [('Primary (RTX 5060)', TTS_PRIMARY),
                              ('Local (ROCm)', TTS_LOCAL)]:
        if not base_url:
            continue
        name = f'TTS {label}'
        health = _tts_health(base_url)
        if health is None:
            checks.append(Check(name, 'warn' if base_url == TTS_LOCAL else 'fail',
                                f'Unreachable ({base_url})'))
            continue

        gpu  = health.get('gpu', 'N/A')
        vram = f"{health.get('vram_used_mb', '?')}/{health.get('vram_total_mb', '?')} MB"
        checks.append(Check(name, 'ok', base_url, f'GPU: {gpu}, VRAM: {vram}'))

    # Synthesis test (primary only, not in quick mode)
    if not quick and _probe(TTS_PRIMARY):
        name = 'TTS synthesis'
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
            # Determine the sample rate
            ct = r.headers.get('Content-Type', '')
            sr = 'N/A'
            for part in ct.split(';'):
                if part.strip().startswith('rate='):
                    sr = part.strip()[5:]
            checks.append(Check(
                name, 'ok' if data else 'fail',
                f'{len(data)} bytes in {dt:.1f}s',
                f'Sample rate: {sr} Hz'
            ))
        except Exception as e:
            checks.append(Check(name, 'fail', str(e)))

    return checks


def check_openhab() -> list[Check]:
    checks = []
    name = f'OpenHAB ({PRIMARY_HOST}:8080)'

    if not _probe(OPENHAB_URL):
        return [Check(name, 'warn', 'Unreachable — the smart home will not work')]

    try:
        all_items = _get(f'{OPENHAB_URL}/rest/items').json()
        llm_items = _get(f'{OPENHAB_URL}/rest/items?tags=ChatGPT').json()
        checks.append(Check(
            name, 'ok' if llm_items else 'warn',
            f'{len(llm_items)} devices tagged ChatGPT (out of {len(all_items)} total)',
            'Without the ChatGPT tag the LLM cannot see the devices' if not llm_items else ''
        ))
    except Exception as e:
        checks.append(Check(name, 'fail', str(e)))

    return checks


def check_filesystem() -> list[Check]:
    checks = []

    # Wake word model
    name = 'Wake word model'
    if os.path.exists(WAKEWORD_MODEL):
        size_kb = os.path.getsize(WAKEWORD_MODEL) // 1024
        checks.append(Check(name, 'ok', WAKEWORD_MODEL, f'{size_kb} KB'))
    else:
        checks.append(Check(
            name, 'fail',
            f'Not found: {WAKEWORD_MODEL}',
            'Required to detect "Эй Лёня"'
        ))

    # ~/.cache/torch (Silero VAD)
    silero_cache = os.path.expanduser('~/.cache/torch/hub/snakers4_silero-vad_master')
    name = 'Silero VAD cache'
    if os.path.isdir(silero_cache):
        checks.append(Check(name, 'ok', silero_cache))
    else:
        checks.append(Check(
            name, 'warn',
            'Not cached — will be downloaded on first run (~8MB)',
            silero_cache
        ))

    # Parakeet STT (onnx-asr downloads it into the Hugging Face hub cache)
    hf_hub = os.path.join(
        os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')), 'hub')
    name = 'Parakeet STT model cache'
    found = os.path.isdir(hf_hub) and any('parakeet' in d.lower() for d in os.listdir(hf_hub))
    if found:
        checks.append(Check(name, 'ok', 'parakeet-tdt ONNX model found', hf_hub))
    else:
        checks.append(Check(
            name, 'warn',
            'Parakeet model not cached — will be downloaded on first run',
            hf_hub
        ))

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
                'Audio inputs', 'ok',
                f'{len(input_devices)} devices found',
                '\n    '.join(input_devices)
            ))
        else:
            checks.append(Check('Audio inputs', 'fail', 'No audio input devices'))

    except ImportError:
        checks.append(Check('PyAudio', 'fail', 'Not installed: pip install pyaudio'))
    except Exception as e:
        checks.append(Check('Audio', 'fail', str(e)))

    try:
        import sounddevice as sd
        devs = sd.query_devices()
        output_devs = [d for d in devs if d['max_output_channels'] > 0]
        if output_devs:
            default_out = sd.query_devices(kind='output')
            checks.append(Check(
                'Audio output', 'ok',
                f"Default: {default_out['name']}",
                f'{int(default_out["default_samplerate"])} Hz'
            ))
        else:
            checks.append(Check('Audio output', 'fail', 'No output devices'))
    except Exception as e:
        checks.append(Check('sounddevice', 'fail', str(e)))

    return checks


def check_python_packages() -> list[Check]:
    checks = []
    packages = [
        ('rclpy',         'ROS2 Python client'),
        ('onnx_asr',      'Parakeet STT (onnx-asr)'),
        ('speechbrain',   'ECAPA speaker verification / voice emotion'),
        ('torch',         'PyTorch (Silero VAD)'),
        ('openwakeword',  'Wake word detector'),
        ('pyaudio',       'Microphone capture'),
        ('sounddevice',   'Audio playback'),
        ('requests',      'HTTP client'),
        ('numpy',         'Audio processing'),
    ]
    for pkg, desc in packages:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, '__version__', '?')
            checks.append(Check(f'{pkg}', 'ok', desc, f'v{version}'))
        except ImportError:
            checks.append(Check(f'{pkg}', 'fail', f'{desc} — not installed'))

    # inmoov_msgs action
    try:
        from inmoov_msgs.action import Speak  # noqa: F401
        checks.append(Check('inmoov_msgs.action.Speak', 'ok', 'Action definition'))
    except ImportError:
        checks.append(Check(
            'inmoov_msgs', 'fail',
            'Not built: colcon build --packages-select inmoov_msgs'
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
            '' if ok else 'jazzy expected'
        ))
    else:
        checks.append(Check(
            'ROS_DISTRO', 'fail',
            'Not set',
            'Run: source /opt/ros/jazzy/setup.bash'
        ))

    # Workspace
    ament_path = os.environ.get('AMENT_PREFIX_PATH', '')
    ws_path = os.path.expanduser('~/ros2_ws')
    if ws_path in ament_path:
        checks.append(Check('Workspace sourced', 'ok', ws_path))
    else:
        checks.append(Check(
            'Workspace sourced', 'warn',
            'Workspace not found in AMENT_PREFIX_PATH',
            f'Run: source {ws_path}/install/setup.bash'
        ))

    # ROS2 nodes (if the daemon is running)
    try:
        import subprocess
        result = subprocess.run(
            ['ros2', 'node', 'list'],
            capture_output=True, text=True, timeout=3.0
        )
        nodes = [n for n in result.stdout.strip().split('\n') if n]
        if nodes:
            checks.append(Check(
                'Running nodes', 'ok' if nodes else 'skip',
                f'{len(nodes)} nodes active',
                ', '.join(nodes[:8]) + ('...' if len(nodes) > 8 else '')
            ))
        else:
            checks.append(Check('Running nodes', 'skip', 'No active nodes'))
    except Exception:
        checks.append(Check('ros2 CLI', 'warn', 'Unavailable'))

    return checks


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
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
    print(f'{icon}  {ok} OK  |  {warn} WARN  |  {fail} FAIL  |  {skip} SKIP  (total {total})')

    if fail > 0:
        print(f'\n{FAIL("Critical problems:")}')
        for c in all_checks:
            if c.status == 'fail':
                print(f'  {FAIL("✗")} {c.name}: {c.message}')

    if warn > 0:
        print(f'\n{WARN("Warnings:")}')
        for c in all_checks:
            if c.status == 'warn':
                print(f'  {WARN("⚠")} {c.name}: {c.message}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='InMoov Voice Pipeline Diagnostic')
    parser.add_argument('--quick', action='store_true',
                        help='Skip inference tests (quick mode)')
    parser.add_argument('--no-audio', action='store_true',
                        help='Do not check audio devices')
    args = parser.parse_args()

    print(BOLD('\n╔══════════════════════════════════════════════════════╗'))
    print(BOLD('║     InMoov Voice Pipeline — Diagnostics            ║'))
    print(BOLD('╚══════════════════════════════════════════════════════╝'))
    if args.quick:
        print(DIM('  [quick mode — inference tests skipped]'))

    all_checks: list[Check] = []

    sections = [
        ('🔌  ROS2 environment',   check_ros2_env()),
        ('📦  Python packages',    check_python_packages()),
        ('📁  Filesystem', check_filesystem()),
        ('🧠  vLLM',             check_llm(args.quick)),
        ('🔊  TTS Server',       check_tts(args.quick)),
        ('🏠  OpenHAB',          check_openhab()),
    ]
    if not args.no_audio:
        sections.append(('🎤  Audio devices', check_audio()))

    for title, checks in sections:
        print_section(title, checks)
        all_checks.extend(checks)

    print_summary(all_checks)
    print()

    # Return a non-zero exit code if any check failed
    fail_count = sum(1 for c in all_checks if c.status == 'fail')
    sys.exit(1 if fail_count > 0 else 0)


if __name__ == '__main__':
    main()
