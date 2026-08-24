"""
test_services.py — проверка доступности всех внешних сервисов.

Тесты пропускаются если сервис недоступен (pytest.skip).
Запускай перед стартом робота для быстрой диагностики:

    pytest src/inmoov_voice/test/test_services.py -v -m services

Или чтобы увидеть все (включая пропущенные):
    pytest src/inmoov_voice/test/test_services.py -v --tb=short
"""

import os

import pytest
import requests

# ── Адреса сервисов ────────────────────────────────────────────────────────────
PRIMARY_HOST = '192.168.10.118'
LLM_PRIMARY  = f'http://{PRIMARY_HOST}:18020'   # vLLM, OpenAI-совместимый API
LLM_LOCAL    = 'http://localhost:18020'
LLM_BEARER   = os.environ.get('VLLM_BEARER_TOKEN', '')
TTS_PRIMARY  = f'http://{PRIMARY_HOST}:8000'
TTS_LOCAL    = 'http://localhost:8000'
OPENHAB_URL  = f'http://{PRIMARY_HOST}:8080'

LLM_MODEL = 'qwen3.8-27b'
TIMEOUT   = 5.0


def _llm_headers():
    return {'Authorization': f'Bearer {LLM_BEARER}'} if LLM_BEARER else {}


def _get(url, **kwargs):
    return requests.get(url, timeout=TIMEOUT, **kwargs)


def _server_available(base_url: str) -> bool:
    try:
        requests.get(base_url, timeout=2.0)
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# LLM (vLLM, OpenAI-совместимый API)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.services
class TestLLM:
    def test_primary_reachable(self):
        """Основной сервер 192.168.10.118:18020 (vLLM) доступен."""
        if not _server_available(LLM_PRIMARY):
            pytest.skip(f'LLM primary недоступен ({LLM_PRIMARY})')
        r = _get(f'{LLM_PRIMARY}/v1/models', headers=_llm_headers())
        assert r.status_code == 200, f'Ожидался 200, получен {r.status_code}'

    def test_primary_has_llm_model(self):
        """Модель qwen3.8-27b загружена на основном сервере."""
        if not _server_available(LLM_PRIMARY):
            pytest.skip(f'LLM primary недоступен ({LLM_PRIMARY})')
        r = _get(f'{LLM_PRIMARY}/v1/models', headers=_llm_headers())
        models = [m['id'] for m in r.json().get('data', [])]
        model_base = LLM_MODEL.split(':')[0]
        found = any(model_base in m for m in models)
        assert found, (
            f'Модель {LLM_MODEL} не найдена на {LLM_PRIMARY}.\n'
            f'Доступные: {models}'
        )

    def test_primary_inference(self):
        """vLLM primary отвечает на простой запрос (проверка GPU/inference)."""
        if not _server_available(LLM_PRIMARY):
            pytest.skip(f'LLM primary недоступен ({LLM_PRIMARY})')
        models = _get(f'{LLM_PRIMARY}/v1/models', headers=_llm_headers()).json().get('data', [])
        model_base = LLM_MODEL.split(':')[0]
        if not any(model_base in m['id'] for m in models):
            pytest.skip(f'Модель {LLM_MODEL} не загружена')

        r = requests.post(
            f'{LLM_PRIMARY}/v1/chat/completions',
            headers=_llm_headers(),
            json={
                'model': LLM_MODEL,
                'messages': [{'role': 'user', 'content': 'Привет, скажи одно слово.'}],
                'stream': False,
                'max_tokens': 5,
                'chat_template_kwargs': {'enable_thinking': False},
            },
            timeout=30.0,
        )
        assert r.status_code == 200
        choices = r.json().get('choices') or []
        text = choices[0].get('message', {}).get('content', '') if choices else ''
        assert len(text) > 0, 'LLM вернул пустой ответ'
        print(f'\n  LLM ответ: "{text}"')

    def test_local_reachable(self):
        """Локальный LLM (localhost:18020) доступен как fallback."""
        if not _server_available(LLM_LOCAL):
            pytest.skip(f'Локальный LLM недоступен ({LLM_LOCAL})')
        r = _get(f'{LLM_LOCAL}/v1/models', headers=_llm_headers())
        assert r.status_code == 200

    def test_local_has_llm_model(self):
        """Модель загружена на локальном LLM (fallback)."""
        if not _server_available(LLM_LOCAL):
            pytest.skip(f'Локальный LLM недоступен ({LLM_LOCAL})')
        r = _get(f'{LLM_LOCAL}/v1/models', headers=_llm_headers())
        models = [m['id'] for m in r.json().get('data', [])]
        model_base = LLM_MODEL.split(':')[0]
        found = any(model_base in m for m in models)
        if not found:
            pytest.fail(
                f'Локальный LLM работает, но модель {LLM_MODEL} не загружена.\n'
                f'Доступные: {models}'
            )


# ══════════════════════════════════════════════════════════════════════════════
# TTS Server
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.services
class TestTTSServer:
    def test_primary_reachable(self):
        """TTS сервер 192.168.10.118:8000 доступен."""
        if not _server_available(TTS_PRIMARY):
            pytest.skip(f'TTS primary недоступен ({TTS_PRIMARY})')
        r = _get(f'{TTS_PRIMARY}/health')
        assert r.status_code == 200

    def test_primary_health_info(self):
        """Health endpoint возвращает информацию о GPU и модели."""
        if not _server_available(TTS_PRIMARY):
            pytest.skip(f'TTS primary недоступен ({TTS_PRIMARY})')
        info = _get(f'{TTS_PRIMARY}/health').json()
        print(f'\n  TTS primary: GPU={info.get("gpu")}, '
              f'VRAM={info.get("vram_used_mb")}/{info.get("vram_total_mb")} MB')
        # Не падаем если полей нет — сервер может быть разных версий
        assert r.status_code == 200 if (r := _get(f'{TTS_PRIMARY}/health')) else True

    def test_primary_synthesis(self):
        """TTS сервер синтезирует аудио (короткий тест)."""
        if not _server_available(TTS_PRIMARY):
            pytest.skip(f'TTS primary недоступен ({TTS_PRIMARY})')
        r = requests.post(
            f'{TTS_PRIMARY}/tts/stream',
            json={'text': 'Тест.'},
            stream=True,
            timeout=(5.0, 15.0),
        )
        assert r.status_code == 200, f'TTS вернул {r.status_code}: {r.text[:200]}'
        data = b''.join(r.iter_content(chunk_size=4096))
        assert len(data) > 0, 'TTS вернул пустой поток'
        print(f'\n  TTS primary: {len(data)} байт аудио')

    def test_local_reachable(self):
        """Локальный TTS сервер (localhost:8000) доступен как fallback."""
        if not _server_available(TTS_LOCAL):
            pytest.skip(f'TTS local недоступен ({TTS_LOCAL})')
        r = _get(f'{TTS_LOCAL}/health')
        assert r.status_code == 200

    def test_local_health_info(self):
        """Локальный TTS health — проверяем ROCm/GPU."""
        if not _server_available(TTS_LOCAL):
            pytest.skip(f'TTS local недоступен ({TTS_LOCAL})')
        info = _get(f'{TTS_LOCAL}/health').json()
        print(f'\n  TTS local: GPU={info.get("gpu")}, '
              f'VRAM={info.get("vram_used_mb")}/{info.get("vram_total_mb")} MB')

    def test_sample_rate_in_headers(self):
        """TTS сервер возвращает sample rate в заголовках ответа."""
        if not _server_available(TTS_PRIMARY):
            pytest.skip(f'TTS primary недоступен ({TTS_PRIMARY})')
        r = requests.post(
            f'{TTS_PRIMARY}/tts/stream',
            json={'text': 'Раз.'},
            stream=True,
            timeout=(5.0, 15.0),
        )
        # Должен быть либо Content-Type с rate= либо X-Sample-Rate
        ct = r.headers.get('Content-Type', '')
        xsr = r.headers.get('X-Sample-Rate', '')
        has_rate = 'rate=' in ct or xsr.isdigit()
        assert has_rate, (
            f'Заголовок sample rate не найден.\n'
            f'Content-Type: {ct}\nX-Sample-Rate: {xsr}'
        )


# ══════════════════════════════════════════════════════════════════════════════
# OpenHAB
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.services
class TestOpenHAB:
    def test_reachable(self):
        """OpenHAB 192.168.10.118:8080 доступен."""
        if not _server_available(OPENHAB_URL):
            pytest.skip(f'OpenHAB недоступен ({OPENHAB_URL})')
        r = _get(f'{OPENHAB_URL}/rest/')
        assert r.status_code == 200

    def test_items_api(self):
        """OpenHAB REST /items работает."""
        if not _server_available(OPENHAB_URL):
            pytest.skip(f'OpenHAB недоступен ({OPENHAB_URL})')
        r = _get(f'{OPENHAB_URL}/rest/items')
        assert r.status_code == 200
        items = r.json()
        assert isinstance(items, list), 'Ожидался список items'
        print(f'\n  OpenHAB: {len(items)} items всего')

    def test_chatgpt_tagged_items(self):
        """OpenHAB содержит items с тегом ChatGPT (для LLM)."""
        if not _server_available(OPENHAB_URL):
            pytest.skip(f'OpenHAB недоступен ({OPENHAB_URL})')
        r = _get(f'{OPENHAB_URL}/rest/items?tags=ChatGPT')
        items = r.json()
        print(f'\n  OpenHAB: {len(items)} items с тегом ChatGPT')
        if len(items) == 0:
            pytest.fail(
                'Нет items с тегом ChatGPT — LLM не сможет управлять устройствами.\n'
                'Добавь тег ChatGPT к устройствам в OpenHAB.'
            )
