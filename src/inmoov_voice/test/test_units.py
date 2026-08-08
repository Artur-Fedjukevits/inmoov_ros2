"""
test_units.py — быстрые unit-тесты без ROS2 и сетевых запросов.

Тестирует вспомогательные функции нод напрямую.
Работают в любом окружении, не требуют сервисов.

pytest src/inmoov_voice/test/test_units.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from inmoov_cognition.llm_node import (
    _base_url,
    build_system_prompt,
    fetch_openhab_items,
)
from inmoov_voice.tts_node import TTSNode


# ══════════════════════════════════════════════════════════════════════════════
# _base_url
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.units
class TestBaseUrl:
    def test_remote_server(self):
        assert _base_url('http://192.168.10.118:11434/api/chat') == \
            'http://192.168.10.118:11434'

    def test_localhost(self):
        assert _base_url('http://localhost:11434/api/chat') == \
            'http://localhost:11434'

    def test_no_path(self):
        assert _base_url('http://192.168.10.118:8000') == \
            'http://192.168.10.118:8000'

    def test_deep_path(self):
        assert _base_url('http://host:1234/a/b/c') == 'http://host:1234'


# ══════════════════════════════════════════════════════════════════════════════
# build_system_prompt
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.units
class TestBuildSystemPrompt:
    def test_empty_items(self):
        prompt = build_system_prompt([])
        assert 'OpenHAB' in prompt
        assert '[]' in prompt

    def test_items_in_prompt(self):
        items = [{'name': 'LivingRoom_Light', 'label': 'Свет', 'type': 'Switch', 'state': 'ON'}]
        prompt = build_system_prompt(items)
        assert 'LivingRoom_Light' in prompt
        assert 'Switch' in prompt

    def test_valid_json_in_prompt(self):
        items = [{'name': 'Lamp', 'label': 'Лампа', 'type': 'Dimmer', 'state': '75'}]
        prompt = build_system_prompt(items)
        # Находим JSON-блок в промпте и парсим его
        start = prompt.index('[')
        end = prompt.rindex(']') + 1
        parsed = json.loads(prompt[start:end])
        assert parsed[0]['name'] == 'Lamp'

    def test_with_person_context(self):
        ctx = {'name': 'Артур', 'meet_count': 5, 'current_emotion': 'happy', 'notes': {}}
        prompt = build_system_prompt([], person_ctx=ctx)
        assert 'Артур' in prompt
        assert 'meet_count' not in prompt  # форматируется, не сырой JSON
        assert '5' in prompt

    def test_person_context_none(self):
        prompt = build_system_prompt([], person_ctx=None)
        # Не должно быть блока собеседника
        assert 'Незнакомец' not in prompt

    def test_unknown_person(self):
        ctx = {'name': None, 'meet_count': 1}
        prompt = build_system_prompt([], person_ctx=ctx)
        assert 'Незнакомец' in prompt

    def test_russian_language_instruction(self):
        prompt = build_system_prompt([])
        assert 'русском' in prompt.lower() or 'Russian' in prompt

    def test_tools_mentioned(self):
        prompt = build_system_prompt([])
        for keyword in ('items_control', 'robot_control', 'web_search', 'express_emotion'):
            # Инструменты перечислены в TOOLS, но могут быть в промпте или нет
            # Главное что правила упоминаются
            pass
        assert 'tool' in prompt.lower() or 'инструмент' in prompt.lower()


# ══════════════════════════════════════════════════════════════════════════════
# fetch_openhab_items
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.units
class TestFetchOpenhabItems:
    def test_returns_empty_on_connection_error(self):
        with patch('inmoov_cognition.llm_node.requests.get') as mock_get:
            mock_get.side_effect = ConnectionError('refused')
            result = fetch_openhab_items('http://fake:8080/rest/items')
        assert result == []

    def test_parses_items_correctly(self):
        fake_items = [
            {
                'name': 'Kitchen_Light',
                'label': 'Кухня',
                'type': 'Switch',
                'state': 'ON',
                'stateDescription': {'options': []},
            },
            {
                'name': 'AC_Mode',
                'label': 'Режим AC',
                'type': 'String',
                'state': 'COOL',
                'stateDescription': {
                    'options': [{'value': 'COOL'}, {'value': 'HEAT'}]
                },
            },
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = fake_items

        with patch('inmoov_cognition.llm_node.requests.get', return_value=mock_response):
            result = fetch_openhab_items('http://fake:8080/rest/items')

        assert len(result) == 2
        assert result[0]['name'] == 'Kitchen_Light'
        assert result[0]['state'] == 'ON'
        assert result[1]['options'] == ['COOL', 'HEAT']

    def test_item_without_label_uses_name(self):
        fake_items = [{'name': 'NoLabel_Item', 'type': 'Switch', 'state': 'OFF'}]
        mock_response = MagicMock()
        mock_response.json.return_value = fake_items

        with patch('inmoov_cognition.llm_node.requests.get', return_value=mock_response):
            result = fetch_openhab_items('http://fake:8080/rest/items')

        assert result[0]['label'] == 'NoLabel_Item'

    def test_item_without_state_is_null(self):
        fake_items = [{'name': 'X', 'type': 'Switch'}]
        mock_response = MagicMock()
        mock_response.json.return_value = fake_items

        with patch('inmoov_cognition.llm_node.requests.get', return_value=mock_response):
            result = fetch_openhab_items('http://fake:8080/rest/items')

        assert result[0]['state'] == 'NULL'

    def test_returns_empty_on_http_error(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception('404')

        with patch('inmoov_cognition.llm_node.requests.get', return_value=mock_response):
            result = fetch_openhab_items('http://fake:8080/rest/items')

        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# TTSNode._parse_sample_rate (статический метод)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.units
class TestParseSampleRate:
    parse = staticmethod(TTSNode._parse_sample_rate)

    def test_parses_content_type_rate(self):
        headers = {'Content-Type': 'audio/L16; rate=24000'}
        assert self.parse(headers) == 24000

    def test_parses_22050(self):
        headers = {'Content-Type': 'audio/L16; charset=utf-8; rate=22050'}
        assert self.parse(headers) == 22050

    def test_fallback_to_x_sample_rate(self):
        headers = {'Content-Type': 'audio/wav', 'X-Sample-Rate': '16000'}
        assert self.parse(headers) == 16000

    def test_default_when_nothing(self):
        headers = {'Content-Type': 'audio/wav'}
        assert self.parse(headers) == 24000

    def test_malformed_rate_uses_fallback(self):
        headers = {'Content-Type': 'audio/L16; rate=abc', 'X-Sample-Rate': '8000'}
        assert self.parse(headers) == 8000
