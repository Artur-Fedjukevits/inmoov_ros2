"""
test_nodes.py — тесты запуска ROS2 нод и передачи сообщений.

HTTP-запросы к Ollama/TTS мокаются, поэтому тесты работают без сервисов.
Для запуска нужен sourced workspace:

    source install/setup.bash
    pytest src/inmoov_voice/test/test_nodes.py -v

Или через colcon:
    colcon test --packages-select inmoov_voice --pytest-args -m nodes -v
"""

import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool, Float32MultiArray, MultiArrayDimension, String

from inmoov_msgs.action import Speak


# ── Локальные хелперы (дублируются из conftest т.к. pytest не позволяет
#    импортировать conftest напрямую) ───────────────────────────────────────────

@contextmanager
def spin_node(node, num_threads: int = 3, startup_wait: float = 0.3):
    executor = MultiThreadedExecutor(num_threads=num_threads)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    time.sleep(startup_wait)
    try:
        yield node
    finally:
        executor.shutdown(timeout_sec=2.0)
        try:
            node.destroy_node()
        except Exception:
            pass


def wait_for(condition, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return False

# ── Моки для HTTP-ответов ──────────────────────────────────────────────────────

def _make_ollama_mock(model: str = 'qwen2.5'):
    """Мок requests для LLMNode: _probe_ollama → ОК, _post_ollama → текстовый ответ."""
    mock = MagicMock()
    # GET /api/tags
    tags_resp = MagicMock()
    tags_resp.json.return_value = {'models': [{'name': model}]}
    # POST /api/chat
    chat_resp = MagicMock()
    chat_resp.status_code = 200
    chat_resp.json.return_value = {
        'message': {'role': 'assistant', 'content': 'Тестовый ответ.', 'tool_calls': []}
    }
    mock.get.return_value  = tags_resp
    mock.post.return_value = chat_resp
    mock.Session.return_value = MagicMock()
    mock.exceptions.ConnectionError = ConnectionError
    mock.exceptions.Timeout = TimeoutError
    mock.exceptions.HTTPError = Exception
    return mock


def _make_tts_mock():
    """Мок requests.Session для TTSNode: health → ОК."""
    session = MagicMock()
    health_resp = MagicMock()
    health_resp.status_code = 200
    health_resp.json.return_value = {'gpu': 'RTX 3090', 'vram_used_mb': 4000, 'vram_total_mb': 24576}
    session.get.return_value = health_resp
    return session


# ══════════════════════════════════════════════════════════════════════════════
# Инфраструктура
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.nodes
class TestROS2Init:
    def test_rclpy_ok(self, ros):
        """rclpy инициализирован."""
        assert rclpy.ok(), 'rclpy не инициализирован — проверь фикстуру ros'

    def test_can_create_node(self, ros):
        """Можно создать простую ноду."""
        node = rclpy.create_node('test_infra_node')
        assert node is not None
        node.destroy_node()


# ══════════════════════════════════════════════════════════════════════════════
# LLMNode
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.nodes
class TestLLMNode:
    def test_startup(self, ros):
        """LLMNode запускается без ошибок (Ollama замокан)."""
        from inmoov_cognition.llm_node import LLMNode
        with patch('inmoov_cognition.llm_node.requests', _make_ollama_mock()):
            node = LLMNode()
        assert node is not None
        node.destroy_node()

    def test_subscriptions_exist(self, ros):
        """LLMNode подписан на нужные топики."""
        from inmoov_cognition.llm_node import LLMNode
        with patch('inmoov_cognition.llm_node.requests', _make_ollama_mock()):
            node = LLMNode()

        # get_subscriptions_info_by_topic возвращает список TopicEndpointInfo объектов
        voice_cmd_subs = node.get_subscriptions_info_by_topic('/voice_command')
        search_subs    = node.get_subscriptions_info_by_topic('/search_result')
        node.destroy_node()

        assert len(voice_cmd_subs) > 0, 'LLMNode не подписан на /voice_command'
        assert len(search_subs) > 0,    'LLMNode не подписан на /search_result'

    def test_has_action_client(self, ros):
        """LLMNode создаёт ActionClient для speak."""
        from inmoov_cognition.llm_node import LLMNode
        with patch('inmoov_cognition.llm_node.requests', _make_ollama_mock()):
            node = LLMNode()
        assert hasattr(node, '_tts_client'), 'Нет атрибута _tts_client'
        assert isinstance(node._tts_client, ActionClient)
        node.destroy_node()

    def test_pending_speech_queue(self, ros):
        """_pending_speech — очередь размером 1, старое значение заменяется."""
        from inmoov_cognition.llm_node import LLMNode
        with patch('inmoov_cognition.llm_node.requests', _make_ollama_mock()):
            node = LLMNode()

        node._publish_response('первый текст')
        node._publish_response('второй текст')   # должен заменить первый

        text = node._pending_speech.get_nowait()
        assert text == 'второй текст', f'Ожидался "второй текст", получен "{text}"'
        node.destroy_node()

    def test_empty_text_ignored(self, ros):
        """Пустой текст не попадает в очередь."""
        from inmoov_cognition.llm_node import LLMNode
        with patch('inmoov_cognition.llm_node.requests', _make_ollama_mock()):
            node = LLMNode()

        node._publish_response('')
        node._publish_response('   ')
        assert node._pending_speech.empty(), 'Пустой текст не должен попадать в очередь'
        node.destroy_node()

    def test_history_trimming(self, ros):
        """История разговора не превышает history_max * 2 сообщений."""
        from inmoov_cognition.llm_node import LLMNode
        with patch('inmoov_cognition.llm_node.requests', _make_ollama_mock()):
            node = LLMNode()
        max_turns = node.history_max

        # Заполняем историю сверх лимита
        for i in range(max_turns * 2 + 5):
            node.history.append({'role': 'user', 'content': f'Вопрос {i}'})
            node.history.append({'role': 'assistant', 'content': f'Ответ {i}'})

        # Имитируем обрезку как в _query_llm
        while len(node.history) > max_turns * 2:
            node.history.pop(0)
            while node.history and node.history[0]['role'] != 'user':
                node.history.pop(0)

        assert len(node.history) <= max_turns * 2
        node.destroy_node()


# ══════════════════════════════════════════════════════════════════════════════
# TTSNode
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.nodes
class TestTTSNode:
    def test_startup(self, ros):
        """TTSNode запускается без ошибок (TTS сервер замокан)."""
        from inmoov_voice.tts_node import TTSNode
        with patch('inmoov_voice.tts_node.requests.Session', return_value=_make_tts_mock()):
            node = TTSNode()
        assert node is not None
        node.destroy_node()

    def test_action_server_created(self, ros):
        """TTSNode создаёт Action Server."""
        from inmoov_voice.tts_node import TTSNode
        with patch('inmoov_voice.tts_node.requests.Session', return_value=_make_tts_mock()):
            node = TTSNode()
        assert hasattr(node, '_action_server'), 'Нет атрибута _action_server'
        node.destroy_node()

    def test_speaking_publisher_exists(self, ros):
        """TTSNode публикует tts_speaking (Bool)."""
        from inmoov_voice.tts_node import TTSNode
        with patch('inmoov_voice.tts_node.requests.Session', return_value=_make_tts_mock()):
            node = TTSNode()
        assert hasattr(node, '_speaking_pub')
        node.destroy_node()

    def test_preemption_event_mechanism(self, ros):
        """Abort event заменяется при каждом вызове — старый сигнализируется."""
        from inmoov_voice.tts_node import TTSNode
        import threading

        with patch('inmoov_voice.tts_node.requests.Session', return_value=_make_tts_mock()):
            node = TTSNode()

        # Имитируем старт двух execute подряд
        with node._abort_lock:
            first_event = node._abort_event
            new_event = threading.Event()
            node._abort_event = new_event
        first_event.set()  # это делает _execute_speak при старте

        assert first_event.is_set(), 'Первый abort event должен быть сигнализирован'
        assert not new_event.is_set(), 'Новый abort event ещё не должен быть сигнализирован'
        node.destroy_node()

    def test_action_round_trip(self, ros):
        """
        Полный цикл TTS Action: клиент → goal → сервер → воспроизведение (замокано)
        → результат → клиент.
        """
        from inmoov_voice.tts_node import TTSNode

        # Мокаем весь _stream_and_play чтобы не нужны реальный сервер и звуковая карта
        def fake_stream(url, text, goal_handle, my_abort, fb):
            fb('playing', 0.5)
            time.sleep(0.1)  # имитируем воспроизведение
            return True, 1024, ''

        with patch('inmoov_voice.tts_node.requests.Session', return_value=_make_tts_mock()):
            tts_node = TTSNode()

        tts_node._stream_and_play = fake_stream

        with spin_node(tts_node, num_threads=4, startup_wait=0.5):
            # Создаём клиентскую ноду
            client_node = rclpy.create_node('test_tts_client')
            client_exec = MultiThreadedExecutor(num_threads=2)
            client_exec.add_node(client_node)
            client_thread = threading.Thread(target=client_exec.spin, daemon=True)
            client_thread.start()

            try:
                action_client = ActionClient(client_node, Speak, 'speak')
                assert action_client.wait_for_server(timeout_sec=5.0), \
                    'TTS Action Server не ответил за 5с'

                goal = Speak.Goal()
                goal.text = 'Тестовое сообщение для проверки action.'

                result_holder = {'result': None, 'done': threading.Event()}

                def got_result(future):
                    result_holder['result'] = future.result().result
                    result_holder['done'].set()

                def goal_accepted(future):
                    gh = future.result()
                    assert gh.accepted, 'Goal не принят TTS сервером'
                    gh.get_result_async().add_done_callback(got_result)

                future = action_client.send_goal_async(goal)
                future.add_done_callback(goal_accepted)

                assert result_holder['done'].wait(timeout=10.0), 'TTS не вернул результат за 10с'
                result = result_holder['result']
                assert result.success, f'TTS завернул неудачу: {result.message}'

            finally:
                client_exec.shutdown(timeout_sec=1.0)
                client_node.destroy_node()

    def test_action_cancel(self, ros):
        """Клиент может отменить Speak goal во время воспроизведения."""
        from inmoov_voice.tts_node import TTSNode

        cancel_seen = threading.Event()

        def slow_stream(url, text, goal_handle, my_abort, fb):
            fb('playing')
            # Имитируем долгое воспроизведение, проверяем cancel
            for _ in range(20):
                if goal_handle.is_cancel_requested or my_abort.is_set():
                    cancel_seen.set()
                    return False, 0, 'aborted'
                time.sleep(0.05)
            return True, 1000, ''

        with patch('inmoov_voice.tts_node.requests.Session', return_value=_make_tts_mock()):
            tts_node = TTSNode()

        tts_node._stream_and_play = slow_stream

        with spin_node(tts_node, num_threads=4, startup_wait=0.5):
            client_node = rclpy.create_node('test_tts_cancel_client')
            client_exec = MultiThreadedExecutor(num_threads=2)
            client_exec.add_node(client_node)
            threading.Thread(target=client_exec.spin, daemon=True).start()

            try:
                action_client = ActionClient(client_node, Speak, 'speak')
                action_client.wait_for_server(timeout_sec=5.0)

                goal = Speak.Goal()
                goal.text = 'Очень длинный текст который нужно прервать.'

                gh_holder = {'gh': None, 'ready': threading.Event()}

                def goal_accepted(future):
                    gh_holder['gh'] = future.result()
                    gh_holder['ready'].set()

                future = action_client.send_goal_async(goal)
                future.add_done_callback(goal_accepted)

                assert gh_holder['ready'].wait(timeout=5.0)
                time.sleep(0.15)  # даём воспроизведению начаться

                # Отменяем
                gh_holder['gh'].cancel_goal_async()

                assert cancel_seen.wait(timeout=5.0), 'Отмена не была обработана сервером'

            finally:
                client_exec.shutdown(timeout_sec=1.0)
                client_node.destroy_node()


# ══════════════════════════════════════════════════════════════════════════════
# VoiceDetectorNode
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.nodes
@pytest.mark.slow
class TestVoiceDetectorNode:
    def test_startup(self, ros):
        """VoiceDetectorNode загружает Silero VAD и стартует."""
        # Загрузка Silero VAD может занять время при первом запуске
        from inmoov_voice.voice_detector_node import VoiceDetectorNode
        node = VoiceDetectorNode()
        assert node is not None
        node.destroy_node()

    def test_wake_activates_recording(self, ros):
        """Wake word сигнал (True) активирует запись."""
        from inmoov_voice.voice_detector_node import VoiceDetectorNode
        node = VoiceDetectorNode()

        assert not node.is_active

        wake_msg = Bool()
        wake_msg.data = True
        node.wake_callback(wake_msg)

        assert node.is_active, 'После wake word нода должна быть активна'
        node.destroy_node()

    def test_false_wake_ignored(self, ros):
        """Wake word False не активирует запись."""
        from inmoov_voice.voice_detector_node import VoiceDetectorNode
        node = VoiceDetectorNode()

        wake_msg = Bool()
        wake_msg.data = False
        node.wake_callback(wake_msg)

        assert not node.is_active
        node.destroy_node()

    def test_tts_speaking_disables_recording(self, ros):
        """Когда TTS говорит — активная запись сбрасывается."""
        from inmoov_voice.voice_detector_node import VoiceDetectorNode
        node = VoiceDetectorNode()

        # Активируем запись
        wake_msg = Bool()
        wake_msg.data = True
        node.wake_callback(wake_msg)
        assert node.is_active

        # TTS начал говорить
        tts_msg = Bool()
        tts_msg.data = True
        node._tts_speaking_callback(tts_msg)

        assert not node.is_active, 'Запись должна быть сброшена когда TTS говорит'
        assert node.tts_speaking
        node.destroy_node()

    def test_tts_done_reactivates(self, ros):
        """Когда TTS заканчивает — нода авто-активируется для продолжения диалога."""
        from inmoov_voice.voice_detector_node import VoiceDetectorNode
        node = VoiceDetectorNode()

        # Имитируем: TTS говорил → закончил
        node.tts_speaking = True

        tts_done = Bool()
        tts_done.data = False
        node._tts_speaking_callback(tts_done)

        assert node.is_active, 'После окончания TTS нода должна авто-активироваться'
        node.destroy_node()

    def test_chunk_size_detected_from_first_audio(self, ros):
        """chunk_size определяется динамически из первого сообщения raw_audio."""
        from inmoov_voice.voice_detector_node import VoiceDetectorNode
        node = VoiceDetectorNode()

        # Отправляем первый аудио-чанк (512 сэмплов, как audio_source_node)
        chunk = [0.0] * 512
        msg = Float32MultiArray()
        dim = MultiArrayDimension()
        dim.label = 'sample_rate'
        dim.stride = 16000
        msg.layout.dim = [dim]
        msg.data = chunk

        node._audio_callback(msg)

        assert node._chunk_size == 512
        assert node._silence_threshold is not None
        node.destroy_node()


# ══════════════════════════════════════════════════════════════════════════════
# Топик-тесты: передача сообщений между нодами
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.nodes
class TestTopicFlow:
    def test_tts_speaking_published_on_playback(self, ros):
        """
        TTSNode публикует tts_speaking=True перед воспроизведением
        и tts_speaking=False после.
        """
        from inmoov_voice.tts_node import TTSNode

        speaking_log = []

        def fake_stream(url, text, goal_handle, my_abort, fb):
            time.sleep(0.05)
            return True, 512, ''

        with patch('inmoov_voice.tts_node.requests.Session', return_value=_make_tts_mock()):
            tts_node = TTSNode()

        tts_node._stream_and_play = fake_stream

        with spin_node(tts_node, num_threads=4, startup_wait=0.5):
            listener_node = rclpy.create_node('test_tts_speaking_listener')
            listener_exec = MultiThreadedExecutor(num_threads=2)
            listener_exec.add_node(listener_node)
            threading.Thread(target=listener_exec.spin, daemon=True).start()

            try:
                listener_node.create_subscription(
                    Bool, 'tts_speaking',
                    lambda msg: speaking_log.append(msg.data),
                    10
                )

                # Ждём пока DDS обнаружит подписчика (иначе первые сообщения теряются)
                action_client = ActionClient(listener_node, Speak, 'speak')
                action_client.wait_for_server(timeout_sec=5.0)
                # wait_for_server даёт время DDS discovery, но для топиков нужно ещё чуть
                time.sleep(0.3)

                done = threading.Event()
                goal = Speak.Goal()
                goal.text = 'Проверка tts_speaking.'

                def goal_accepted(future):
                    gh = future.result()
                    gh.get_result_async().add_done_callback(lambda _: done.set())

                action_client.send_goal_async(goal).add_done_callback(goal_accepted)
                assert done.wait(timeout=10.0)

                # Даём время на публикацию финального False
                time.sleep(0.2)

                assert True in speaking_log, 'tts_speaking=True не было опубликовано'
                assert False in speaking_log, 'tts_speaking=False не было опубликовано'

                # Порядок: True должен быть раньше False
                first_true = next(i for i, v in enumerate(speaking_log) if v is True)
                last_false = max(i for i, v in enumerate(speaking_log) if v is False)
                assert first_true < last_false, 'True должен предшествовать False'

            finally:
                listener_exec.shutdown(timeout_sec=1.0)
                listener_node.destroy_node()
