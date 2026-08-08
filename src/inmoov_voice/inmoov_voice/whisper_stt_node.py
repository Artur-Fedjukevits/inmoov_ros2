import io
import struct
import threading
import time
import wave

import numpy as np
import requests
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from std_msgs.msg import Float32MultiArray, String


class WhisperSTTNode(LifecycleNode):
    def __init__(self):
        super().__init__('whisper_stt_node')
        self.subscription  = None
        self.text_pub      = None
        self._transcribing = False
        self._lock         = threading.Lock()

    # ── Lifecycle: Phase 2 ─────────────────────────────────────────────────

    def _dp(self, name, default=None):
        """Безопасный declare_parameter: игнорирует повторное объявление при re-configure."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def on_configure(self, state):
        self._dp('server_url',          'http://127.0.0.1:8765')
        self._dp('language',            'ru')
        self._dp('min_confidence',      0.6)
        self._dp('no_speech_threshold', 0.6)
        self._dp('logprob_threshold',   -1.5)
        self._dp('request_timeout_sec', 30.0)
        self._dp('min_audio_sec',       0.8)

        self._url                = self.get_parameter('server_url').value.rstrip('/')
        self.language            = self.get_parameter('language').value
        self.min_confidence      = self.get_parameter('min_confidence').value
        self.no_speech_threshold = self.get_parameter('no_speech_threshold').value
        self.logprob_threshold   = self.get_parameter('logprob_threshold').value
        self.request_timeout     = self.get_parameter('request_timeout_sec').value
        self.min_audio_sec       = self.get_parameter('min_audio_sec').value
        self._transcribing       = False

        self._hallucination_markers = {
            'субтитры создавал', 'субтитры делал', 'субтитры сделаны',
            'редактор субтитров', 'субтитры:', 'продолжение следует',
            'dimatorzok', 'амара.орг', 'amara.org',
            'подписывайтесь на канал', 'www.', 'переведено и озвучено',
        }

        self.subscription = self.create_subscription(
            Float32MultiArray, 'audio_to_whisper', self.audio_callback, 10)
        self.text_pub = self.create_lifecycle_publisher(String, 'voice_command', 10)

        # Проверяем сервер non-blocking (warn если недоступен, продолжаем)
        self._check_server_once()
        return TransitionCallbackReturn.SUCCESS

    def _check_server_once(self):
        try:
            r = requests.get(f'{self._url}/health', timeout=2.0)
            if r.status_code in (200, 404):
                self.get_logger().info(f'Whisper сервер доступен: {self._url}')
                return
        except Exception:
            pass
        self.get_logger().warn(f'Whisper сервер недоступен: {self._url} — продолжаю')

    def on_activate(self, state):
        self.text_pub.on_activate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.text_pub.on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        return TransitionCallbackReturn.SUCCESS

    # ── Ожидание готовности HTTP сервера ──────────────────────────────────────
    def _wait_for_server(self):
        health_url = f'{self._url}/health'
        self.get_logger().info(f'Ожидаю whisper.cpp сервер на {self._url}...')
        for attempt in range(30):
            try:
                r = requests.get(health_url, timeout=2.0)
                if r.status_code in (200, 404):  # 404 — сервер жив, нет /health
                    self.get_logger().info('Whisper сервер доступен.')
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(2.0)
        self.get_logger().warn(
            'Whisper сервер не ответил за 60с — продолжаю, попробуем при первом аудио.')

    # ── Приём аудио ───────────────────────────────────────────────────────────
    def audio_callback(self, msg: Float32MultiArray):
        with self._lock:
            if self._transcribing:
                self.get_logger().warn(
                    'Транскрипция уже идёт, новое аудио пропущено.')
                return
            self._transcribing = True

        thread = threading.Thread(
            target=self._transcribe, args=(msg,), daemon=True)
        thread.start()

    # ── Транскрипция ──────────────────────────────────────────────────────────
    def _transcribe(self, msg: Float32MultiArray):
        try:
            audio = np.array(msg.data, dtype=np.float32)

            sample_rate = 16000
            if msg.layout.dim and msg.layout.dim[0].label == 'sample_rate':
                sample_rate = msg.layout.dim[0].stride

            duration = len(audio) / sample_rate

            if duration < self.min_audio_sec:
                self.get_logger().warn(
                    f'Аудио слишком короткое ({duration:.2f}с < {self.min_audio_sec}с) — пропускаю.')
                self._publish('')
                return

            self.get_logger().info(f'Транскрибирую {duration:.1f}с аудио...')
            t0 = time.perf_counter()

            wav_bytes = self._to_wav(audio, sample_rate)

            response = requests.post(
                f'{self._url}/inference',
                files={'file': ('audio.wav', wav_bytes, 'audio/wav')},
                data={
                    'language': self.language,
                    'response_format': 'verbose_json',
                    'temperature': '0.0',
                },
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            result = response.json()

            full_text, accepted = self._filter_segments(result)

            if not full_text:
                self.get_logger().warn(
                    f'Речь не распознана. '
                    f'Сегментов: {len(result.get("segments", []))}, '
                    f'принято: {accepted}')
                self._publish('')
                return

            text_lower = full_text.lower()
            if any(marker in text_lower for marker in self._hallucination_markers):
                self.get_logger().warn(
                    f'Галлюцинация Whisper отброшена: "{full_text}"')
                self._publish('')
                return

            self.get_logger().info(
                f'Распознано за {time.perf_counter() - t0:.1f}с: "{full_text}"')
            self._publish(full_text)

        except requests.exceptions.Timeout:
            self.get_logger().error(
                f'Таймаут запроса к whisper серверу ({self.request_timeout}с)')
            self._publish('')
        except requests.exceptions.ConnectionError as e:
            self.get_logger().error(f'Whisper сервер недоступен: {e}')
            self._publish('')
        except Exception as e:
            self.get_logger().error(f'Ошибка транскрипции: {e}')
            self._publish('')
        finally:
            with self._lock:
                self._transcribing = False

    # ── Фильтрация сегментов (как в faster-whisper ноде) ─────────────────────
    def _filter_segments(self, result: dict) -> tuple[str, int]:
        segments = result.get('segments', [])

        # Если сервер вернул просто текст без сегментов — используем как есть
        if not segments:
            text = result.get('text', '').strip()
            return text, (1 if text else 0)

        accepted = []
        for seg in segments:
            no_speech = seg.get('no_speech_prob', 0.0)
            avg_logprob = seg.get('avg_logprob', 0.0)
            text = seg.get('text', '').strip()

            if no_speech > self.no_speech_threshold:
                self.get_logger().warn(
                    f'Сегмент отброшен (no_speech_prob={no_speech:.2f}): "{text}"')
                continue
            if avg_logprob < self.logprob_threshold:
                self.get_logger().warn(
                    f'Сегмент отброшен (avg_logprob={avg_logprob:.2f}): "{text}"')
                continue
            accepted.append(text)

        return ' '.join(accepted).strip(), len(accepted)

    # ── Float32 → WAV (16-bit PCM, mono, in-memory) ───────────────────────────
    @staticmethod
    def _to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
        audio_int16 = (
            np.clip(audio, -1.0, 1.0) * 32767
        ).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return buf.getvalue()

    def _publish(self, text: str):
        msg = String()
        msg.data = text
        self.text_pub.publish(msg)




def main():
    rclpy.init()
    node = WhisperSTTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
