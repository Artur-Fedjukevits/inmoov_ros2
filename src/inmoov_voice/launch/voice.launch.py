#!/usr/bin/env python3
"""
voice.launch.py — запуск всего голосового пайплайна InMoov.

Порядок запуска:
  audio_source_node  → raw_audio
  openwakeword_node  → wake_detected
  voice_detector_node → audio_to_whisper
  whisper_stt_node   → voice_command
  tts_node           (action server /speak)
  llm_node           (intent generator — пакет behavior_manager_node)

Использование:
  ros2 launch inmoov_voice voice.launch.py
  ros2 launch inmoov_voice voice.launch.py ollama_url:=http://localhost:11434/api/chat
  ros2 launch inmoov_voice voice.launch.py tavily_api_key:=tvly-...
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# PulseAudio/PipeWire сессия для аудио нод
_uid = os.getuid()
_AUDIO_ENV = {
    'PULSE_SERVER': f'unix:/run/user/{_uid}/pulse/native',
    'DBUS_SESSION_BUS_ADDRESS': os.environ.get('DBUS_SESSION_BUS_ADDRESS', f'unix:path=/run/user/{_uid}/bus'),
}


def generate_launch_description():

    # ── Аргументы (переопределяются из командной строки) ──────────────────────
    args = [
        # Серверы
        DeclareLaunchArgument('ollama_url',          default_value='http://192.168.10.118:11434/api/chat'),
        DeclareLaunchArgument('ollama_fallback_url', default_value='http://localhost:11434/api/chat'),
        DeclareLaunchArgument('tts_server_url',      default_value='http://192.168.10.118:8000'),
        DeclareLaunchArgument('tts_fallback_url',    default_value='http://localhost:8000'),
        DeclareLaunchArgument('openhab_url',         default_value='http://192.168.10.118:8080'),

        # Модель LLM
        DeclareLaunchArgument('llm_model',           default_value='qwen3.6:27b'),
        DeclareLaunchArgument('llm_temperature',     default_value='0.1'),
        DeclareLaunchArgument('llm_max_tokens',      default_value='512'),

        # Whisper STT (whisper.cpp HTTP server)
        DeclareLaunchArgument('whisper_server_url',  default_value='http://127.0.0.1:8765'),
        DeclareLaunchArgument('whisper_language',    default_value='ru'),

        # Wake word
        DeclareLaunchArgument('wakeword_model',      default_value='/home/artur/openWakeWord/my_custom_model/ey_lyonya.onnx'),
        DeclareLaunchArgument('wakeword_threshold',  default_value='0.3'),

        # Аудио
        DeclareLaunchArgument('audio_device_index',  default_value='-1'),   # -1 = системное по умолчанию
        DeclareLaunchArgument('audio_device_name',   default_value='pulse'),
        DeclareLaunchArgument('output_device_name',  default_value=''),
        DeclareLaunchArgument('sample_rate',         default_value='16000'),

        # VAD
        DeclareLaunchArgument('vad_threshold',          default_value='0.5'),
        DeclareLaunchArgument('silence_duration_sec',   default_value='2.5'),
        DeclareLaunchArgument('pipeline_timeout_sec',   default_value='45.0'),

        # Speaker Verification
        DeclareLaunchArgument('speaker_verification',   default_value='true'),
        DeclareLaunchArgument('sv_threshold',           default_value='0.45'),
        DeclareLaunchArgument('sv_segment_sec',         default_value='3.0'),
    ]

    # ── Ноды ──────────────────────────────────────────────────────────────────

    # 1. Источник аудио — запускается первым
    audio_source = Node(
        package='inmoov_voice',
        executable='audio_source_node',
        name='audio_source_node',
        output='screen',
        additional_env=_AUDIO_ENV,
        parameters=[{
            'sample_rate':  LaunchConfiguration('sample_rate'),
            'chunk_size':   512,
            'device_index': LaunchConfiguration('audio_device_index'),
            'device_name':  LaunchConfiguration('audio_device_name'),
        }],
    )

    # 2. Wake word — подписывается на raw_audio
    wakeword = Node(
        package='inmoov_voice',
        executable='wakeword_node',
        name='wakeword_node',
        output='screen',

        parameters=[{
            'model_path':    LaunchConfiguration('wakeword_model'),
            'threshold':     LaunchConfiguration('wakeword_threshold'),
            'debounce_sec':  1.5,
        }],
    )

    # 3. Voice detector (VAD) — подписывается на raw_audio + wake_detected
    voice_detector = Node(
        package='inmoov_voice',
        executable='voice_detector_node',
        name='voice_detector_node',
        output='screen',

        parameters=[{
            'sample_rate':           LaunchConfiguration('sample_rate'),
            'vad_threshold':         LaunchConfiguration('vad_threshold'),
            'silence_duration_sec':  LaunchConfiguration('silence_duration_sec'),
            'pipeline_timeout_sec':  LaunchConfiguration('pipeline_timeout_sec'),
            'min_phrase_sec':        0.8,
            'max_phrase_sec':        20.0,
            'no_speech_timeout_sec': 8.0,
            'speaker_verification':  LaunchConfiguration('speaker_verification'),
            'sv_threshold':          LaunchConfiguration('sv_threshold'),
            'sv_segment_sec':        LaunchConfiguration('sv_segment_sec'),
        }],
    )

    # 4. Voice emotion — читает audio_to_whisper (после VAD), загружает SpeechBrain wav2vec2
    voice_emotion = TimerAction(
        period=5.0,   # ждём пока SpeechBrain/wav2vec2 загрузится без гонки с VAD
        actions=[Node(
            package='inmoov_voice',
            executable='voice_emotion_node',
            name='voice_emotion_node',
            output='screen',
            parameters=[{
                'min_confidence': 0.55,
                'savedir':        '/home/artur/.cache/speechbrain/voice_emotion',
            }],
        )],
    )

    # 5. TTS — Action Server, запускаем до LLM
    tts = Node(
        package='inmoov_voice',
        executable='tts_node',
        name='tts_node',
        output='screen',
        additional_env=_AUDIO_ENV,
        parameters=[{
            'tts_server_url':        LaunchConfiguration('tts_server_url'),
            'tts_fallback_url':      LaunchConfiguration('tts_fallback_url'),
            'output_device_name':    LaunchConfiguration('output_device_name'),
            'connect_timeout_sec':   5.0,
            'timeout_sec':           30.0,
            'chunk_size':            4096,
            'jaw_speed_deg_per_sec': 400.0,
        }],
    )

    # 5. Whisper STT — HTTP-клиент к whisper.cpp серверу (запускать сервер отдельно: ~/start_whisper_server.sh)
    whisper = TimerAction(
        period=2.0,
        actions=[Node(
            package='inmoov_voice',
            executable='whisper_stt_node',
            name='whisper_stt_node',
            output='screen',

            parameters=[{
                'server_url':          LaunchConfiguration('whisper_server_url'),
                'language':            LaunchConfiguration('whisper_language'),
                'min_confidence':      0.6,
                'no_speech_threshold': 0.9,
                'request_timeout_sec': 30.0,
            }],
        )],
    )

    # 6. LLM — запускаем последним (зависит от TTS action server)
    llm = TimerAction(
        period=3.0,   # ждём пока TTS action server поднимется
        actions=[Node(
            package='inmoov_cognition',
            executable='llm_node',
            name='llm_node',
            output='screen',
    
            parameters=[{
                'ollama_url':          LaunchConfiguration('ollama_url'),
                'ollama_fallback_url': LaunchConfiguration('ollama_fallback_url'),
                'model':               LaunchConfiguration('llm_model'),
                'temperature':         LaunchConfiguration('llm_temperature'),
                'max_tokens':          LaunchConfiguration('llm_max_tokens'),
                'connect_timeout_sec': 5.0,
                'timeout_sec':         60.0,
                'keep_history':        True,
                'history_max_turns':   5,
                'openhab_url':         LaunchConfiguration('openhab_url'),
            }],
        )],
    )

    return LaunchDescription(args + [
        audio_source,
        wakeword,
        voice_detector,
        voice_emotion,
        tts,
        whisper,
        llm,
    ])
