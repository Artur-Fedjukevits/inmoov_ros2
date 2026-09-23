#!/usr/bin/env python3
"""
voice.launch.py — launches the whole InMoov voice pipeline.

Startup order:
  audio_source_node  → raw_audio
  openwakeword_node  → wake_detected
  voice_detector_node → audio_to_whisper
  parakeet_stt_node  → voice_command
  tts_node           (action server /speak)
  sound_localization_node → sound_direction (separate stereo mic pair)
  llm_node           (intent generator — from the behavior_manager_node package)

Usage:
  ros2 launch inmoov_voice voice.launch.py
  ros2 launch inmoov_voice voice.launch.py llm_url:=http://localhost:18020/v1/chat/completions
  ros2 launch inmoov_voice voice.launch.py tavily_api_key:=tvly-...

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# PulseAudio/PipeWire session for the audio nodes
_uid = os.getuid()
_AUDIO_ENV = {
    'PULSE_SERVER': f'unix:/run/user/{_uid}/pulse/native',
    'DBUS_SESSION_BUS_ADDRESS': os.environ.get('DBUS_SESSION_BUS_ADDRESS', f'unix:path=/run/user/{_uid}/bus'),
}


def generate_launch_description():

    # ── Arguments (overridable from the command line) ─────────────────────────
    args = [
        # Servers
        # llm_url — OpenAI-compatible chat.completions endpoint (currently vLLM).
        DeclareLaunchArgument('llm_url',
            default_value='http://192.168.10.118:18020/v1/chat/completions'),
        DeclareLaunchArgument('llm_fallback_url',
            default_value=''),
        DeclareLaunchArgument('llm_bearer_token',
            default_value=os.environ.get('VLLM_BEARER_TOKEN', '')),
        DeclareLaunchArgument('tts_server_url',      default_value='http://192.168.10.118:8000'),
        DeclareLaunchArgument('tts_fallback_url',    default_value=''),
        DeclareLaunchArgument('openhab_url',         default_value='http://192.168.10.118:8080'),

        # LLM model
        DeclareLaunchArgument('llm_model',           default_value='qwen3.8-27b'),
        DeclareLaunchArgument('llm_temperature',     default_value='0.1'),
        DeclareLaunchArgument('llm_max_tokens',      default_value='512'),

        # Wake word
        DeclareLaunchArgument('wakeword_model',      default_value='/home/artur/openWakeWord/my_custom_model/ey_lyonya.onnx'),
        DeclareLaunchArgument('wakeword_threshold',  default_value='0.2'),

        # Audio
        DeclareLaunchArgument('audio_device_index',  default_value='-1'),   # -1 = system default
        DeclareLaunchArgument('audio_device_name',   default_value='pulse'),
        DeclareLaunchArgument('output_device_name',  default_value=''),
        DeclareLaunchArgument('sample_rate',         default_value='16000'),

        # VAD
        DeclareLaunchArgument('vad_threshold',          default_value='0.4'),
        DeclareLaunchArgument('silence_duration_sec',   default_value='2.5'),
        DeclareLaunchArgument('pipeline_timeout_sec',   default_value='45.0'),

        # Speaker Verification
        DeclareLaunchArgument('speaker_verification',   default_value='true'),
        DeclareLaunchArgument('sv_threshold',           default_value='0.55'),
        DeclareLaunchArgument('sv_segment_sec',         default_value='1.0'),
    ]

    # ── Nodes ─────────────────────────────────────────────────────────────────

    # 1. Audio source — started first
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

    # 2. Wake word — subscribes to raw_audio
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

    # 3. Voice detector (VAD) — subscribes to raw_audio + wake_detected
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

    # 3b. Sound localization — separate stereo mic pair (raw ALSA), TDOA → sound_direction
    sound_localization = Node(
        package='inmoov_voice',
        executable='sound_localization_node',
        name='sound_localization_node',
        output='screen',
    )

    # 4. Voice emotion — reads audio_to_whisper (after VAD), loads SpeechBrain wav2vec2
    voice_emotion = TimerAction(
        period=5.0,   # wait so that SpeechBrain/wav2vec2 loads without racing the VAD
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

    # 5. TTS — Action Server, started before the LLM
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

    # 5. STT — Parakeet-TDT-0.6b-v3 (CPU, ONNX), replaced whisper.cpp on 2026-08-27
    parakeet = TimerAction(
        period=2.0,
        actions=[Node(
            package='inmoov_voice',
            executable='parakeet_stt_node',
            name='parakeet_stt_node',
            output='screen',

            parameters=[{
                'language': 'ru',
            }],
        )],
    )

    # 6. LLM — started last (depends on the TTS action server)
    llm = TimerAction(
        period=3.0,   # wait for the TTS action server to come up
        actions=[Node(
            package='inmoov_cognition',
            executable='llm_node',
            name='llm_node',
            output='screen',
    
            parameters=[{
                'llm_url':             LaunchConfiguration('llm_url'),
                'llm_fallback_url':    LaunchConfiguration('llm_fallback_url'),
                'bearer_token':        LaunchConfiguration('llm_bearer_token'),
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
        sound_localization,
        voice_emotion,
        tts,
        parakeet,
        llm,
    ])
