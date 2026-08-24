#!/usr/bin/env python3
"""
inmoov_full.launch.py — полный стек InMoov.

Включает:
  inmoov_control   — Arduino (серво, PIR, ультразвук) + face_expressions_node
  inmoov_voice     — wake word, VAD, STT, TTS
  behavior_manager — поведенческое дерево + llm_node (генератор намерений)
  inmoov_memory    — социальная память (SQLite, /memory/query)
  inmoov_vision    — камера, детекция/трекинг/распознавание лиц, head tracker

Использование:
  ros2 launch behavior_manager_node inmoov_full.launch.py
  ros2 launch behavior_manager_node inmoov_full.launch.py tavily_api_key:=tvly-...
  ros2 launch behavior_manager_node inmoov_full.launch.py vision:=false
  ros2 launch behavior_manager_node inmoov_full.launch.py port_right:=/dev/ttyUSB0
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ── Аргументы ─────────────────────────────────────────────────────────────

    args = [
        # Серверы
        # llm_url — OpenAI-совместимый chat.completions endpoint (сейчас vLLM).
        DeclareLaunchArgument('llm_url',
            default_value='http://192.168.10.118:18020/v1/chat/completions'),
        DeclareLaunchArgument('llm_fallback_url',
            default_value='http://localhost:11434/v1/chat/completions'),
        DeclareLaunchArgument('llm_bearer_token',
            default_value=os.environ.get('VLLM_BEARER_TOKEN', '')),
        DeclareLaunchArgument('tts_server_url',
            default_value='http://192.168.10.118:8000'),
        DeclareLaunchArgument('tts_fallback_url',
            default_value='http://localhost:8000'),
        DeclareLaunchArgument('openhab_url',
            default_value='http://192.168.10.118:8080'),

        # LLM
        DeclareLaunchArgument('llm_model',
            default_value='qwen3.8-27b'),

        # Wake word
        DeclareLaunchArgument('wakeword_model',
            default_value='/home/artur/openWakeWord/my_custom_model/ey_lyonya.onnx'),
        DeclareLaunchArgument('wakeword_threshold',
            default_value='0.2'),

        # Аудио
        DeclareLaunchArgument('audio_device_name',
            default_value='pulse'),

        # Tavily
        DeclareLaunchArgument('tavily_api_key',
            default_value=os.environ.get('TAVILY_API_KEY', '')),

        # Arduino serial ports
        DeclareLaunchArgument('port_right',
            default_value='/dev/serial/by-path/pci-0000:c6:00.3-usb-0:5:1.0-port0',
            description='Serial port for Right Arduino Mega'),
        DeclareLaunchArgument('port_left',
            default_value='/dev/serial/by-path/pci-0000:c6:00.3-usb-0:1.3:1.0-port0',
            description='Serial port for Left Arduino Mega'),

        # Memory
        DeclareLaunchArgument('memory_db_path',
            default_value='/home/artur/inmoov_memory.db',
            description='Path to SQLite face memory database'),

        # Vision — можно отключить: vision:=false
        DeclareLaunchArgument('vision',
            default_value='true',
            description='Enable vision pipeline (face detection, tracking, etc.)'),
        DeclareLaunchArgument('cam_left',
            default_value='/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1.1:1.0-video-index0'),
        DeclareLaunchArgument('cam_right',
            default_value='/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1.2:1.0-video-index0'),
        DeclareLaunchArgument('greet_cooldown_sec', default_value='120.0'),

        # Telegram Bridge — включить: telegram:=true
        DeclareLaunchArgument('telegram',
            default_value='false',
            description='Enable Telegram bridge (requires TELEGRAM_BOT_TOKEN env var)'),
        DeclareLaunchArgument('allowed_chat_id',
            default_value=os.environ.get('TELEGRAM_ALLOWED_CHAT_ID', '0'),
            description='Telegram chat_id allowed to send commands'),
    ]

    # ── inmoov_control ────────────────────────────────────────────────────────

    control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('inmoov_control'), 'launch', 'inmoov_control.launch.py'
            ])
        ]),
        launch_arguments={
            'port_right': LaunchConfiguration('port_right'),
            'port_left':  LaunchConfiguration('port_left'),
        }.items(),
    )

    # ── inmoov_voice ──────────────────────────────────────────────────────────

    voice = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('inmoov_voice'), 'launch', 'voice.launch.py'
            ])
        ]),
        launch_arguments={
            'llm_url':             LaunchConfiguration('llm_url'),
            'llm_fallback_url':    LaunchConfiguration('llm_fallback_url'),
            'llm_bearer_token':    LaunchConfiguration('llm_bearer_token'),
            'tts_server_url':      LaunchConfiguration('tts_server_url'),
            'tts_fallback_url':    LaunchConfiguration('tts_fallback_url'),
            'openhab_url':         LaunchConfiguration('openhab_url'),
            'llm_model':           LaunchConfiguration('llm_model'),
            'wakeword_model':      LaunchConfiguration('wakeword_model'),
            'wakeword_threshold':  LaunchConfiguration('wakeword_threshold'),
            'audio_device_name':   LaunchConfiguration('audio_device_name'),
        }.items(),
    )

    # ── behavior_manager_node + identity_manager_node ────────────────────────

    behavior = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('inmoov_cognition'), 'launch',
                'behavior_manager.launch.py'
            ])
        ]),
        launch_arguments={
            'tavily_api_key':      LaunchConfiguration('tavily_api_key'),
            'greet_cooldown_sec':  LaunchConfiguration('greet_cooldown_sec'),
            'llm_url':             LaunchConfiguration('llm_url'),
            'llm_fallback_url':    LaunchConfiguration('llm_fallback_url'),
            'llm_bearer_token':    LaunchConfiguration('llm_bearer_token'),
            'name_extract_model':  LaunchConfiguration('llm_model'),
            'openhab_url':         LaunchConfiguration('openhab_url'),
        }.items(),
    )

    # ── inmoov_memory ─────────────────────────────────────────────────────────

    memory = Node(
        package='inmoov_memory',
        executable='memory_node',
        name='memory_node',
        output='screen',
        parameters=[{
            'db_path':              LaunchConfiguration('memory_db_path'),
            'similarity_threshold': 0.55,
            'llm_url':              LaunchConfiguration('llm_url'),
            'llm_model':            LaunchConfiguration('llm_model'),
            'bearer_token':         LaunchConfiguration('llm_bearer_token'),
        }],
    )

    # ── inmoov_vision (опционально) ───────────────────────────────────────────

    vision = GroupAction(
        condition=IfCondition(LaunchConfiguration('vision')),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([
                        FindPackageShare('inmoov_vision'), 'launch', 'vision.launch.py'
                    ])
                ]),
                launch_arguments={
                    'cam_left':  LaunchConfiguration('cam_left'),
                    'cam_right': LaunchConfiguration('cam_right'),
                }.items(),
            ),
        ],
    )

    # ── Telegram Bridge (опционально) ────────────────────────────────────────

    telegram = GroupAction(
        condition=IfCondition(LaunchConfiguration('telegram')),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([
                        FindPackageShare('inmoov_cognition'), 'launch',
                        'telegram_bridge.launch.py'
                    ])
                ]),
                launch_arguments={
                    'allowed_chat_id': LaunchConfiguration('allowed_chat_id'),
                    'memory_db_path':  LaunchConfiguration('memory_db_path'),
                }.items(),
            ),
        ],
    )

    return LaunchDescription(args + [
        memory,    # memory первым — /memory/query нужен face_recognition
        control,
        voice,
        behavior,
        vision,
        telegram,
    ])
