#!/usr/bin/env python3
"""
behavior_manager.launch.py — запуск behavior_manager_node + identity_manager_node.

identity_manager — центральный поведенческий блок: агрегирует лицо, голос
и память, публикует /social_context для Behavior Tree.

Использование:
  ros2 launch inmoov_cognition behavior_manager.launch.py
  ros2 launch inmoov_cognition behavior_manager.launch.py tavily_api_key:=tvly-...
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('tavily_api_key',  default_value=os.environ.get('TAVILY_API_KEY', ''),
                              description='Tavily Search API key (для web_search)'),
        DeclareLaunchArgument('bt_tick_rate_hz', default_value='10.0',
                              description='Частота тика Behavior Tree'),
        DeclareLaunchArgument('greet_cooldown_sec',  default_value='120.0'),
        DeclareLaunchArgument('ollama_url',          default_value='http://192.168.10.118:11434/api/chat'),
        DeclareLaunchArgument('ollama_fallback_url', default_value='http://localhost:11434/api/chat'),
        DeclareLaunchArgument('name_extract_model',  default_value='qwen3.6:27b'),
        DeclareLaunchArgument('openhab_url',         default_value='http://192.168.10.118:8080'),
    ]

    behavior_manager = Node(
        package='inmoov_cognition',
        executable='behavior_manager_node',
        name='behavior_manager_node',
        output='screen',
        parameters=[{
            'tavily_api_key':  LaunchConfiguration('tavily_api_key'),
            'bt_tick_rate_hz': LaunchConfiguration('bt_tick_rate_hz'),
        }],
    )

    identity_manager = Node(
        package='inmoov_cognition',
        executable='identity_manager_node',
        name='identity_manager_node',
        output='screen',
        parameters=[{
            'greet_cooldown_sec':     LaunchConfiguration('greet_cooldown_sec'),
            'no_face_timeout_sec':    15.0,
            'no_human_timeout_sec':   30.0,
            'introduce_cooldown_sec': 120.0,
            'emotion_react_thresh':   0.70,
            'ollama_url':             LaunchConfiguration('ollama_url'),
            'ollama_fallback_url':    LaunchConfiguration('ollama_fallback_url'),
            'name_extract_model':     LaunchConfiguration('name_extract_model'),
        }],
    )

    openhab_bridge = Node(
        package='inmoov_cognition',
        executable='openhab_bridge_node',
        name='openhab_bridge_node',
        output='screen',
        parameters=[{
            'openhab_url':       LaunchConfiguration('openhab_url'),
            'items_tag':         'ChatGPT',
            'reconnect_sec':     15.0,
            'schema_repeat_sec': 5.0,
        }],
    )

    return LaunchDescription(args + [behavior_manager, identity_manager, openhab_bridge])
