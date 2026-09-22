#!/usr/bin/env python3
"""
behavior_manager.launch.py — launches behavior_manager_node + identity_manager_node.

identity_manager — the central behavioral hub: aggregates face, voice
and memory, publishes /social_context for the Behavior Tree.

Usage:
  ros2 launch inmoov_cognition behavior_manager.launch.py
  ros2 launch inmoov_cognition behavior_manager.launch.py tavily_api_key:=tvly-...

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('tavily_api_key',  default_value=os.environ.get('TAVILY_API_KEY', ''),
                              description='Tavily Search API key (for web_search)'),
        DeclareLaunchArgument('bt_tick_rate_hz', default_value='10.0',
                              description='Behavior Tree tick rate'),
        DeclareLaunchArgument('greet_cooldown_sec',  default_value='120.0'),
        # llm_url — OpenAI-compatible chat.completions endpoint (currently vLLM).
        DeclareLaunchArgument('llm_url',
            default_value='http://192.168.10.118:18020/v1/chat/completions'),
        DeclareLaunchArgument('llm_fallback_url',
            default_value='http://localhost:11434/v1/chat/completions'),
        DeclareLaunchArgument('llm_bearer_token',
            default_value=os.environ.get('VLLM_BEARER_TOKEN', '')),
        DeclareLaunchArgument('name_extract_model',  default_value='qwen3.8-27b'),
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
            'llm_url':                LaunchConfiguration('llm_url'),
            'llm_fallback_url':       LaunchConfiguration('llm_fallback_url'),
            'bearer_token':           LaunchConfiguration('llm_bearer_token'),
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
