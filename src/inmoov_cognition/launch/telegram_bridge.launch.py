#!/usr/bin/env python3
"""
telegram_bridge.launch.py — launches the Telegram ↔ ROS2 bridge.

Requires an environment variable:
  TELEGRAM_BOT_TOKEN=<token>

Usage:
  ros2 launch inmoov_cognition telegram_bridge.launch.py allowed_chat_id:=123456789

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
        DeclareLaunchArgument(
            'allowed_chat_id',
            default_value=os.environ.get('TELEGRAM_ALLOWED_CHAT_ID', '0'),
            description='Telegram chat_id allowed to send commands'),
        DeclareLaunchArgument(
            'llm_timeout_sec',
            default_value='35.0',
            description='Timeout waiting for a reply from llm_node (sec)'),
        DeclareLaunchArgument(
            'memory_db_path',
            default_value='/home/artur/inmoov_memory.db',
            description='Path to the SQLite DB for looking up telegram_id'),
        DeclareLaunchArgument(
            'cam_device',
            default_value=(
                '/dev/v4l/by-path/'
                'pci-0000:c6:00.3-usb-0:1.1:1.0-video-index0'),
            description='Left camera device for /photo'),
    ]

    telegram_bridge = Node(
        package='inmoov_cognition',
        executable='telegram_bridge_node',
        name='telegram_bridge_node',
        output='screen',
        parameters=[{
            'allowed_chat_id': LaunchConfiguration('allowed_chat_id'),
            'llm_timeout_sec': LaunchConfiguration('llm_timeout_sec'),
            'memory_db_path':  LaunchConfiguration('memory_db_path'),
            'cam_device':      LaunchConfiguration('cam_device'),
        }],
    )

    return LaunchDescription(args + [telegram_bridge])
