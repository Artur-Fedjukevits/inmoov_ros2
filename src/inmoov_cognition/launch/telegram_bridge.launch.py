#!/usr/bin/env python3
"""
telegram_bridge.launch.py — запуск Telegram ↔ ROS2 моста.

Требует переменную окружения:
  TELEGRAM_BOT_TOKEN=<токен>

Использование:
  ros2 launch inmoov_cognition telegram_bridge.launch.py allowed_chat_id:=123456789
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
            description='Telegram chat_id которому разрешены команды'),
        DeclareLaunchArgument(
            'llm_timeout_sec',
            default_value='35.0',
            description='Таймаут ожидания ответа от llm_node (сек)'),
        DeclareLaunchArgument(
            'memory_db_path',
            default_value='/home/artur/inmoov_memory.db',
            description='Путь к SQLite БД для поиска telegram_id'),
        DeclareLaunchArgument(
            'cam_device',
            default_value=(
                '/dev/v4l/by-path/'
                'pci-0000:c6:00.3-usb-0:1.1:1.0-video-index0'),
            description='Устройство левой камеры для /photo'),
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
