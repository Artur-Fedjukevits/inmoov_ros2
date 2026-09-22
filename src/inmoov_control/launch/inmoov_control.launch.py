#!/usr/bin/env python3
"""
inmoov_control.launch.py
========================

Standalone launch file for the inmoov_control package: starts both Arduino
bridge nodes (right / left), joint_state_publisher and face_expressions_node.

Launch arguments:
  port_right  Serial port of the Right Arduino Mega
              (default: /dev/serial/by-path/pci-0000:c6:00.3-usb-0:5:1.0-port0)
  port_left   Serial port of the Left Arduino Mega
              (default: /dev/serial/by-path/pci-0000:c6:00.3-usb-0:1.3:1.0-port0)

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    port_right = DeclareLaunchArgument(
        'port_right',
        default_value='/dev/serial/by-path/pci-0000:c6:00.3-usb-0:5:1.0-port0',
        description='Serial port for Right Arduino Mega'
    )
    port_left = DeclareLaunchArgument(
        'port_left',
        default_value='/dev/serial/by-path/pci-0000:c6:00.3-usb-0:1.3:1.0-port0',
        description='Serial port for Left Arduino Mega'
    )

    return LaunchDescription([
        port_right,
        port_left,
        Node(
            package='inmoov_control',
            executable='arduino_right_node',
            name='arduino_right',
            output='screen',
            arguments=['--port', LaunchConfiguration('port_right')],
        ),
        Node(
            package='inmoov_control',
            executable='arduino_left_node',
            name='arduino_left',
            output='screen',
            arguments=['--port', LaunchConfiguration('port_left')],
        ),
        Node(
            package='inmoov_control',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen',
        ),
        Node(
            package='inmoov_control',
            executable='face_expressions_node',
            name='face_expressions',
            output='screen',
        ),
    ])
