#!/usr/bin/env python3
"""RViz2 only — no robot_state_publisher, no joint_state_publisher_gui.

Use when something else already publishes /robot_description and
/joint_states.

  ros2 launch inmoov_description rviz_standalone.launch.py

Adapted from Sentience-Robotics/inmoov_urdf's launch/rviz_standalone.launch.py
(GPL-3.0): https://github.com/Sentience-Robotics/inmoov_urdf

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_rviz = os.path.join(
        get_package_share_directory("inmoov_description"),
        "config",
        "inmoov_rviz.rviz",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz,
                description="Display config .rviz path",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", LaunchConfiguration("rviz_config")],
            ),
        ]
    )
