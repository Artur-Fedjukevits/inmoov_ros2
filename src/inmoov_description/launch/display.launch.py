#!/usr/bin/env python3
"""RViz2 display of the InMoov URDF (description/inmoov_i2.urdf.xacro).

  ros2 launch inmoov_description display.launch.py

Adapted from Sentience-Robotics/inmoov_urdf's launch/joint_preview.launch.py
(GPL-3.0): https://github.com/Sentience-Robotics/inmoov_urdf

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("inmoov_description")

    default_model = os.path.join(share, "description", "inmoov_i2.urdf.xacro")
    default_rviz = os.path.join(share, "config", "inmoov_rviz.rviz")

    model_arg = DeclareLaunchArgument(
        "model",
        default_value=default_model,
        description="Path to the URDF to load",
    )
    use_gui_arg = DeclareLaunchArgument(
        "use_gui",
        default_value="true",
        description="Spawn joint_state_publisher_gui sliders",
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Launch RViz2",
    )
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=default_rviz,
        description="RViz display config",
    )

    # value_type=str forces launch to treat xacro's stdout as a plain string
    # instead of trying to YAML-parse it (it starts with `<?xml ...>`). The
    # model is real xacro now (model_scale from properties.xacro), so xacro
    # needs to actually run, not just pass the file through.
    robot_description = ParameterValue(
        Command(["xacro ", LaunchConfiguration("model")]),
        value_type=str,
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_gui")),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        [
            model_arg,
            use_gui_arg,
            rviz_arg,
            rviz_config_arg,
            robot_state_publisher,
            joint_state_publisher_gui,
            rviz,
        ]
    )
