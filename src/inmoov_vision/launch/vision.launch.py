#!/usr/bin/env python3
"""
vision.launch.py — launches the vision pipeline for InMoov.

Data flow:
  PIR (/pir_state) -> inmoov_cognition -> /face_detection/enable
  /face_detection/enable -> face_detection_node_{left,right}
                         -> face_tracker_node_{left,right}
  inmoov_cognition  -> /head_tracker/enable -> vision_head_tracker_node

  face_capture_node           -> /camera/eye_{left,right}/compressed
  face_detection_node_left    -> /face/detections/left
  face_detection_node_right   -> /face/detections/right  (buffalo_l)
  face_tracker_node_left      -> /face/tracks/left
  face_tracker_node_right     -> /face/tracks/right
  face_recognition_node       -> /face/identity  (from /face/tracks/left)
  emotion_recognition_node    -> /face/emotion   (from /face/tracks/left)
  vision_head_tracker_node    -> /joint_command, /face_command

  identity_manager_node lives in the inmoov_cognition package.

Usage:
  ros2 launch inmoov_vision vision.launch.py
  ros2 launch inmoov_vision vision.launch.py cam_left:=... cam_right:=...

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('cam_left',
            default_value='/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1.1:1.0-video-index0'),
        DeclareLaunchArgument('cam_right',
            default_value='/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1.2:1.0-video-index0'),
        DeclareLaunchArgument('fps',                  default_value='15'),
        DeclareLaunchArgument('detection_hz',         default_value='5.0'),
        DeclareLaunchArgument('det_thresh',           default_value='0.5'),
        DeclareLaunchArgument('analysis_hz',          default_value='2.0'),
        DeclareLaunchArgument('gain_head',            default_value='0.3'),
        DeclareLaunchArgument('gain_eye',             default_value='0.6'),
        DeclareLaunchArgument('rest_rothead',         default_value='90.0'),
        DeclareLaunchArgument('rest_neck',            default_value='40.0'),
        # OAK-D Lite (depthai v3 — model from Luxonis HubAI)
        DeclareLaunchArgument('oak_model',            default_value='yolov6-nano'),
        DeclareLaunchArgument('oak_conf_threshold',   default_value='0.5'),
        # Scene manager — static name of the robot's current location (the robot
        # is on wheels and is moved by hand; can be changed on the fly via `ros2 param set`)
        DeclareLaunchArgument('scene_location',       default_value=''),
    ]

    # 1. Camera — always running (lightweight process)
    face_capture = Node(
        package='inmoov_vision',
        executable='face_capture_node',
        name='face_capture_node',
        output='screen',
        parameters=[{
            'cam_left':  LaunchConfiguration('cam_left'),
            'cam_right': LaunchConfiguration('cam_right'),
            'fps':          LaunchConfiguration('fps'),
            'width':        640,
            'height':       480,
            'jpeg_quality': 85,
        }],
    )

    # 2a. Face detection — left eye (buffalo_l: bbox + embedding)
    face_detection_left = Node(
        package='inmoov_vision',
        executable='face_detection_node',
        name='face_detection_node_left',
        output='screen',
        parameters=[{
            'camera_side':  'left',
            'detection_hz': LaunchConfiguration('detection_hz'),
            'det_size':     640,
            'det_thresh':   LaunchConfiguration('det_thresh'),
            'model_name':   'buffalo_l',
        }],
    )

    # 2b. Face detection — right eye (buffalo_l: bbox + embedding for fallback/redundancy)
    # A true on-demand fallback: insightface on the right eye is switched on only
    # when the left (primary) really has not published for >1.5 s, rather than
    # running constantly in parallel with the left — saves CPU.
    face_detection_right = Node(
        package='inmoov_vision',
        executable='face_detection_node',
        name='face_detection_node_right',
        output='screen',
        parameters=[{
            'camera_side':  'right',
            'detection_hz': LaunchConfiguration('detection_hz'),
            'det_size':     640,
            'det_thresh':   LaunchConfiguration('det_thresh'),
            'model_name':   'buffalo_l',
            'fallback_for':        '/face/detections/left',
            'primary_timeout_sec': 1.5,
        }],
    )

    # 3a. Left-eye tracker
    face_tracker_left = Node(
        package='inmoov_vision',
        executable='face_tracker_node',
        name='face_tracker_node_left',
        output='screen',
        parameters=[{
            'camera_side':    'left',
            'iou_threshold':  0.20,
            'max_lost_frames': 20,
            'max_tracks':      4,
        }],
    )

    # 3b. Right-eye tracker
    face_tracker_right = Node(
        package='inmoov_vision',
        executable='face_tracker_node',
        name='face_tracker_node_right',
        output='screen',
        parameters=[{
            'camera_side':    'right',
            'iou_threshold':  0.20,
            'max_lost_frames': 20,
            'max_tracks':      4,
        }],
    )

    # 4. Recognition — requires the /memory/query service
    face_recognition = Node(
        package='inmoov_vision',
        executable='face_recognition_node',
        name='face_recognition_node',
        output='screen',
        parameters=[{
            'recognition_cooldown_sec': 3.0,
        }],
    )

    # 5. Face emotion
    emotion_recognition = Node(
        package='inmoov_vision',
        executable='emotion_recognition_node',
        name='emotion_recognition_node',
        output='screen',
        parameters=[{
            'analysis_hz':    LaunchConfiguration('analysis_hz'),
            'min_face_size':  48,
        }],
    )

    # 9. OAK-D Lite — YOLO object detection + depth (Myriad X VPU)
    oak = Node(
        package='inmoov_vision',
        executable='oak_node',
        name='oak_node',
        output='screen',
        parameters=[{
            'model_name':     LaunchConfiguration('oak_model'),
            'conf_threshold': LaunchConfiguration('oak_conf_threshold'),
        }],
    )

    # 10. Human detection — body presence via OAK-D Lite YOLO detections
    human_detection = Node(
        package='inmoov_vision',
        executable='human_detection_node',
        name='human_detection_node',
        output='screen',
        parameters=[{
            'max_distance_m':   4.0,
            'min_confidence':   0.45,
            'lost_timeout_sec': 4.0,
            'publish_rate_hz':  5.0,
        }],
    )

    # 11. Scene manager — scene summary (objects + people) from /objects/detections
    scene_manager = Node(
        package='inmoov_vision',
        executable='scene_manager_node',
        name='scene_manager_node',
        output='screen',
        parameters=[{
            'min_confidence':       0.5,
            'max_distance_m':       4.0,
            'object_ttl_sec':       8.0,
            'smoothing_window_sec': 1.0,
            'publish_rate_hz':      1.0,
            'top_k_objects':        8,
            'location_name':        LaunchConfiguration('scene_location'),
        }],
    )

    # 7. Photo gallery — automatic face capture for gallery-based recognition
    face_gallery = Node(
        package='inmoov_vision',
        executable='face_gallery_node',
        name='face_gallery_node',
        output='screen',
        parameters=[{
            'gallery_dir':           '/home/artur/inmoov_faces',
            'enroll_interval_sec':   1.0,
            'interact_interval_sec': 15.0,
            'min_det_score':         0.75,
            'min_face_px':           60,
        }],
    )

    # 8. Head tracker — turns the head/eyes towards the face (dual-eye leader/follower)
    head_tracker = Node(
        package='inmoov_vision',
        executable='vision_head_tracker_node',
        name='vision_head_tracker_node',
        output='screen',
        parameters=[{
            'image_width':         640,
            'image_height':        480,
            'fov_h_deg':           60.0,
            'fov_v_deg':           45.0,
            'gain_head':           LaunchConfiguration('gain_head'),
            'gain_eye':            LaunchConfiguration('gain_eye'),
            'rest_rothead':        LaunchConfiguration('rest_rothead'),
            'rest_neck':           LaunchConfiguration('rest_neck'),
            'rest_eye_lr':         90.0,
            'rest_eye_ud':        100.0,
            'dead_zone_px':        20,
            'head_dead_zone_px':   80,
            'eye_limit_deg':       8.0,
            'return_timeout_sec':  10.0,  # synchronized with inmoov_bringup/inmoov.launch.py (2026-09-01)
            'track_hz':            10.0,
            'max_step_deg':        2.0,
            'bbox_ema_alpha':      0.4,
            'max_stale_ticks':     5,
        }],
    )

    return LaunchDescription(args + [
        face_capture,
        face_detection_left,
        face_detection_right,
        face_tracker_left,
        face_tracker_right,
        face_recognition,
        face_gallery,
        emotion_recognition,
        head_tracker,
        oak,
        human_detection,
        scene_manager,
    ])
