#!/usr/bin/env python3
"""
inmoov.launch.py — main launch file for the InMoov Robot (inmoov_bringup).

Order: lifecycle_manager starts first and brings up all nodes
tier by tier (0→6) — no TimerAction, driven by actual readiness.

Machine-specific defaults (servers, device paths, data paths) come from
config/robot.yaml (or the file in $INMOOV_ROBOT_CONFIG); secrets from the
environment. Every value can still be overridden as a launch argument.

Usage:
  ros2 launch inmoov_bringup inmoov.launch.py
  ros2 launch inmoov_bringup inmoov.launch.py tavily_api_key:=tvly-...
  ros2 launch inmoov_bringup inmoov.launch.py vision:=false
  ros2 launch inmoov_bringup inmoov.launch.py telegram:=true

Author: Artur Fedjukevits
Assisted by: Claude Code (Anthropic)
License: GNU General Public License v3.0 (see repository root LICENSE)
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode, Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

# PulseAudio/PipeWire — needed only by audio nodes
_uid = os.getuid()
_AUDIO_ENV = {
    'PULSE_SERVER': f'unix:/run/user/{_uid}/pulse/native',
    'DBUS_SESSION_BUS_ADDRESS': os.environ.get(
        'DBUS_SESSION_BUS_ADDRESS', f'unix:path=/run/user/{_uid}/bus'),
}


# Nodes started only with vision:=true (must match the names in config/lifecycle.yaml)
_VISION_NODES = [
    'face_capture_node', 'oak_node',
    'face_detection_node_left', 'face_detection_node_right',
    'face_tracker_node_left', 'face_tracker_node_right',
    'face_recognition_node', 'face_gallery_node', 'emotion_recognition_node',
    'vision_head_tracker_node', 'human_detection_node', 'scene_manager_node',
]
_TELEGRAM_NODES = ['telegram_bridge_node']


def _load_robot_config() -> dict:
    """config/robot.yaml (or $INMOOV_ROBOT_CONFIG) flattened to {launch_arg: str}."""
    path = os.environ.get('INMOOV_ROBOT_CONFIG') or os.path.join(
        get_package_share_directory('inmoov_bringup'), 'config', 'robot.yaml')
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    flat = {}
    for section in cfg.values():
        flat.update(section or {})
    return {k: os.path.expanduser(v) if isinstance(v, str) else str(v)
            for k, v in flat.items()}


def _is_true(context, arg: str) -> bool:
    return LaunchConfiguration(arg).perform(context).strip().lower() in ('true', '1', 'yes')


def _make_lifecycle_manager(context, config_path):
    # Nodes switched off by launch arguments are passed to the manager as
    # disabled_nodes — otherwise it would wait out retries/timeouts on every tier.
    disabled = []
    if not _is_true(context, 'vision'):
        disabled += _VISION_NODES
    if not _is_true(context, 'telegram'):
        disabled += _TELEGRAM_NODES
    return [Node(
        package='inmoov_bringup',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        parameters=[{
            'config_file':              config_path,
            'retry_count':              3,
            'retry_interval_sec':       10.0,
            'transition_timeout_sec':   30.0,
            'tier_advance_timeout_sec': 90.0,
            'autostart_delay_sec':      5.0,
            'disabled_nodes':           ','.join(disabled),
        }],
    )]


def generate_launch_description():
    robot = _load_robot_config()

    # ── Arguments ────────────────────────────────────────────────────────────
    args = [
        # Servers
        # llm_url — OpenAI-compatible chat.completions endpoint (currently vLLM), shared
        # by llm_node and identity_manager_node (name extraction). llm_fallback_url —
        # optional backup endpoint; empty = no fallback. (The local Ollama qwen2.5:7b on the
        # NUC was dropped: ~375 s CPU prefill for the ~8.7k-token prompt + 4096 ctx truncation.)
        DeclareLaunchArgument('llm_url',          default_value=robot['llm_url']),
        DeclareLaunchArgument('llm_fallback_url', default_value=robot['llm_fallback_url']),
        DeclareLaunchArgument('vision_llm_url',   default_value=robot['vision_llm_url']),
        DeclareLaunchArgument('llm_bearer_token',
            default_value=os.environ.get('VLLM_BEARER_TOKEN', '')),
        DeclareLaunchArgument('tts_server_url',   default_value=robot['tts_server_url']),
        DeclareLaunchArgument('tts_fallback_url', default_value=robot['tts_fallback_url']),
        DeclareLaunchArgument('cast_to_file_url', default_value=robot['cast_to_file_url']),
        DeclareLaunchArgument('openhab_url',      default_value=robot['openhab_url']),

        # LLM
        DeclareLaunchArgument('llm_model',
            default_value='qwen3.8-27b'),
        DeclareLaunchArgument('llm_temperature',    default_value='0.1'),
        DeclareLaunchArgument('llm_max_tokens',     default_value='512'),

        # Wake word
        DeclareLaunchArgument('wakeword_model', default_value=robot['wakeword_model']),
        DeclareLaunchArgument('wakeword_threshold', default_value='0.9'),
        DeclareLaunchArgument('wakeword_patience',  default_value='2'),

        # Audio
        DeclareLaunchArgument('audio_device_index', default_value='-1'),
        DeclareLaunchArgument('audio_device_name',  default_value='pulse'),
        DeclareLaunchArgument('output_device_name', default_value=''),
        DeclareLaunchArgument('sample_rate',        default_value='16000'),
        DeclareLaunchArgument('pa_source_check',    default_value='Jabra'),  # '' = don't check

        # VAD / SV
        DeclareLaunchArgument('vad_threshold',          default_value='0.4'),
        DeclareLaunchArgument('silence_duration_sec',   default_value='2.5'),
        DeclareLaunchArgument('pipeline_timeout_sec',   default_value='45.0'),
        DeclareLaunchArgument('speaker_verification',   default_value='true'),
        # Phrase-level SV (see voice_detector_node): cosine threshold for a whole phrase,
        # minimum speech to judge, and a folder that keeps every judged phrase as WAV
        # (for tuning the threshold on real data; '' = off)
        DeclareLaunchArgument('sv_threshold',           default_value='0.35'),
        DeclareLaunchArgument('sv_min_speech_sec',      default_value='1.2'),
        DeclareLaunchArgument('sv_debug_dir',           default_value=os.path.expanduser('~/inmoov_sv_debug')),

        # Tavily
        DeclareLaunchArgument('tavily_api_key',
            default_value=os.environ.get('TAVILY_API_KEY', '')),

        # Arduino
        DeclareLaunchArgument('port_right', default_value=robot['port_right']),
        DeclareLaunchArgument('port_left',  default_value=robot['port_left']),

        # Memory
        DeclareLaunchArgument('memory_db_path',   default_value=robot['memory_db_path']),
        DeclareLaunchArgument('episodic_db_path', default_value=robot['episodic_db_path']),
        DeclareLaunchArgument('semantic_db_path', default_value=robot['semantic_db_path']),
        DeclareLaunchArgument('chroma_path',      default_value=robot['chroma_path']),
        DeclareLaunchArgument('reminder_db_path', default_value=robot['reminder_db_path']),
        DeclareLaunchArgument('gallery_dir',      default_value=robot['gallery_dir']),

        # Vision
        DeclareLaunchArgument('vision',    default_value='true'),
        DeclareLaunchArgument('cam_left',  default_value=robot['cam_left']),
        DeclareLaunchArgument('cam_right', default_value=robot['cam_right']),
        DeclareLaunchArgument('fps',                  default_value='15'),
        DeclareLaunchArgument('detection_hz',         default_value='5.0'),
        DeclareLaunchArgument('det_thresh',           default_value='0.5'),
        DeclareLaunchArgument('analysis_hz',          default_value='2.0'),
        DeclareLaunchArgument('gain_head',            default_value='0.3'),
        DeclareLaunchArgument('gain_eye',             default_value='0.6'),
        DeclareLaunchArgument('rest_rothead',         default_value='90.0'),
        DeclareLaunchArgument('rest_neck',            default_value='40.0'),
        DeclareLaunchArgument('oak_model',            default_value='yolov6-nano'),
        DeclareLaunchArgument('oak_conf_threshold',   default_value='0.5'),
        DeclareLaunchArgument('scene_location',       default_value=''),

        # Cognition
        DeclareLaunchArgument('greet_cooldown_sec',   default_value='120.0'),
        DeclareLaunchArgument('bt_tick_rate_hz',      default_value='10.0'),

        # Telegram
        DeclareLaunchArgument('telegram',
            default_value='false'),
        # Android app / external clients: rosbridge WebSocket (ws://<robot>:<port>)
        # + urdf_bridge (servo <-> URDF conversion, /urdf_joint_states, /urdf_joint_cmd)
        DeclareLaunchArgument('rosbridge', default_value='true'),
        DeclareLaunchArgument('rosbridge_port', default_value='9090'),
        DeclareLaunchArgument('allowed_chat_id',
            default_value=os.environ.get('TELEGRAM_ALLOWED_CHAT_ID', '0')),
    ]

    # ── Lifecycle Manager (a plain Node — manages the others) ────────────────
    # lifecycle.yaml contains complex structures (tiers) — can't pass it as
    # --params-file (the ROS2 parser doesn't support nested lists). Read it in Python instead.
    lifecycle_config_path = PathJoinSubstitution([
        FindPackageShare('inmoov_bringup'), 'config', 'lifecycle.yaml'
    ])
    lifecycle_manager = OpaqueFunction(
        function=_make_lifecycle_manager, args=[lifecycle_config_path])

    # ── Tier 0: memory_node ──────────────────────────────────────────────────
    memory_node = LifecycleNode(
        package='inmoov_memory',
        executable='memory_node',
        name='memory_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'db_path':              LaunchConfiguration('memory_db_path'),
            'episodic_db_path':     LaunchConfiguration('episodic_db_path'),
            'semantic_db_path':     LaunchConfiguration('semantic_db_path'),
            'chroma_path':          LaunchConfiguration('chroma_path'),
            'reminder_db_path':     LaunchConfiguration('reminder_db_path'),
            'gallery_dir':          LaunchConfiguration('gallery_dir'),
            'similarity_threshold': 0.55,
            'llm_url':              LaunchConfiguration('llm_url'),
            'llm_model':            LaunchConfiguration('llm_model'),
            'bearer_token':         LaunchConfiguration('llm_bearer_token'),
        }],
    )

    # ── Tier 1: Hardware ─────────────────────────────────────────────────────
    audio_source = LifecycleNode(
        package='inmoov_voice',
        executable='audio_source_node',
        name='audio_source_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        additional_env=_AUDIO_ENV,
        parameters=[{
            'sample_rate':     LaunchConfiguration('sample_rate'),
            'chunk_size':      512,
            'device_index':    LaunchConfiguration('audio_device_index'),
            'device_name':     LaunchConfiguration('audio_device_name'),
            'pa_source_check': LaunchConfiguration('pa_source_check'),
        }],
    )

    arduino_right = LifecycleNode(
        package='inmoov_control',
        executable='arduino_right_node',
        name='arduino_right',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        arguments=['--port', LaunchConfiguration('port_right')],
    )

    arduino_left = LifecycleNode(
        package='inmoov_control',
        executable='arduino_left_node',
        name='arduino_left',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        arguments=['--port', LaunchConfiguration('port_left')],
    )

    face_capture = LifecycleNode(
        package='inmoov_vision',
        executable='face_capture_node',
        name='face_capture_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'cam_left':     LaunchConfiguration('cam_left'),
            'cam_right':    LaunchConfiguration('cam_right'),
            'fps':          LaunchConfiguration('fps'),
            'width':        640,
            'height':       480,
            'jpeg_quality': 85,
        }],
    )

    oak_node = LifecycleNode(
        package='inmoov_vision',
        executable='oak_node',
        name='oak_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'model_name':     LaunchConfiguration('oak_model'),
            'conf_threshold': LaunchConfiguration('oak_conf_threshold'),
        }],
    )

    sound_localization = LifecycleNode(
        package='inmoov_voice',
        executable='sound_localization_node',
        name='sound_localization_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
    )

    # ── Tier 2: HW Consumers ─────────────────────────────────────────────────
    wakeword = LifecycleNode(
        package='inmoov_voice',
        executable='wakeword_node',
        name='wakeword_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'model_path':   LaunchConfiguration('wakeword_model'),
            'threshold':    LaunchConfiguration('wakeword_threshold'),
            'patience':     LaunchConfiguration('wakeword_patience'),
            'debounce_sec': 1.5,
        }],
    )

    voice_detector = LifecycleNode(
        package='inmoov_voice',
        executable='voice_detector_node',
        name='voice_detector_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
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
            'sv_min_speech_sec':     LaunchConfiguration('sv_min_speech_sec'),
            'sv_debug_dir':          LaunchConfiguration('sv_debug_dir'),
        }],
    )

    tts_node = LifecycleNode(
        package='inmoov_voice',
        executable='tts_node',
        name='tts_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
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

    joint_state_publisher = LifecycleNode(
        package='inmoov_control',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
    )

    # servo <-> URDF conversion for the Android app, RViz, MoveIt (see urdf_bridge_node.py)
    urdf_bridge = LifecycleNode(
        package='inmoov_control',
        executable='urdf_bridge_node',
        name='urdf_bridge',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
    )

    # Plain (non-lifecycle) node — not managed by lifecycle_manager.
    # No authentication: the globs below limit clients to what the Android app
    # needs (everything else — lifecycle services, /robot_sleep, ... — is refused).
    # Glob format of rosbridge_server 2.x: a string "['a', 'b']", fnmatch patterns.
    rosbridge = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'port': ParameterValue(LaunchConfiguration('rosbridge_port'), value_type=int),
            'address': '',
            # /joint_cmd: calibration drives the servo directly (priority 90)
            'topics_pub_glob': "['/urdf_joint_cmd', '/joint_cmd', "
                               "'/urdf_bridge/set_calibration']",
            # the rest is the app's read-only diagnostics (servo pose, arbitration,
            # failsafe, lifecycle state); change_state & co. stay refused
            'topics_sub_glob': "['/urdf_joint_states', '/urdf_bridge/status', "
                               "'/joint_states', '/face_joint_states', '/joint_commanded', "
                               "'/arduino_*/failsafe', '/arduino_*/joint_owners']",
            'services_glob': "['/urdf_bridge/get_map', '/*/get_state']",
            'actions_glob': '[]',
        }],
        condition=IfCondition(LaunchConfiguration('rosbridge')),
    )

    face_expressions = LifecycleNode(
        package='inmoov_control',
        executable='face_expressions_node',
        name='face_expressions',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
    )

    face_detection_left = LifecycleNode(
        package='inmoov_vision',
        executable='face_detection_node',
        name='face_detection_node_left',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'camera_side':  'left',
            'detection_hz': LaunchConfiguration('detection_hz'),
            'det_size':     640,
            'det_thresh':   LaunchConfiguration('det_thresh'),
            'model_name':   'buffalo_l',
        }],
    )

    face_detection_right = LifecycleNode(
        package='inmoov_vision',
        executable='face_detection_node',
        name='face_detection_node_right',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'camera_side':  'right',
            'detection_hz': LaunchConfiguration('detection_hz'),
            'det_size':     640,
            'det_thresh':   LaunchConfiguration('det_thresh'),
            'model_name':   'buffalo_l',
            # A real fallback: insightface on the right eye only turns on
            # when the left (primary) actually stops publishing for >1.5s, rather than
            # running constantly in parallel with the left — saves CPU.
            'fallback_for':        '/face/detections/left',
            'primary_timeout_sec': 1.5,
        }],
    )

    # ── Tier 3: Processing ───────────────────────────────────────────────────
    face_tracker_left = LifecycleNode(
        package='inmoov_vision',
        executable='face_tracker_node',
        name='face_tracker_node_left',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'camera_side':     'left',
            'iou_threshold':   0.20,
            'max_lost_frames': 20,
            'max_tracks':      4,
        }],
    )

    face_tracker_right = LifecycleNode(
        package='inmoov_vision',
        executable='face_tracker_node',
        name='face_tracker_node_right',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'camera_side':     'right',
            'iou_threshold':   0.20,
            'max_lost_frames': 20,
            'max_tracks':      4,
        }],
    )

    voice_emotion = LifecycleNode(
        package='inmoov_voice',
        executable='voice_emotion_node',
        name='voice_emotion_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'min_confidence': 0.55,
        }],
    )

    # STT: Parakeet-TDT-0.6b-v3 (ONNX/CPU) — replaced whisper.cpp on 2026-08-27,
    # 2-4x faster in live testing.
    parakeet_stt = LifecycleNode(
        package='inmoov_voice',
        executable='parakeet_stt_node',
        name='parakeet_stt_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'language': 'ru',
        }],
    )

    # ── Tier 4: Intelligence ─────────────────────────────────────────────────
    face_recognition = LifecycleNode(
        package='inmoov_vision',
        executable='face_recognition_node',
        name='face_recognition_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{'recognition_cooldown_sec': 3.0}],
    )

    face_gallery = LifecycleNode(
        package='inmoov_vision',
        executable='face_gallery_node',
        name='face_gallery_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'gallery_dir':           LaunchConfiguration('gallery_dir'),
            'enroll_interval_sec':   1.0,
            'interact_interval_sec': 15.0,
            'min_det_score':         0.75,
            'min_face_px':           60,
        }],
    )

    emotion_recognition = LifecycleNode(
        package='inmoov_vision',
        executable='emotion_recognition_node',
        name='emotion_recognition_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'analysis_hz':   LaunchConfiguration('analysis_hz'),
            'min_face_size': 48,
        }],
    )

    head_tracker = LifecycleNode(
        package='inmoov_vision',
        executable='vision_head_tracker_node',
        name='vision_head_tracker_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'image_width':        640,
            'image_height':       480,
            'gain_head':          LaunchConfiguration('gain_head'),
            'gain_eye':           LaunchConfiguration('gain_eye'),
            'rest_rothead':       LaunchConfiguration('rest_rothead'),
            'rest_neck':          LaunchConfiguration('rest_neck'),
            'rest_eye_lr':        90.0,
            'rest_eye_ud':        100.0,
            'dead_zone_px':       20,
            'head_dead_zone_px':  80,
            'return_timeout_sec': 10.0,  # was 7.0 — still too little, dropped the head to rest mid-dialogue on a brief loss of face at the edge of the frame (2026-09-01)
            'track_hz':           10.0,
            'max_step_deg':       2.0,
            'bbox_ema_alpha':     0.4,
        }],
    )

    human_detection = LifecycleNode(
        package='inmoov_vision',
        executable='human_detection_node',
        name='human_detection_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'max_distance_m':   4.0,
            'min_confidence':   0.45,
            'lost_timeout_sec': 4.0,
            'publish_rate_hz':  5.0,
        }],
    )

    scene_manager = LifecycleNode(
        package='inmoov_vision',
        executable='scene_manager_node',
        name='scene_manager_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
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

    llm_node = LifecycleNode(
        package='inmoov_cognition',
        executable='llm_node',
        name='llm_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
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
            'vision_llm_url':      LaunchConfiguration('vision_llm_url'),
            'tts_server_url':      LaunchConfiguration('tts_server_url'),
            'tts_fallback_url':    LaunchConfiguration('tts_fallback_url'),
            'cast_to_file_url':    LaunchConfiguration('cast_to_file_url'),
        }],
    )

    openhab_bridge = LifecycleNode(
        package='inmoov_cognition',
        executable='openhab_bridge_node',
        name='openhab_bridge_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'openhab_url':       LaunchConfiguration('openhab_url'),
            'items_tag':         'ChatGPT',
            'reconnect_sec':     15.0,
            'schema_repeat_sec': 5.0,
        }],
    )

    # ── Tier 5: Orchestration ────────────────────────────────────────────────
    identity_manager = LifecycleNode(
        package='inmoov_cognition',
        executable='identity_manager_node',
        name='identity_manager_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'greet_cooldown_sec':     LaunchConfiguration('greet_cooldown_sec'),
            'no_face_timeout_sec':    15.0,
            'no_human_timeout_sec':   30.0,
            'introduce_cooldown_sec': 120.0,
            'emotion_react_thresh':   0.70,
            'llm_url':                LaunchConfiguration('llm_url'),
            'llm_fallback_url':       LaunchConfiguration('llm_fallback_url'),
            'bearer_token':           LaunchConfiguration('llm_bearer_token'),
            'name_extract_model':     LaunchConfiguration('llm_model'),
        }],
    )

    behavior_manager = LifecycleNode(
        package='inmoov_cognition',
        executable='behavior_manager_node',
        name='behavior_manager_node',
        namespace='',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[{
            'tavily_api_key':  LaunchConfiguration('tavily_api_key'),
            'bt_tick_rate_hz': LaunchConfiguration('bt_tick_rate_hz'),
        }],
    )

    # ── Tier 6: Optional ─────────────────────────────────────────────────────
    telegram_bridge = GroupAction(
        condition=IfCondition(LaunchConfiguration('telegram')),
        actions=[
            LifecycleNode(
                package='inmoov_cognition',
                executable='telegram_bridge_node',
                name='telegram_bridge_node',
                namespace='',
                output='screen',
                respawn=True,
                respawn_delay=2.0,
                parameters=[{
                    'allowed_chat_id': LaunchConfiguration('allowed_chat_id'),
                    'llm_timeout_sec': 35.0,
                    'memory_db_path':  LaunchConfiguration('memory_db_path'),
                    'cam_device':      LaunchConfiguration('cam_left'),
                }],
            ),
        ],
    )

    # vision:=false — none of the camera/face/OAK nodes are started
    vision_group = GroupAction(
        condition=IfCondition(LaunchConfiguration('vision')),
        actions=[
            face_capture,
            oak_node,
            face_detection_left,
            face_detection_right,
            face_tracker_left,
            face_tracker_right,
            face_recognition,
            face_gallery,
            emotion_recognition,
            head_tracker,
            human_detection,
            scene_manager,
        ],
    )

    return LaunchDescription(args + [
        # lifecycle_manager starts first — it will bring up the rest
        lifecycle_manager,
        # All nodes start in the Unconfigured state — the manager controls them
        memory_node,
        audio_source,
        arduino_right,
        arduino_left,
        sound_localization,
        wakeword,
        voice_detector,
        tts_node,
        joint_state_publisher,
        urdf_bridge,
        rosbridge,
        face_expressions,
        voice_emotion,
        parakeet_stt,
        vision_group,
        llm_node,
        openhab_bridge,
        identity_manager,
        behavior_manager,
        telegram_bridge,
    ])
