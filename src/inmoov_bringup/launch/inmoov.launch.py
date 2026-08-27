#!/usr/bin/env python3
"""
inmoov.launch.py — главный launch-файл InMoov Robot (inmoov_bringup).

Порядок: lifecycle_manager стартует первым и поднимает все ноды
по тирам (0→6) — без TimerAction, по реальной готовности.

Использование:
  ros2 launch inmoov_bringup inmoov.launch.py
  ros2 launch inmoov_bringup inmoov.launch.py tavily_api_key:=tvly-...
  ros2 launch inmoov_bringup inmoov.launch.py vision:=false
  ros2 launch inmoov_bringup inmoov.launch.py telegram:=true
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode, Node
from launch_ros.substitutions import FindPackageShare

# PulseAudio/PipeWire — нужен только аудио-нодам
_uid = os.getuid()
_AUDIO_ENV = {
    'PULSE_SERVER': f'unix:/run/user/{_uid}/pulse/native',
    'DBUS_SESSION_BUS_ADDRESS': os.environ.get(
        'DBUS_SESSION_BUS_ADDRESS', f'unix:path=/run/user/{_uid}/bus'),
}


def generate_launch_description():

    # ── Аргументы ────────────────────────────────────────────────────────────
    args = [
        # Серверы
        # llm_url — OpenAI-совместимый chat.completions endpoint (сейчас vLLM), общий
        # для llm_node и identity_manager_node (name extraction). llm_fallback_url —
        # резервный (сейчас локальный NUC — станет OpenAI-совместимым позже).
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
        DeclareLaunchArgument('llm_temperature',    default_value='0.1'),
        DeclareLaunchArgument('llm_max_tokens',     default_value='512'),

        # Wake word
        DeclareLaunchArgument('wakeword_model',
            default_value='/home/artur/openWakeWord/my_custom_model/ey_lyonya.onnx'),
        DeclareLaunchArgument('wakeword_threshold', default_value='0.3'),

        # Аудио
        DeclareLaunchArgument('audio_device_index', default_value='-1'),
        DeclareLaunchArgument('audio_device_name',  default_value='pulse'),
        DeclareLaunchArgument('output_device_name', default_value=''),
        DeclareLaunchArgument('sample_rate',        default_value='16000'),
        DeclareLaunchArgument('pa_source_check',    default_value='Jabra'),  # '' = не проверять

        # VAD / SV
        DeclareLaunchArgument('vad_threshold',          default_value='0.5'),
        DeclareLaunchArgument('silence_duration_sec',   default_value='2.5'),
        DeclareLaunchArgument('pipeline_timeout_sec',   default_value='45.0'),
        DeclareLaunchArgument('speaker_verification',   default_value='true'),
        DeclareLaunchArgument('sv_threshold',           default_value='0.45'),
        DeclareLaunchArgument('sv_segment_sec',         default_value='3.0'),

        # Tavily
        DeclareLaunchArgument('tavily_api_key',
            default_value=os.environ.get('TAVILY_API_KEY', '')),

        # Arduino
        DeclareLaunchArgument('port_right',
            default_value='/dev/serial/by-path/pci-0000:c6:00.3-usb-0:5:1.0-port0'),
        DeclareLaunchArgument('port_left',
            default_value='/dev/serial/by-path/pci-0000:c6:00.3-usb-0:1.3:1.0-port0'),

        # Memory
        DeclareLaunchArgument('memory_db_path',
            default_value='/home/artur/inmoov_memory.db'),

        # Vision
        DeclareLaunchArgument('vision',    default_value='true'),
        DeclareLaunchArgument('cam_left',
            default_value='/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1.1:1.0-video-index0'),
        DeclareLaunchArgument('cam_right',
            default_value='/dev/v4l/by-path/pci-0000:c6:00.3-usb-0:1.2:1.0-video-index0'),
        DeclareLaunchArgument('fps',                  default_value='15'),
        DeclareLaunchArgument('detection_hz',         default_value='2.5'),
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
        DeclareLaunchArgument('allowed_chat_id',
            default_value=os.environ.get('TELEGRAM_ALLOWED_CHAT_ID', '0')),
    ]

    # ── Lifecycle Manager (обычный Node — управляет другими) ─────────────────
    # lifecycle.yaml содержит сложные структуры (tiers) — нельзя передавать как
    # --params-file (ROS2 парсер не поддерживает nested lists). Читаем Python-ом.
    lifecycle_config_path = PathJoinSubstitution([
        FindPackageShare('inmoov_bringup'), 'config', 'lifecycle.yaml'
    ])
    lifecycle_manager = Node(
        package='inmoov_bringup',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        parameters=[{
            'config_file':              lifecycle_config_path,
            'retry_count':              3,
            'retry_interval_sec':       10.0,
            'transition_timeout_sec':   30.0,
            'tier_advance_timeout_sec': 90.0,
            'autostart_delay_sec':      5.0,
        }],
    )

    # ── Тир 0: memory_node ───────────────────────────────────────────────────
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
            'similarity_threshold': 0.55,
            'llm_url':              LaunchConfiguration('llm_url'),
            'llm_model':            LaunchConfiguration('llm_model'),
            'bearer_token':         LaunchConfiguration('llm_bearer_token'),
        }],
    )

    # ── Тир 1: Hardware ──────────────────────────────────────────────────────
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

    # ── Тир 2: HW Consumers ──────────────────────────────────────────────────
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
            'sv_segment_sec':        LaunchConfiguration('sv_segment_sec'),
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
            # Настоящий fallback: insightface на правом включается только
            # когда левый (primary) реально не публикует >1.5с, а не гоняется
            # постоянно параллельно с левым — экономия CPU.
            'fallback_for':        '/face/detections/left',
            'primary_timeout_sec': 1.5,
        }],
    )

    # ── Тир 3: Processing ────────────────────────────────────────────────────
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
            'savedir':        '/home/artur/.cache/speechbrain/voice_emotion',
        }],
    )

    # STT: Parakeet-TDT-0.6b-v3 (ONNX/CPU) — заменил whisper.cpp 2026-08-27,
    # 2-4x быстрее вживую. См. project_stt_parakeet_eval.md
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

    # ── Тир 4: Intelligence ──────────────────────────────────────────────────
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
            'gallery_dir':           '/home/artur/inmoov_faces',
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
            'fov_h_deg':          60.0,
            'fov_v_deg':          45.0,
            'gain_head':          LaunchConfiguration('gain_head'),
            'gain_eye':           LaunchConfiguration('gain_eye'),
            'rest_rothead':       LaunchConfiguration('rest_rothead'),
            'rest_neck':          LaunchConfiguration('rest_neck'),
            'rest_eye_lr':        90.0,
            'rest_eye_ud':        100.0,
            'dead_zone_px':       20,
            'head_dead_zone_px':  80,
            'eye_limit_deg':      8.0,
            'return_timeout_sec': 7.0,  # было 3.0 — слишком мало для догона после наведения по OAK-D (2026-08-24)
            'track_hz':           15.0,
            'max_step_deg':       2.0,
            'bbox_ema_alpha':     0.4,
            'max_stale_ticks':    5,
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

    # ── Тир 5: Orchestration ─────────────────────────────────────────────────
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

    # ── Тир 6: Optional ──────────────────────────────────────────────────────
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

    return LaunchDescription(args + [
        # lifecycle_manager стартует первым — он будет поднимать остальные
        lifecycle_manager,
        # Все ноды стартуют в состоянии Unconfigured — manager ими управляет
        memory_node,
        audio_source,
        arduino_right,
        arduino_left,
        face_capture,
        oak_node,
        sound_localization,
        wakeword,
        voice_detector,
        tts_node,
        joint_state_publisher,
        face_expressions,
        face_detection_left,
        face_detection_right,
        face_tracker_left,
        face_tracker_right,
        voice_emotion,
        parakeet_stt,
        face_recognition,
        face_gallery,
        emotion_recognition,
        head_tracker,
        human_detection,
        scene_manager,
        llm_node,
        openhab_bridge,
        identity_manager,
        behavior_manager,
        telegram_bridge,
    ])
