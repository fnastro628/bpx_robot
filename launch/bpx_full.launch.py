"""
Full system launch file.

Brings up: BPX driver, VSLAM, object detector, face recognizer,
           speech-to-text, command parser, behavior manager,
           and the full acoustic pipeline (CAP 1–15).

Usage:
  ros2 launch bpx_robot bpx_full.launch.py
  ros2 launch bpx_robot bpx_full.launch.py robot_ip:=192.168.0.1
  ros2 launch bpx_robot bpx_full.launch.py vslam:=false     # skip RTAB-Map
  ros2 launch bpx_robot bpx_full.launch.py acoustic:=false  # skip acoustic pipeline
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # ── Arguments ──────────────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument("robot_ip",    default_value="10.21.20.1"),
        DeclareLaunchArgument("vslam",       default_value="true"),
        DeclareLaunchArgument("detection",   default_value="true"),
        DeclareLaunchArgument("face_recog",  default_value="true"),
        DeclareLaunchArgument("speech",      default_value="true"),
        DeclareLaunchArgument("whisper_model", default_value="small"),
        DeclareLaunchArgument("wake_word",   default_value="hey robot"),
        DeclareLaunchArgument("yolo_model",  default_value="yolov8n.pt"),
        DeclareLaunchArgument("acoustic",    default_value="true"),
        DeclareLaunchArgument("wake_model",  default_value="hey_jarvis"),
    ]

    robot_ip       = LaunchConfiguration("robot_ip")
    use_vslam      = LaunchConfiguration("vslam")
    use_detection  = LaunchConfiguration("detection")
    use_face       = LaunchConfiguration("face_recog")
    use_speech     = LaunchConfiguration("speech")
    use_acoustic   = LaunchConfiguration("acoustic")

    # ── BPX driver ─────────────────────────────────────────────────────────────
    bpx_node = Node(
        package="bpx_driver",
        executable="bpx_node",
        name="bpx_node",
        parameters=[{
            "robot_ip":        robot_ip,
            "state_rate_hz":   100,
            "command_rate_hz": 50,
            "publish_rate_hz": 100,
        }],
        output="screen",
    )

    # ── Behavior manager ───────────────────────────────────────────────────────
    behavior_manager = Node(
        package="bpx_robot",
        executable="behavior_manager.py",
        name="behavior_manager",
        output="screen",
    )

    # ── Speech-to-text ────────────────────────────────────────────────────────
    stt_node = Node(
        package="bpx_robot",
        executable="stt_node.py",
        name="stt_node",
        condition=IfCondition(use_speech),
        parameters=[{
            "model_size": LaunchConfiguration("whisper_model"),
            "device":     "cuda",
            "wake_word":  LaunchConfiguration("wake_word"),
        }],
        output="screen",
    )

    command_parser = Node(
        package="bpx_robot",
        executable="command_parser.py",
        name="command_parser",
        condition=IfCondition(use_speech),
        output="screen",
    )

    # ── Object detection ──────────────────────────────────────────────────────
    detector = Node(
        package="bpx_robot",
        executable="detector.py",
        name="object_detector",
        condition=IfCondition(use_detection),
        parameters=[{
            "model_path":     LaunchConfiguration("yolo_model"),
            "conf_threshold": 0.50,
            "device":         "0",
        }],
        output="screen",
    )

    # ── Face recognition ──────────────────────────────────────────────────────
    face_recognizer = Node(
        package="bpx_robot",
        executable="recognizer.py",
        name="face_recognizer",
        condition=IfCondition(use_face),
        parameters=[{"similarity_threshold": 0.45}],
        output="screen",
    )

    # ── RTAB-Map VSLAM ────────────────────────────────────────────────────────
    config_dir = os.path.join(
        get_package_share_directory("bpx_robot"), "config"
    )

    rtabmap = Node(
        package="rtabmap_ros",
        executable="rtabmap",
        name="rtabmap",
        condition=IfCondition(use_vslam),
        arguments=["--delete_db_on_start"],   # remove to load existing map
        parameters=[os.path.join(config_dir, "rtabmap.yaml")],
        remappings=[
            ("/left/image_rect",   "/camera/left/image_raw"),
            ("/right/image_rect",  "/camera/right/image_raw"),
            ("/left/camera_info",  "/camera/left/camera_info"),
            ("/right/camera_info", "/camera/right/camera_info"),
        ],
        output="screen",
    )

    # ── Acoustic pipeline (CAP 1–15) ──────────────────────────────────────────
    xvf3800_node = Node(
        package="bpx_robot",
        executable="acoustic/xvf3800_node.py",
        name="xvf3800_node",
        condition=IfCondition(use_acoustic),
        output="screen",
    )

    head_controller = Node(
        package="bpx_robot",
        executable="head_controller.py",
        name="head_controller",
        condition=IfCondition(use_acoustic),
        output="screen",
    )

    doa_tracker = Node(
        package="bpx_robot",
        executable="acoustic/doa_tracker.py",
        name="doa_tracker",
        condition=IfCondition(use_acoustic),
        output="screen",
    )

    clap_detector = Node(
        package="bpx_robot",
        executable="acoustic/clap_detector.py",
        name="clap_detector",
        condition=IfCondition(use_acoustic),
        output="screen",
    )

    passive_tracker = Node(
        package="bpx_robot",
        executable="acoustic/passive_tracker.py",
        name="passive_tracker",
        condition=IfCondition(use_acoustic),
        output="screen",
    )

    proximity_estimator = Node(
        package="bpx_robot",
        executable="acoustic/proximity_estimator.py",
        name="proximity_estimator",
        condition=IfCondition(use_acoustic),
        output="screen",
    )

    paging_node = Node(
        package="bpx_robot",
        executable="acoustic/paging.py",
        name="paging_node",
        condition=IfCondition(use_acoustic),
        parameters=[{"wake_model": LaunchConfiguration("wake_model")}],
        output="screen",
    )

    event_detector = Node(
        package="bpx_robot",
        executable="acoustic/event_detector.py",
        name="event_detector",
        condition=IfCondition(use_acoustic),
        output="screen",
    )

    room_classifier = Node(
        package="bpx_robot",
        executable="acoustic/room_acoustics/room_classifier.py",
        name="room_classifier",
        condition=IfCondition(use_acoustic),
        output="screen",
    )

    speaker_memory = Node(
        package="bpx_robot",
        executable="acoustic/speaker_memory.py",
        name="speaker_memory",
        condition=IfCondition(use_acoustic),
        output="screen",
    )

    emotion_detector = Node(
        package="bpx_robot",
        executable="acoustic/emotion_detector.py",
        name="emotion_detector",
        condition=IfCondition(use_acoustic),
        output="screen",
    )

    acoustic_slam = Node(
        package="bpx_robot",
        executable="acoustic/acoustic_slam.py",
        name="acoustic_slam",
        condition=IfCondition(use_acoustic),
        output="screen",
    )

    return LaunchDescription(args + [
        bpx_node,
        behavior_manager,
        stt_node,
        command_parser,
        detector,
        face_recognizer,
        rtabmap,
        # Acoustic pipeline
        xvf3800_node,
        head_controller,
        doa_tracker,
        clap_detector,
        passive_tracker,
        proximity_estimator,
        paging_node,
        event_detector,
        room_classifier,
        speaker_memory,
        emotion_detector,
        acoustic_slam,
    ])
