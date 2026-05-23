# BPX Robot — Jetson Orin Nano Super Control System

Autonomous quadruped control stack for the BPX robot using an articulated
sensor head mounted on the chassis.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      JETSON ORIN NANO SUPER                          │
│                                                                      │
│  ┌─────────────┐   /speech/raw    ┌──────────────────┐              │
│  │  stt_node   │ ──────────────► │ command_parser   │              │
│  │  (Whisper)  │                  └────────┬─────────┘              │
│  └─────────────┘                           │ /commands              │
│                                            ▼                        │
│  ┌─────────────┐   /bpx/*        ┌──────────────────┐              │
│  │  bpx_node   │ ◄──────────────►│ behavior_manager │              │
│  │  (C++ SDK)  │   /cmd_vel      │  come/heel/fetch │              │
│  └─────────────┘                  └────────▲─────────┘              │
│                                            │                        │
│  ┌─────────────┐  /perception/*            │                        │
│  │  detector   │ ──────────────────────────┘                        │
│  │  (YOLOv8)  │                                                     │
│  │  recognizer │                                                     │
│  │  (ArcFace)  │                                                     │
│  └─────────────┘                                                     │
│                                                                      │
│  ┌─────────────┐  /map /odom                                         │
│  │  rtabmap    │ (VSLAM — stereo camera)                             │
│  └─────────────┘                                                     │
└─────────────────────────────────────────────────────────────────────┘
              │ TCP/UDP (Ethernet)
              ▼
        ┌─────────────┐
        │  BPX Robot  │  192.168.1.80 (default wired)
        └─────────────┘
```

---

## Repository Layout

```
bpx_robot/
├── hardware/
│   └── head_design.md          # BOM, wiring, 3D print specs
├── bpx_driver/                 # ROS2 C++ package — wraps the BPX SDK
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── include/bpx_driver/bpx_node.hpp
│   └── src/bpx_node.cpp
├── speech/
│   ├── stt_node.py             # Whisper STT + VAD
│   └── command_parser.py       # Text → structured command JSON
├── behaviors/
│   ├── gait_map.py             # SDK enum mirrors + phrase→command maps
│   ├── behavior_manager.py     # Central dispatcher (asyncio state machine)
│   └── primitives/
│       ├── come.py             # Navigate toward person
│       ├── heel.py             # Follow owner at fixed distance
│       └── fetch.py            # Find object, retrieve, return
├── perception/
│   ├── vslam/rtabmap.yaml      # RTAB-Map stereo VSLAM config
│   ├── object_detection/
│   │   └── detector.py         # YOLOv8 → /perception/person|object_detections
│   └── face_recognition/
│       ├── face_db.py          # SQLite + numpy face embedding store
│       └── recognizer.py       # InsightFace ArcFace → /perception/faces
├── config/
│   └── robot.yaml              # Global parameters
├── launch/
│   └── bpx_full.launch.py      # Full system launch
└── requirements.txt
```

---

## ROS2 Topic Map

| Topic | Type | Direction | Purpose |
|---|---|---|---|
| `/bpx/joint_states` | sensor_msgs/JointState | OUT | 12-DOF positions/velocities/torques |
| `/bpx/imu` | sensor_msgs/Imu | OUT | Body orientation + rates |
| `/bpx/battery` | sensor_msgs/BatteryState | OUT | Battery % and current |
| `/bpx/motion_state` | std_msgs/UInt8 | OUT | SDK MotionState enum value |
| `/bpx/odometry` | nav_msgs/Odometry | OUT | Leg odometry |
| `/cmd_vel` | geometry_msgs/Twist | IN → BPX | Velocity command |
| `/speech/raw` | std_msgs/String | STT → parser | Whisper transcript |
| `/commands` | std_msgs/String (JSON) | parser → manager | Structured command |
| `/perception/person_detections` | std_msgs/String (JSON) | detector → behaviors | Nearest person bbox |
| `/perception/object_detections` | std_msgs/String (JSON) | detector → behaviors | Largest non-person bbox |
| `/perception/faces` | std_msgs/String (JSON) | recognizer → behaviors | Identified faces |
| `/perception/enroll_face` | std_msgs/String | IN → recognizer | Trigger enrollment |

### BPX Services

| Service | Type | Action |
|---|---|---|
| `/bpx/stand` | std_srvs/Trigger | Stand up |
| `/bpx/sit` | std_srvs/Trigger | Sit down |
| `/bpx/damp` | std_srvs/Trigger | Joint damping (safe power-down) |
| `/bpx/zero_position` | std_srvs/Trigger | Calibrate zero (feet on floor) |

---

## Voice Commands

Wake word: **"hey robot"** (configurable)

| Say | Action |
|---|---|
| "stand up" / "get up" | Robot stands |
| "sit down" / "sit" | Robot sits |
| "relax" / "damp" | Joints go compliant |
| "stop" / "halt" / "stay" | Zero velocity |
| "forward" / "back" / "left" / "right" | Directional move |
| "turn left" / "turn right" | Yaw rotation |
| "spin" | Rotate in place |
| "walk" / "run" | Change gait |
| "bipedal" / "flip" / "pose" | Special gaits |
| "come" / "come here" | Robot approaches you |
| "heel" / "follow me" | Robot follows at ~0.8 m |
| "fetch [object]" | Find object, bring it back |
| "patrol" | Square patrol loop |

---

## Setup

### Prerequisites
- Ubuntu 22.04 + JetPack 6.x on Jetson Orin Nano Super
- ROS2 Humble: `sudo apt install ros-humble-desktop`
- RTAB-Map: `sudo apt install ros-humble-rtabmap-ros`

### Install Python deps
```bash
pip install -r requirements.txt
```

### Build the C++ driver
```bash
# Place bpx_sdk_open/ at the same level as bpx_robot/
cd ~/ros2_ws
ln -s ~/bpx_robot/bpx_driver src/bpx_driver
colcon build --packages-select bpx_driver
source install/setup.bash
```

### Camera calibration (do this once)
```bash
ros2 run camera_calibration cameracalibrator \
  --size 8x6 --square 0.025 \
  left:=/camera/left/image_raw \
  right:=/camera/right/image_raw
```

### Export YOLO to TensorRT (do this once on the Jetson)
```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='engine', device=0)"
# Then update yolo_model param to yolov8n.engine
```

### Run
```bash
# Full system
ros2 launch bpx_robot bpx_full.launch.py

# Without robot (perception + speech only)
ros2 launch bpx_robot bpx_full.launch.py robot_ip:=none vslam:=false

# Enroll a new face
ros2 topic pub /perception/enroll_face std_msgs/String "data: 'Alice'" --once
# (hold Alice in front of the camera for 1 second, then it auto-completes)
```

---

## What Works Without the Robot

Everything except `bpx_node` can be developed and tested now:

| Module | Can test now? | How |
|---|---|---|
| Whisper STT | ✅ | Any microphone + PC/Jetson |
| Command parser | ✅ | Unit tests or `ros2 topic pub` |
| Face DB | ✅ | Any webcam |
| Face recognizer | ✅ | Any webcam |
| YOLOv8 detector | ✅ | Any camera |
| Behavior logic | ✅ | Mock the `/bpx/*` topics |
| RTAB-Map VSLAM | ✅ | Any stereo camera or bag file |
| Head servo control | ✅ | PCA9685 + servos independently |
| BPX driver | ❌ | Needs robot |

---

## Hardware

See [hardware/head_design.md](hardware/head_design.md) for the full BOM,
wiring diagram, and 3D print specifications for the articulated head.

**Quick summary:** 3S LiPo → 5V buck (Jetson) + 7.4V buck (servos).
PCA9685 over I²C drives pan/tilt servos. Waveshare stereo camera on CSI.
ReSpeaker USB mic array. Total head weight target < 600 g.
