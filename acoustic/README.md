# Acoustic Capabilities — XVF3800 Pipeline

All 15 acoustic capabilities enabled by the reSpeaker XVF3800 USB 4-mic array.

---

## Hardware

| Component | Detail |
|---|---|
| Mic array | reSpeaker XVF3800 USB 4-mic circular array |
| Chip | XMOS XVF3800 |
| Interface | USB (HID control + audio stream) |
| Onboard DSP | DOA, AEC, beamforming, NS, VAD, AGC |
| Speaker | USB or 3.5mm output device for bark signal |

---

## Capability Map

| # | Capability | File | PASS Test |
|---|---|---|---|
| 1 | XVF3800 ROS2 driver | `xvf3800_node.py` | `/acoustic/doa` changes with movement |
| 2 | DOA head tracking | `doa_tracker.py` + `head_controller.py` | Head follows voice in the dark |
| 3 | Enhanced STT | `speech/stt_node.py` (modified) | Far-field transcription at 4 m |
| 4 | Clap commands | `clap_detector.py` | 1/2/3 claps → come/stand-sit/stop |
| 5 | Passive person tracking | `passive_tracker.py` | Head tracks footsteps when camera off |
| 6 | Proximity estimation | `proximity_estimator.py` | < 1 m at 0.5 m, > 2 m at 3 m |
| 7 | Paging ("find me") | `paging.py` | Robot navigates to kitchen when called |
| 8 | Acoustic event detection | `event_detector.py` | Glass break / alarm detected within 1 s |
| 9 | Active bark fingerprinting | `room_acoustics/bark_signal.py` + `rir_extractor.py` | Distinct RIR per room |
| 10 | Room identification | `room_acoustics/room_classifier.py` + `room_db.py` | ≥ 90% accuracy after calibration |
| 11 | Position within room | `room_acoustics/room_classifier.py` (grid mode) | Correct cell ≥ 70% of time |
| 12 | Speaker location memory | `speaker_memory.py` | Morning → kitchen predicted at 07:00 |
| 13 | Emotional tone | `emotion_detector.py` | Raised voice → "stressed" |
| 14 | Acoustic SLAM landmarks | `acoustic_slam.py` | TV registered at correct map position |
| 15 | Selective beamforming | `acoustic_slam.py` | Beam lock suppresses off-axis speech |

---

## ROS2 Topics

| Topic | Type | Producer |
|---|---|---|
| `/acoustic/doa` | Float32 | xvf3800_node |
| `/acoustic/vad` | Bool | xvf3800_node |
| `/acoustic/energy` | Float32 | xvf3800_node |
| `/acoustic/event` | String (JSON) | event_detector |
| `/acoustic/speaker_position` | String (JSON) | proximity_estimator |
| `/acoustic/room_id` | String (JSON) | room_classifier |
| `/acoustic/room_position` | String (JSON) | room_classifier |
| `/acoustic/predicted_location` | String (JSON) | speaker_memory |
| `/acoustic/emotion` | String (JSON) | emotion_detector |
| `/acoustic/landmark` | String (JSON) | acoustic_slam |
| `/acoustic/beam_target_deg` | Float32 | (external command) |
| `/head/pan_deg` | Float32 | doa_tracker / passive_tracker |
| `/head/tilt_deg` | Float32 | (vision node) |

### Services
| Service | Type | Node |
|---|---|---|
| `/acoustic/lock_beam` | std_srvs/SetBool | acoustic_slam |
| `/acoustic/bark_trigger` | (topic: Bool) | room_classifier |

---

## Quick Start

```bash
# Full system with acoustic pipeline
ros2 launch bpx_robot bpx_full.launch.py

# Acoustic only (no robot required)
ros2 run bpx_robot acoustic/xvf3800_node.py
ros2 run bpx_robot acoustic/event_detector.py

# Room calibration (standalone, no ROS2 needed)
python acoustic/room_acoustics/calibrator.py --room living_room --barks 7
python acoustic/room_acoustics/calibrator.py --list

# Grid calibration (CAP 11 — within-room position)
python acoustic/room_acoustics/calibrator.py --room living_room --grid 3x3

# Test RIR measurement and plot
python acoustic/room_acoustics/rir_extractor.py --plot

# Selective beamforming
ros2 topic pub /acoustic/beam_target_deg std_msgs/Float32 "data: 90.0"
ros2 service call /acoustic/lock_beam std_srvs/SetBool "data: true"
```

---

## Test Checklist

Run each test in sequence. Each depends on the previous capability working.

- [ ] **CAP 1** — `ros2 topic echo /acoustic/doa` changes as you walk around
- [ ] **CAP 1** — `ros2 topic echo /acoustic/vad` True when speaking
- [ ] **CAP 1** — `ros2 topic echo /acoustic/energy` non-zero, changes with noise
- [ ] **CAP 2** — Head pans left/right with voice; returns to centre after 3s silence
- [ ] **CAP 4** — Single clap from 3 m → robot comes
- [ ] **CAP 4** — Double clap → stand/sit toggle
- [ ] **CAP 4** — Triple clap → stop
- [ ] **CAP 5** — Cover cameras → head follows footsteps
- [ ] **CAP 6** — 0.5 m speech → `distance_est_m` < 1.0
- [ ] **CAP 6** — 3 m speech → `distance_est_m` > 2.0
- [ ] **CAP 7** — Call from kitchen → robot navigates there
- [ ] **CAP 8** — Play glass break → event within 1 s
- [ ] **CAP 8** — Play smoke alarm → alarm event
- [ ] **CAP 8** — Normal speech → no event triggered
- [ ] **CAP 9** — `rir_extractor.py --plot` shows distinct shapes per room
- [ ] **CAP 9** — T60 bathroom < 0.3 s; bedroom 0.3–0.5 s; living room 0.5–0.8 s
- [ ] **CAP 10** — Blind test: bark in each room → correct ID ≥ 90% across 20 barks
- [ ] **CAP 11** — Bark at 4 positions → correct grid cell ≥ 70%
- [ ] **CAP 12** — After 2 mornings, `/acoustic/predicted_location` shows kitchen at 07:xx
- [ ] **CAP 13** — Calm speech → "neutral"; raised voice → "stressed"
- [ ] **CAP 14** — TV on → landmark at correct map position after 3 visits
- [ ] **CAP 15** — Beam lock → off-axis speech suppressed

---

## Dependencies

```
pyusb>=1.2.1
openwakeword>=0.6.0
tflite-runtime>=2.14.0
speechbrain>=1.0.0
onnxruntime>=1.17.0
adafruit-circuitpython-pca9685>=2.4.1
adafruit-blinka>=8.0.0
```

Model files (download to `~/.bpx/`):
- `yamnet.tflite` — [TF Hub YAMNet](https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1)
- `yamnet_class_map.csv` — from same TF Hub page
- `emotion_model.onnx` — export from SpeechBrain `emotion-recognition-wav2vec2-IEMOCAP`
