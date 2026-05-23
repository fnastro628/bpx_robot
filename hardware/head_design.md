# BPX Articulated Head — Hardware Design

## Overview

A pan/tilt head mounted at the front of the BPX chassis that houses:
- Jetson Orin Nano Super (8 GB)
- Waveshare dual/stereo camera
- Microphone array
- Speaker
- Head battery pack
- Servo driver board

The head moves independently of the body, giving the robot gaze control
and allowing the stereo camera to track targets without requiring full-body turns.

---

## Mechanical Design

### Pan/Tilt Mechanism

```
         ┌─────────────────────┐
         │   Camera Module     │  ← Waveshare stereo (left + right)
         │   [ L ]     [ R ]  │
         └────────┬────────────┘
                  │ tilt axis (pitch) ← Servo 2
         ┌────────┴────────────┐
         │    Head Shell       │  ← 3D printed PETG/ASA
         │  [Jetson Orin Nano] │
         │  [Mic Array]        │
         │  [Speaker]          │
         │  [PCA9685]          │
         └────────┬────────────┘
                  │ pan axis (yaw) ← Servo 1
         ┌────────┴────────────┐
         │   Neck Bracket      │  ← bolts to BPX chassis top plate
         └─────────────────────┘
```

### Degrees of Freedom

| Axis | Range    | Servo               | Notes                    |
|------|----------|---------------------|--------------------------|
| Pan  | ±90°     | Dynamixel XL430 or  | Geared, high-torque      |
| Tilt | −20/+40° | standard DS3218MG   | Carry camera + cables    |

**Recommended servos:**
- Pan:  Dynamixel XL430-W250-T (TTL, 2.5 N·m, feedback) — ideal, allows position feedback
- Tilt: DS3218MG 20kg standard servo — adequate and cheaper
- Alternative budget: two MG996R for both axes

### Servo Driver
- **Adafruit PCA9685** 16-channel I²C PWM driver
- Connected to Jetson via I²C (pins 3/5 on 40-pin header)
- Provides 12-bit resolution, up to 1526 Hz PWM

---

## Electronics

### Power Architecture

```
  Head LiPo Pack (3S 11.1V, 5000 mAh)
       │
       ├──► Buck converter 1 (5V 5A) ──► Jetson Orin Nano Super (barrel)
       ├──► Buck converter 2 (7.4V 3A) ──► Servos (via PCA9685 V+)
       └──► 3.3V LDO ──► PCA9685 VCC, misc logic
```

**Recommended battery:** CNHL 3S 5000mAh 50C LiPo  
**Runtime estimate:** ~2–3 hours (Jetson ~7–10W typical, servos ~2W idle)

### Component BOM

| Component | Part | Notes |
|---|---|---|
| SBC | NVIDIA Jetson Orin Nano Super 8 GB | Main compute |
| Camera | Waveshare IMX219-83 Stereo Camera | 60 mm baseline, CSI |
| Servo driver | Adafruit PCA9685 | I²C, 16ch |
| Pan servo | Dynamixel XL430-W250-T | Optional: MG996R |
| Tilt servo | DS3218MG | 20 kg·cm |
| Microphone | reSpeaker XVF3800 USB 4-Mic Array | 4-mic circular, USB + I2S, onboard DOA/AEC/BF (XMOS XVF3800) — current gen |
| Speaker | 5W 4Ω 78mm speaker + PAM8403 amp | I²S or USB audio |
| Battery | 3S 5000mAh 50C LiPo | Head-mounted |
| Buck 5V | Pololu D24V50F5 (5V 5A) | Jetson power |
| Buck 7.4V | Pololu D24V50F7 (7.4V 5A) | Servo power |
| XT30 connector | XT30 male/female | Battery disconnect |
| NVMe SSD | Samsung 980 256 GB M.2 2242 | Map/model storage |

---

## Wiring

```
Jetson 40-pin Header
  Pin 3  (I²C SDA) ──► PCA9685 SDA
  Pin 5  (I²C SCL) ──► PCA9685 SCL
  Pin 2  (5V)      ──► PCA9685 VCC
  Pin 6  (GND)     ──► PCA9685 GND

PCA9685
  V+  (servo power) ──► Buck 7.4V output
  GND                ──► common ground
  Ch0 PWM ──► Pan  servo signal
  Ch1 PWM ──► Tilt servo signal

CSI Camera Ribbon Cables
  CAM0 (22-pin) ──► Waveshare left  camera
  CAM1 (22-pin) ──► Waveshare right camera

USB
  USB3 port 1 ──► reSpeaker XVF3800 (or I2S via 40-pin header pins 12/35/38/40)
  USB3 port 2 ──► PAM8403 USB audio (or I²S via pins 12/35/38/40)
```

---

## 3D Print Specs

- **Material:** PETG (heat tolerance) or ASA (outdoor)
- **Layer height:** 0.2 mm
- **Infill:** 40% gyroid
- **Key dimensions:**
  - Jetson board: 100 × 79 mm — leave 10 mm clearance all sides
  - Camera mount: 60 mm center-to-center for left/right lenses
  - Head total height (estimate): ~140 mm
  - Total weight target: < 600 g (including Jetson + battery)

**Files to design (FreeCAD / Fusion 360):**
1. `head_shell.stl` — main enclosure with Jetson slot
2. `camera_bracket.stl` — holds stereo camera, bolts to tilt servo horn
3. `tilt_yoke.stl` — connects pan servo horn to head shell
4. `neck_base.stl` — pan servo mount, bolts to BPX top plate

---

## Mounting to BPX Chassis

- BPX top plate has M3 mounting holes — verify pattern before printing neck base
- Use M3 × 8 mm socket head screws
- Route Ethernet or WiFi to main body (Jetson has M.2 WiFi slot or USB WiFi adapter)
- Consider a slip ring for continuous pan rotation if full ±360° rotation is desired

---

## Software Interface

Pan/tilt is controlled by [head_controller.py](../behaviors/head_controller.py):
```python
# Example
head.set_pan_deg(30)    # look right
head.set_tilt_deg(-10)  # look slightly down
head.track_point(cx_norm, cy_norm)  # visual servoing
```

The PCA9685 driver uses `adafruit-circuitpython-pca9685` (pip installable on Jetson).
