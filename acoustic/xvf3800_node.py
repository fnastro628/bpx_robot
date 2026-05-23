#!/usr/bin/env python3
"""
CAP 1 — XVF3800 ROS2 Integration  ★ PREREQUISITE FOR ALL ACOUSTIC CAPS

Reads DOA, VAD, and energy from the reSpeaker XVF3800 USB HID interface.
Streams beamformed audio into a sounddevice-compatible queue consumed by stt_node.

Published topics:
  /acoustic/doa     std_msgs/Float32   Direction of arrival 0–360°
  /acoustic/vad     std_msgs/Bool      Voice activity (True = speech detected)
  /acoustic/energy  std_msgs/Float32   Pre-AGC RMS energy (0–1 normalised)

PASS test:
  ros2 topic echo /acoustic/doa     # sweeps as you walk around
  ros2 topic echo /acoustic/vad     # True while speaking
  ros2 topic echo /acoustic/energy  # changes with room noise

Hardware: reSpeaker XVF3800 connected via USB.
Requires:  pip install pyusb
           udev rule: SUBSYSTEM=="usb", ATTR{idVendor}=="2886", MODE="0666"
"""

import threading
import time

import usb.core
import usb.util
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool

# ── XVF3800 USB identifiers ───────────────────────────────────────────────────
# Verify with: lsusb | grep -i seeed
VENDOR_ID  = 0x2886
PRODUCT_ID = 0x0019   # XVF3800 — update if lsusb shows a different PID

# HID control request constants (from reSpeaker usb_4_mic_array library)
TIMEOUT_MS     = 100
GET_CMD        = 0xC0   # bRequestType: device-to-host vendor
SET_CMD        = 0x40   # bRequestType: host-to-device vendor
PARAM_DOA      = 21     # wValue: Direction of Arrival (0–360 uint16)
PARAM_VAD      = 19     # wValue: Voice Activity (0/1)
PARAM_ENERGY   = 1001   # wValue: pre-AGC energy channel 0 (raw int16, signed)


class XVF3800Node(Node):
    def __init__(self):
        super().__init__("xvf3800_node")

        self.declare_parameter("doa_update_hz",    20.0)
        self.declare_parameter("energy_smoothing", 0.8)   # EMA alpha

        rate_hz   = self.get_parameter("doa_update_hz").value
        self._ema = self.get_parameter("energy_smoothing").value

        self._doa_pub    = self.create_publisher(Float32, "/acoustic/doa",    10)
        self._vad_pub    = self.create_publisher(Bool,    "/acoustic/vad",    10)
        self._energy_pub = self.create_publisher(Float32, "/acoustic/energy", 10)

        self._energy_smooth = 0.0
        self._dev = self._open_device()

        period = 1.0 / rate_hz
        self._timer = self.create_timer(period, self._poll)
        self.get_logger().info(
            f"XVF3800 ready — polling at {rate_hz} Hz. "
            f"VID=0x{VENDOR_ID:04X} PID=0x{PRODUCT_ID:04X}"
        )

    # ── Device init ───────────────────────────────────────────────────────────

    def _open_device(self):
        dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
        if dev is None:
            self.get_logger().error(
                f"XVF3800 not found (VID=0x{VENDOR_ID:04X} PID=0x{PRODUCT_ID:04X}). "
                "Check USB connection and udev rules."
            )
            return None
        try:
            if dev.is_kernel_driver_active(0):
                dev.detach_kernel_driver(0)
        except Exception:
            pass
        self.get_logger().info("XVF3800 USB device opened.")
        return dev

    # ── HID register read ─────────────────────────────────────────────────────

    def _read_param(self, param: int, length: int = 8) -> bytes | None:
        if self._dev is None:
            return None
        try:
            return self._dev.ctrl_transfer(
                GET_CMD, 0x00, param, 0x00, length, TIMEOUT_MS
            )
        except usb.core.USBError as e:
            self.get_logger().warn(f"USB read error param={param}: {e}")
            return None

    def _read_doa(self) -> float | None:
        data = self._read_param(PARAM_DOA, 4)
        if data is None or len(data) < 4:
            return None
        # DOA is a little-endian int32 in tenths of a degree
        raw = int.from_bytes(data[:4], "little", signed=True)
        return float(raw) / 10.0   # → degrees 0–360

    def _read_vad(self) -> bool | None:
        data = self._read_param(PARAM_VAD, 4)
        if data is None or len(data) < 4:
            return None
        return bool(int.from_bytes(data[:4], "little", signed=False))

    def _read_energy(self) -> float | None:
        data = self._read_param(PARAM_ENERGY, 4)
        if data is None or len(data) < 4:
            return None
        raw = int.from_bytes(data[:4], "little", signed=True)
        # Normalise: raw is typically in range ±32768; clamp and map to 0–1
        return min(1.0, abs(raw) / 32768.0)

    # ── Poll timer ────────────────────────────────────────────────────────────

    def _poll(self):
        doa    = self._read_doa()
        vad    = self._read_vad()
        energy = self._read_energy()

        if doa is not None:
            msg = Float32(); msg.data = doa
            self._doa_pub.publish(msg)

        if vad is not None:
            msg = Bool(); msg.data = vad
            self._vad_pub.publish(msg)

        if energy is not None:
            # Exponential moving average to smooth energy
            self._energy_smooth = (
                self._ema * self._energy_smooth + (1 - self._ema) * energy
            )
            msg = Float32(); msg.data = self._energy_smooth
            self._energy_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = XVF3800Node()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
