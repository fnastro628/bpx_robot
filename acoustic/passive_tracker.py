#!/usr/bin/env python3
"""
CAP 5 — Passive Person Tracking

Tracks sound source direction continuously — not just during speech.
Acts as a fallback head-tracking signal when the camera has no person detection.

Priority (highest wins):
  1. Vision: /perception/person_detections present → vision controls head
  2. Acoustic: energy above threshold AND vision absent → acoustic controls head

Publishes /head/pan_deg only when vision tracking is inactive.

PASS test:
  Cover cameras → head smoothly follows footsteps / ambient person sounds
  Walk into camera view → head switches to vision (no jitter at handoff)
  Dark room → acoustic tracking active throughout
"""

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool, String

VISION_TIMEOUT_SEC = 0.5   # treat vision as absent after this silence


class PassiveTracker(Node):
    def __init__(self):
        super().__init__("passive_tracker")

        self.declare_parameter("passive_energy_threshold", 0.05)
        self.declare_parameter("doa_smoothing",            0.85)  # heavier than DOA tracker
        self.declare_parameter("default_pan_deg",          0.0)

        self._threshold = self.get_parameter("passive_energy_threshold").value
        self._alpha     = self.get_parameter("doa_smoothing").value
        self._default   = self.get_parameter("default_pan_deg").value

        self._pan_pub = self.create_publisher(Float32, "/head/pan_deg", 5)

        self._raw_doa        = self._default
        self._smooth_pan     = self._default
        self._energy         = 0.0
        self._last_vision_t  = 0.0   # epoch seconds of last vision detection

        self.create_subscription(Float32, "/acoustic/doa",    self._on_doa,    10)
        self.create_subscription(Float32, "/acoustic/energy", self._on_energy, 10)
        self.create_subscription(
            String, "/perception/person_detections", self._on_vision, 1
        )

        self.create_timer(0.05, self._update)   # 20 Hz
        self.get_logger().info("Passive tracker ready.")

    def _on_doa(self, msg: Float32):
        raw = msg.data
        if raw > 180.0:
            raw -= 360.0
        self._raw_doa = raw

    def _on_energy(self, msg: Float32):
        self._energy = msg.data

    def _on_vision(self, msg: String):
        self._last_vision_t = time.monotonic()

    def _vision_active(self) -> bool:
        return (time.monotonic() - self._last_vision_t) < VISION_TIMEOUT_SEC

    def _update(self):
        if self._vision_active():
            return   # vision has priority — do not publish

        if self._energy < self._threshold:
            return   # too quiet to track

        self._smooth_pan = (
            self._alpha * self._smooth_pan + (1.0 - self._alpha) * self._raw_doa
        )
        msg = Float32()
        msg.data = self._smooth_pan
        self._pan_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PassiveTracker()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
