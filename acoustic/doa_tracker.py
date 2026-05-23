#!/usr/bin/env python3
"""
CAP 2 — DOA Head Tracking

Subscribes to /acoustic/doa and /acoustic/vad.
When voice is active, commands the head pan servo to face the speaker.
Returns to default (0°) after a configurable silence timeout.

Publishes:
  /head/pan_deg   std_msgs/Float32   (consumed by head_controller.py)

PASS test:
  Speak from left  → head pans left
  Speak from right → head pans right
  Stop speaking    → head returns to 0° after doa_return_timeout_sec
  Camera covered   → still tracks by sound alone
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool


class DoaTracker(Node):
    def __init__(self):
        super().__init__("doa_tracker")

        self.declare_parameter("doa_smoothing",          0.6)
        self.declare_parameter("doa_return_timeout_sec", 3.0)
        self.declare_parameter("default_pan_deg",        0.0)

        self._alpha   = self.get_parameter("doa_smoothing").value
        self._timeout = self.get_parameter("doa_return_timeout_sec").value
        self._default = self.get_parameter("default_pan_deg").value

        self._pan_pub = self.create_publisher(Float32, "/head/pan_deg", 10)

        self._vad_active   = False
        self._last_vad_t   = self.get_clock().now()
        self._smooth_pan   = self._default
        self._raw_doa      = self._default

        self.create_subscription(Float32, "/acoustic/doa", self._on_doa, 10)
        self.create_subscription(Bool,    "/acoustic/vad", self._on_vad, 10)

        # Update at 20 Hz
        self.create_timer(0.05, self._update)
        self.get_logger().info("DOA tracker ready.")

    def _on_doa(self, msg: Float32):
        # Convert 0–360° to ±180° (right = negative for servo convention)
        raw = msg.data
        if raw > 180.0:
            raw -= 360.0
        self._raw_doa = raw

    def _on_vad(self, msg: Bool):
        self._vad_active = msg.data
        if msg.data:
            self._last_vad_t = self.get_clock().now()

    def _update(self):
        now     = self.get_clock().now()
        silence = (now - self._last_vad_t).nanoseconds * 1e-9

        if self._vad_active or silence < self._timeout:
            target = self._raw_doa
        else:
            target = self._default

        # Exponential moving average smoothing
        self._smooth_pan = (
            self._alpha * self._smooth_pan + (1.0 - self._alpha) * target
        )

        msg = Float32()
        msg.data = self._smooth_pan
        self._pan_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DoaTracker()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
