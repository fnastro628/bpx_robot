#!/usr/bin/env python3
"""
CAP 4 — Clap Commands

Detects hand-clap patterns from /acoustic/energy and /acoustic/vad.
An impulse is a sharp energy spike that is NOT preceded by sustained VAD
(distinguishes claps from consonants in speech).

Clap patterns → /commands (JSON):
  1 clap  → {"type": "come"}
  2 claps → {"type": "stand"} / {"type": "sit"}  (toggled)
  3 claps → {"type": "stop"}

PASS test:
  Single clap at 3 m → robot executes "come"
  Double clap        → stand/sit toggle
  Triple clap        → stop
  Speech does NOT trigger clap detection (VAD gate)
"""

import json
import time
import collections

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool, String


class ClapDetector(Node):
    def __init__(self):
        super().__init__("clap_detector")

        self.declare_parameter("clap_energy_threshold", 0.35)
        self.declare_parameter("clap_window_sec",       1.0)
        self.declare_parameter("impulse_min_rise",      3.0)   # energy ratio rise

        self._threshold  = self.get_parameter("clap_energy_threshold").value
        self._window     = self.get_parameter("clap_window_sec").value
        self._min_rise   = self.get_parameter("impulse_min_rise").value

        self._cmd_pub = self.create_publisher(String, "/commands", 10)

        self._vad       = False
        self._energy    = 0.0
        self._prev_e    = 0.0
        self._clap_times: collections.deque = collections.deque()
        self._stand_state = False   # for stand/sit toggle

        self.create_subscription(Bool,    "/acoustic/vad",    self._on_vad,    10)
        self.create_subscription(Float32, "/acoustic/energy", self._on_energy, 10)

        self.get_logger().info("Clap detector ready.")

    def _on_vad(self, msg: Bool):
        self._vad = msg.data

    def _on_energy(self, msg: Float32):
        e = msg.data
        rise = e / (self._prev_e + 1e-6)
        self._prev_e = e

        # Impulse: energy above threshold AND rising sharply AND not speech
        if e > self._threshold and rise > self._min_rise and not self._vad:
            self._register_clap()

        self._energy = e

    def _register_clap(self):
        now = time.monotonic()
        self._clap_times.append(now)

        # Flush old claps outside the window
        cutoff = now - self._window
        while self._clap_times and self._clap_times[0] < cutoff:
            self._clap_times.popleft()

        count = len(self._clap_times)

        # Wait briefly for more claps before acting (debounce via timer)
        # We use a simple approach: act on count after a stable gap
        # The timer below fires 0.8 s after the first clap in the window
        if count == 1:
            self.create_timer(self._window * 0.8, self._evaluate)

    def _evaluate(self):
        now    = time.monotonic()
        cutoff = now - self._window
        while self._clap_times and self._clap_times[0] < cutoff:
            self._clap_times.popleft()

        count = len(self._clap_times)
        self._clap_times.clear()

        if count == 1:
            self._publish({"type": "come"})
            self.get_logger().info("CLAP ×1 → come")
        elif count == 2:
            self._stand_state = not self._stand_state
            cmd = "stand" if self._stand_state else "sit"
            self._publish({"type": cmd})
            self.get_logger().info(f"CLAP ×2 → {cmd}")
        elif count >= 3:
            self._publish({"type": "stop"})
            self.get_logger().info("CLAP ×3 → stop")

    def _publish(self, cmd: dict):
        msg = String()
        msg.data = json.dumps(cmd)
        self._cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ClapDetector()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
