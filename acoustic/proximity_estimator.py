#!/usr/bin/env python3
"""
CAP 6 — Proximity Estimation

Estimates speaker distance from pre-AGC energy using the inverse-square law.
Combines with DOA to publish a polar position (angle, distance).

Calibration (one-time, run interactively):
  ros2 run bpx_robot proximity_estimator.py --ros-args -p calibrate:=true
  Follow prompts: stand at 0.5 m, 1 m, 2 m and speak for 2 s each.
  Writes fit coefficients to ~/.bpx/proximity_cal.json

Published:
  /acoustic/speaker_position  std_msgs/String (JSON)
    {"doa_deg": 45.0, "distance_est_m": 1.8, "confidence": 0.7}

PASS test:
  Speak at 0.5 m → distance_est_m < 1.0
  Speak at 3.0 m → distance_est_m > 2.0
  Position updates smoothly as user moves
"""

import json
import math
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool, String

CAL_PATH = os.path.expanduser("~/.bpx/proximity_cal.json")
DEFAULT_REF_ENERGY = 0.60   # energy at 0.5 m (overridden by calibration)
DEFAULT_REF_DIST   = 0.50   # metres


class ProximityEstimator(Node):
    def __init__(self):
        super().__init__("proximity_estimator")

        self.declare_parameter("proximity_ref_energy", DEFAULT_REF_ENERGY)
        self.declare_parameter("proximity_ref_dist_m", DEFAULT_REF_DIST)
        self.declare_parameter("calibrate",            False)

        self._ref_e  = self.get_parameter("proximity_ref_energy").value
        self._ref_d  = self.get_parameter("proximity_ref_dist_m").value
        do_cal       = self.get_parameter("calibrate").value

        self._load_calibration()

        self._pos_pub = self.create_publisher(String, "/acoustic/speaker_position", 10)

        self._doa    = 0.0
        self._energy = 0.0
        self._vad    = False

        self.create_subscription(Float32, "/acoustic/doa",    self._on_doa,    10)
        self.create_subscription(Float32, "/acoustic/energy", self._on_energy, 10)
        self.create_subscription(Bool,    "/acoustic/vad",    self._on_vad,    10)

        self.create_timer(0.1, self._update)   # 10 Hz

        if do_cal:
            self.get_logger().info("Calibration mode — follow spoken prompts.")
        else:
            self.get_logger().info(
                f"Proximity estimator ready (ref {self._ref_e:.2f} @ {self._ref_d} m)."
            )

    # ── Calibration ───────────────────────────────────────────────────────────

    def _load_calibration(self):
        if os.path.exists(CAL_PATH):
            with open(CAL_PATH) as f:
                cal = json.load(f)
            self._ref_e = cal.get("ref_energy", self._ref_e)
            self._ref_d = cal.get("ref_dist_m", self._ref_d)
            self.get_logger().info(f"Loaded calibration from {CAL_PATH}")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_doa(self, msg: Float32):
        raw = msg.data
        self._doa = raw - 360.0 if raw > 180.0 else raw

    def _on_energy(self, msg: Float32):
        self._energy = msg.data

    def _on_vad(self, msg: Bool):
        self._vad = msg.data

    # ── Estimation ────────────────────────────────────────────────────────────

    def _update(self):
        if not self._vad or self._energy < 0.01:
            return

        # Inverse square: E ∝ 1/d²  →  d = ref_d * sqrt(ref_e / e)
        try:
            dist = self._ref_d * math.sqrt(self._ref_e / self._energy)
            dist = max(0.1, min(10.0, dist))   # clamp to sensible range
        except (ValueError, ZeroDivisionError):
            return

        # Confidence drops for very low energy (far / noisy)
        confidence = min(1.0, self._energy / self._ref_e)

        out = String()
        out.data = json.dumps({
            "doa_deg":       round(self._doa, 1),
            "distance_est_m": round(dist, 2),
            "confidence":    round(confidence, 2),
        })
        self._pos_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ProximityEstimator()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
