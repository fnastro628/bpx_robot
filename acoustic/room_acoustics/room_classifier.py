#!/usr/bin/env python3
"""
CAP 10/11 — Room Classifier + Within-Room Position

ROS2 node that listens for a bark trigger, measures the RIR via RIRExtractor,
and publishes two topics:

  /acoustic/room_id       std_msgs/String (JSON)
    {"room": "living_room", "confidence": 0.92, "timestamp": 1234.5}

  /acoustic/room_position std_msgs/String (JSON)  — CAP 11 extension
    {"room": "living_room", "grid_cell": "C2", "confidence": 0.74,
     "position_est_m": [1.2, 0.8], "timestamp": 1234.5}

Bark triggering:
  - Automatic: every bark_interval_sec (default 60s) when IDLE
  - On demand:  ros2 topic pub /acoustic/bark_trigger std_msgs/Bool "data: true"

Classifier:
  - k-NN (k=3) with cosine distance on the 15-d feature vector.
  - Falls back to VSLAM-derived room estimate when confidence < 0.70.

CAP 11 — within-room grid positions:
  After calibration the DB holds per-room grid cells (e.g. 3×3 = 9 cells).
  Each cell is a separate "room" entry named "living_room/A1", "living_room/B2" etc.
  The classifier identifies both room and cell from the same feature vector.
"""

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String, UInt8

from acoustic.room_acoustics.bark_signal import BarkSignal
from acoustic.room_acoustics.rir_extractor import RIRExtractor
from acoustic.room_acoustics.room_db import RoomDatabase
from behaviors.gait_map import MotionState

BARK_INTERVAL_SEC    = 60.0
CONFIDENCE_THRESHOLD = 0.70
KNN_K                = 3


class RoomClassifier(Node):
    def __init__(self):
        super().__init__("room_classifier")

        self.declare_parameter("bark_interval_sec",    BARK_INTERVAL_SEC)
        self.declare_parameter("confidence_threshold", CONFIDENCE_THRESHOLD)
        self.declare_parameter("db_path",              "")

        interval   = self.get_parameter("bark_interval_sec").value
        self._thr  = self.get_parameter("confidence_threshold").value
        db_path    = self.get_parameter("db_path").value or None

        self._db    = RoomDatabase(db_path) if db_path else RoomDatabase()
        self._bark  = BarkSignal()
        self._rx    = RIRExtractor(self._bark)

        self._room_pub = self.create_publisher(String, "/acoustic/room_id",       10)
        self._pos_pub  = self.create_publisher(String, "/acoustic/room_position",  10)

        self._motion_state  = MotionState.LYING_DOWN
        self._vslam_room    = ""

        self.create_subscription(UInt8,   "/bpx/motion_state",    self._on_state,       10)
        self.create_subscription(Bool,    "/acoustic/bark_trigger", self._on_trigger,    10)
        self.create_subscription(String,  "/vslam/current_room",  self._on_vslam_room,  10)

        self._idle_timer = self.create_timer(interval, self._auto_bark)

        self.get_logger().info(
            f"Room classifier ready — {len(self._db.list_rooms())} rooms in DB."
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_state(self, msg: UInt8):
        self._motion_state = msg.data

    def _on_trigger(self, msg: Bool):
        if msg.data:
            self._classify()

    def _on_vslam_room(self, msg: String):
        self._vslam_room = msg.data

    # ── Bark triggering ───────────────────────────────────────────────────────

    def _auto_bark(self):
        idle = self._motion_state in (
            MotionState.LYING_DOWN, MotionState.SIT_DOWN, MotionState.PASSIVE
        )
        if idle and self._db.count_vectors() > 0:
            self._classify()

    # ── Classification ────────────────────────────────────────────────────────

    def _classify(self):
        self.get_logger().info("Barking for room classification...")
        try:
            features = self._rx.measure()
        except Exception as e:
            self.get_logger().error(f"RIR measurement failed: {e}")
            return

        room, confidence = self._knn_classify(features)

        if room is None or confidence < self._thr:
            room       = self._vslam_room or "unknown"
            confidence = 0.0
            self.get_logger().info(
                f"Low confidence ({confidence:.2f}) — using VSLAM room: {room}"
            )

        # Parse room/cell if CAP 11 grid notation (e.g. "living_room/B2")
        if "/" in room:
            room_name, cell = room.rsplit("/", 1)
            self._publish_room(room_name, confidence)
            self._publish_position(room_name, cell, confidence)
        else:
            self._publish_room(room, confidence)

        self.get_logger().info(f"Room: {room} ({confidence:.2f})")

    def _knn_classify(self, features: np.ndarray) -> tuple[str | None, float]:
        """k-NN cosine similarity over all stored feature vectors."""
        with self._db._conn() as con:
            rows = con.execute("SELECT name, feature_blob FROM rooms").fetchall()

        if not rows:
            return None, 0.0

        q = RoomDatabase._normalize(features)
        scored = []
        for name, blob in rows:
            stored = np.frombuffer(blob, dtype=np.float32)
            sim    = float(np.dot(q, RoomDatabase._normalize(stored)))
            scored.append((sim, name))

        scored.sort(reverse=True)
        top_k = scored[:KNN_K]

        # Weighted vote: weight = similarity score
        votes: dict[str, float] = {}
        for sim, name in top_k:
            votes[name] = votes.get(name, 0.0) + sim

        best = max(votes, key=votes.__getitem__)
        confidence = votes[best] / sum(votes.values())

        return best, float(confidence)

    # ── Publishers ────────────────────────────────────────────────────────────

    def _publish_room(self, room: str, confidence: float):
        msg      = String()
        msg.data = json.dumps({
            "room":       room,
            "confidence": round(confidence, 3),
            "timestamp":  time.time(),
        })
        self._room_pub.publish(msg)

    def _publish_position(self, room: str, cell: str, confidence: float):
        msg      = String()
        msg.data = json.dumps({
            "room":       room,
            "grid_cell":  cell,
            "confidence": round(confidence, 3),
            "timestamp":  time.time(),
        })
        self._pos_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RoomClassifier()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
