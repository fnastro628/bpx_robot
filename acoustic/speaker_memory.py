#!/usr/bin/env python3
"""
CAP 12 — Speaker Location Memory

Logs where speech occurs (room, robot pose, DOA) to a SQLite heatmap, keyed
by day-of-week and hour. After sufficient data, publishes a predicted location
at the start of each new hour so the robot can pre-position itself.

Subscriptions:
  /acoustic/doa           std_msgs/Float32  — current DOA (0-360°)
  /acoustic/vad           std_msgs/Bool     — True when someone is speaking
  /acoustic/room_id       std_msgs/String   — current room JSON
  /bpx/odometry           nav_msgs/Odometry — robot pose in map frame

Published:
  /acoustic/predicted_location  std_msgs/String (JSON)
    {"room": "kitchen", "doa_deg": 45.0, "hour": 7, "day": 0,
     "confidence": 0.82, "timestamp": 1234.5}

Prediction is published once at the top of each hour (when enough history
exists) and on demand via /acoustic/predict_now (std_msgs/Bool trigger).

PASS test:
  - Speak from kitchen for two mornings → DB shows kitchen cluster at ~07:00
  - ros2 topic echo /acoustic/predicted_location shows kitchen at 07:xx
  - Robot pre-positions near kitchen before being paged
"""

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from nav_msgs.msg import Odometry

DB_PATH = Path.home() / ".bpx" / "speaker_memory.db"
MIN_SAMPLES_TO_PREDICT = 10    # require at least this many log entries to predict


class SpeakerMemory(Node):
    def __init__(self):
        super().__init__("speaker_memory")

        self.declare_parameter("db_path", str(DB_PATH))
        self.declare_parameter("min_samples", MIN_SAMPLES_TO_PREDICT)

        db_path          = Path(self.get_parameter("db_path").value)
        self._min_samples = self.get_parameter("min_samples").value

        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

        self._pred_pub = self.create_publisher(
            String, "/acoustic/predicted_location", 10
        )

        self._doa   = 0.0
        self._vad   = False
        self._room  = ""
        self._pose_x = 0.0
        self._pose_y = 0.0
        self._pose_yaw = 0.0

        self.create_subscription(Float32,  "/acoustic/doa",     self._on_doa,    10)
        self.create_subscription(Bool,     "/acoustic/vad",     self._on_vad,    10)
        self.create_subscription(String,   "/acoustic/room_id", self._on_room,   10)
        self.create_subscription(Odometry, "/bpx/odometry",     self._on_odom,   10)
        self.create_subscription(Bool, "/acoustic/predict_now", self._on_predict, 10)

        # Log at 1 Hz when VAD is active; predict at top of each hour
        self.create_timer(1.0,    self._log_tick)
        self.create_timer(60.0,   self._hourly_tick)

        self._last_log_hour = -1
        self.get_logger().info(f"Speaker memory ready — DB: {db_path}")

    # ── Schema ────────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _init_schema(self):
        with self._conn() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS speech_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   REAL    NOT NULL,
                    day_of_week INTEGER NOT NULL,   -- 0=Mon … 6=Sun
                    hour        INTEGER NOT NULL,   -- 0-23
                    room        TEXT    NOT NULL,
                    robot_x     REAL    DEFAULT 0.0,
                    robot_y     REAL    DEFAULT 0.0,
                    doa_deg     REAL    NOT NULL
                )
            """)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_doa(self, msg: Float32):
        raw = msg.data
        self._doa = raw - 360.0 if raw > 180.0 else raw

    def _on_vad(self, msg: Bool):
        self._vad = msg.data

    def _on_room(self, msg: String):
        try:
            self._room = json.loads(msg.data).get("room", "")
        except (json.JSONDecodeError, AttributeError):
            self._room = msg.data

    def _on_odom(self, msg: Odometry):
        self._pose_x = msg.pose.pose.position.x
        self._pose_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # yaw from quaternion
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._pose_yaw = float(np.arctan2(siny, cosy))

    def _on_predict(self, msg: Bool):
        if msg.data:
            self._publish_prediction()

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_tick(self):
        if not self._vad or not self._room:
            return
        now  = time.time()
        lt   = time.localtime(now)
        with self._conn() as con:
            con.execute(
                "INSERT INTO speech_log "
                "(timestamp, day_of_week, hour, room, robot_x, robot_y, doa_deg) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (now, lt.tm_wday, lt.tm_hour, self._room,
                 self._pose_x, self._pose_y, self._doa),
            )

    # ── Prediction ────────────────────────────────────────────────────────────

    def _hourly_tick(self):
        lt = time.localtime()
        if lt.tm_min == 0:   # only at the top of the hour
            self._publish_prediction()

    def _publish_prediction(self):
        lt      = time.localtime()
        dow     = lt.tm_wday
        hour    = lt.tm_hour

        result = self._predict(dow, hour)
        if result is None:
            return

        msg      = String()
        msg.data = json.dumps({
            "room":       result["room"],
            "doa_deg":    round(result["doa_deg"], 1),
            "hour":       hour,
            "day":        dow,
            "confidence": round(result["confidence"], 3),
            "timestamp":  time.time(),
        })
        self._pred_pub.publish(msg)
        self.get_logger().info(
            f"Predicted location: {result['room']} "
            f"(doa={result['doa_deg']:.0f}°, confidence={result['confidence']:.2f})"
        )

    def _predict(self, day_of_week: int, hour: int) -> dict | None:
        """
        Find the most common room+DOA cluster for this hour ± 1 on the same
        day, weighted by recency.  Returns None if too few samples.
        """
        with self._conn() as con:
            rows = con.execute(
                "SELECT room, doa_deg, timestamp FROM speech_log "
                "WHERE day_of_week = ? AND abs(hour - ?) <= 1",
                (day_of_week, hour),
            ).fetchall()

        if len(rows) < self._min_samples:
            return None

        # Count votes per room, weight by recency (exponential decay over 14 days)
        now    = time.time()
        votes: dict[str, float] = {}
        doas:  dict[str, list[float]] = {}
        for room, doa, ts in rows:
            age_days = (now - ts) / 86400.0
            weight   = float(np.exp(-age_days / 14.0))
            votes[room]   = votes.get(room, 0.0) + weight
            doas.setdefault(room, []).append(doa)

        best_room  = max(votes, key=votes.__getitem__)
        confidence = votes[best_room] / sum(votes.values())

        # Circular mean of DOA angles for the winning room
        doa_rads   = np.deg2rad(doas[best_room])
        mean_doa   = float(np.degrees(np.arctan2(
            np.mean(np.sin(doa_rads)),
            np.mean(np.cos(doa_rads)),
        )))

        return {"room": best_room, "doa_deg": mean_doa, "confidence": confidence}


def main(args=None):
    rclpy.init(args=args)
    node = SpeakerMemory()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
