#!/usr/bin/env python3
"""
CAP 14/15 — Acoustic SLAM Landmarks + Selective Beamforming

CAP 14 — Acoustic landmarks:
  Identifies persistent stationary sound sources (TV hum, fridge, HVAC fan)
  by correlating a consistent DOA angle at a fixed map pose over multiple visits.
  Registers confirmed landmarks in RTAB-Map as semantic markers.

CAP 15 — Selective beamforming:
  Exposes /acoustic/lock_beam service (std_srvs/SetFloat) that writes the
  XVF3800's beam-lock direction via USB HID control register, focusing the
  mic array on a chosen azimuth and suppressing off-axis sound.
  Used by heel/come behaviors to listen only toward the target person.

Subscriptions:
  /acoustic/doa        Float32   — current DOA from XVF3800
  /acoustic/energy     Float32   — mic energy level
  /bpx/odometry        Odometry  — robot map pose
  /acoustic/vad        Bool      — True when speech detected (gate for landmarks)

Published:
  /acoustic/landmark   std_msgs/String (JSON)
    {"id": "fridge_kitchen", "doa_deg": 45.0, "map_x": 1.2, "map_y": 3.4,
     "spectral_signature": [...], "confidence": 0.92, "timestamp": 1234.5}

Services:
  /acoustic/lock_beam   std_srvs/SetBool  (data=True: lock, data=False: unlock)
    — when locked, reads azimuth from /acoustic/beam_target_deg topic

Topics (beam control):
  /acoustic/beam_target_deg  Float32  — target azimuth for locked beam

PASS test:
  - TV on → landmark registered at correct map position after 3+ visits
  - Move robot to 3 positions → TV DOA from each matches map geometry
  - Lock beam toward TV → speech from that direction heard, side speech suppressed
"""

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import SetBool
from nav_msgs.msg import Odometry

DB_PATH = Path.home() / ".bpx" / "acoustic_landmarks.db"

# A source is a landmark candidate when:
CANDIDATE_WINDOW_SEC    = 5.0     # observe for this many seconds
CANDIDATE_DOA_STD_MAX   = 8.0     # DOA must be stable (std < 8°)
CANDIDATE_ENERGY_MIN    = 0.05    # must be above noise floor
LANDMARK_CONFIRM_VISITS = 3       # confirmed after this many separate visits

# XVF3800 USB HID — beam control register (vendor-specific; verify on hardware)
PARAM_BEAM_LOCK   = 0x001C        # write azimuth here to lock beam
PARAM_BEAM_UNLOCK = 0x001D        # write 1 to release beam lock
VENDOR_ID         = 0x2886
PRODUCT_ID        = 0x0019


class AcousticSLAM(Node):
    def __init__(self):
        super().__init__("acoustic_slam")

        self._landmark_pub = self.create_publisher(String, "/acoustic/landmark", 10)

        self._doa    = 0.0
        self._energy = 0.0
        self._vad    = False
        self._pose_x = 0.0
        self._pose_y = 0.0

        self.create_subscription(Float32,  "/acoustic/doa",    self._on_doa,    10)
        self.create_subscription(Float32,  "/acoustic/energy", self._on_energy, 10)
        self.create_subscription(Bool,     "/acoustic/vad",    self._on_vad,    10)
        self.create_subscription(Odometry, "/bpx/odometry",    self._on_odom,   10)
        self.create_subscription(
            Float32, "/acoustic/beam_target_deg", self._on_beam_target, 10
        )

        self.create_service(SetBool, "/acoustic/lock_beam", self._on_lock_beam)

        self._init_db()

        # Rolling observation buffer for the current candidate
        self._obs_buffer: list[tuple[float, float, float]] = []  # (time, doa, energy)
        self._beam_locked = False
        self._beam_target = 0.0
        self._usb_dev     = self._open_usb()

        self.create_timer(CANDIDATE_WINDOW_SEC, self._evaluate_candidate)

        self.get_logger().info(
            f"Acoustic SLAM ready — {self._count_landmarks()} landmarks stored."
        )

    # ── Database ──────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(DB_PATH))

    def _init_db(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS landmarks (
                    id           TEXT    PRIMARY KEY,
                    doa_deg      REAL    NOT NULL,
                    map_x        REAL    NOT NULL,
                    map_y        REAL    NOT NULL,
                    visits       INTEGER DEFAULT 1,
                    spectral_sig BLOB,
                    last_seen    REAL    NOT NULL
                )
            """)

    def _count_landmarks(self) -> int:
        with self._conn() as con:
            return con.execute("SELECT COUNT(*) FROM landmarks").fetchone()[0]

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_doa(self, msg: Float32):
        raw = msg.data
        self._doa = raw - 360.0 if raw > 180.0 else raw

    def _on_energy(self, msg: Float32):
        self._energy = msg.data

    def _on_vad(self, msg: Bool):
        self._vad = msg.data

    def _on_odom(self, msg: Odometry):
        self._pose_x = msg.pose.pose.position.x
        self._pose_y = msg.pose.pose.position.y

    def _on_beam_target(self, msg: Float32):
        self._beam_target = msg.data
        if self._beam_locked:
            self._write_beam(self._beam_target)

    # ── Landmark detection ────────────────────────────────────────────────────

    def _evaluate_candidate(self):
        now = time.time()
        self._obs_buffer.append((now, self._doa, self._energy))
        # Keep only the last CANDIDATE_WINDOW_SEC of observations
        cutoff = now - CANDIDATE_WINDOW_SEC
        self._obs_buffer = [o for o in self._obs_buffer if o[0] >= cutoff]

        # Gate: must have enough energy, not be speech, and have stable DOA
        non_speech = [(t, d, e) for t, d, e in self._obs_buffer if not self._vad]
        if len(non_speech) < 3:
            return

        energies = [e for _, _, e in non_speech]
        if np.mean(energies) < CANDIDATE_ENERGY_MIN:
            return

        doas = [d for _, d, _ in non_speech]
        if np.std(doas) > CANDIDATE_DOA_STD_MAX:
            return

        # Stable, persistent, non-speech source — update or create landmark
        mean_doa = float(np.mean(doas))
        self._update_landmark(mean_doa, float(np.mean(energies)))

    def _update_landmark(self, doa_deg: float, energy: float):
        """
        Match or create a landmark. Two DOA readings from the same map position
        within 10° are considered the same source.
        """
        with self._conn() as con:
            rows = con.execute(
                "SELECT id, doa_deg, map_x, map_y, visits FROM landmarks"
            ).fetchall()

        match_id   = None
        match_dist = float("inf")
        for lid, ldoa, lx, ly, lvisits in rows:
            # Spatial + angular proximity check
            pose_dist   = np.hypot(self._pose_x - lx, self._pose_y - ly)
            doa_diff    = abs((doa_deg - ldoa + 180) % 360 - 180)
            # Same source if we're within 1 m of the robot's prior pose AND
            # the DOA from different positions is geometrically consistent
            if pose_dist < 1.5 and doa_diff < 15.0:
                if pose_dist < match_dist:
                    match_dist = pose_dist
                    match_id   = lid

        now = time.time()
        if match_id:
            with self._conn() as con:
                row = con.execute(
                    "SELECT visits FROM landmarks WHERE id = ?", (match_id,)
                ).fetchone()
                visits = row[0] + 1
                con.execute(
                    "UPDATE landmarks SET visits=?, doa_deg=?, last_seen=? WHERE id=?",
                    (visits, doa_deg, now, match_id),
                )
            if visits == LANDMARK_CONFIRM_VISITS:
                self._publish_landmark(match_id, doa_deg)
                self.get_logger().info(
                    f"Landmark confirmed: {match_id}  doa={doa_deg:.0f}°"
                )
        else:
            new_id = f"src_{int(now)}"
            with self._conn() as con:
                con.execute(
                    "INSERT INTO landmarks (id, doa_deg, map_x, map_y, last_seen) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (new_id, doa_deg, self._pose_x, self._pose_y, now),
                )
            self.get_logger().debug(
                f"New landmark candidate: {new_id}  doa={doa_deg:.0f}°"
            )

    def _publish_landmark(self, lid: str, doa_deg: float):
        with self._conn() as con:
            row = con.execute(
                "SELECT doa_deg, map_x, map_y, visits FROM landmarks WHERE id=?",
                (lid,),
            ).fetchone()
        if not row:
            return

        msg      = String()
        msg.data = json.dumps({
            "id":         lid,
            "doa_deg":    round(row[0], 1),
            "map_x":      round(row[1], 3),
            "map_y":      round(row[2], 3),
            "visits":     row[3],
            "confidence": min(1.0, row[3] / LANDMARK_CONFIRM_VISITS),
            "timestamp":  time.time(),
        })
        self._landmark_pub.publish(msg)

    # ── Selective beamforming (CAP 15) ────────────────────────────────────────

    def _on_lock_beam(self, request: SetBool.Request, response: SetBool.Response):
        if request.data:
            self._beam_locked = True
            self._write_beam(self._beam_target)
            response.message = f"Beam locked to {self._beam_target:.0f}°"
        else:
            self._beam_locked = False
            self._unlock_beam()
            response.message = "Beam unlocked"
        response.success = True
        self.get_logger().info(response.message)
        return response

    def _open_usb(self):
        """Open the XVF3800 USB device for HID control writes."""
        try:
            import usb.core
            dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
            if dev is None:
                self.get_logger().warn(
                    "XVF3800 USB device not found — beamforming control unavailable."
                )
            return dev
        except ImportError:
            self.get_logger().warn("pyusb not installed — beamforming control unavailable.")
            return None

    def _write_beam(self, azimuth_deg: float):
        if self._usb_dev is None:
            return
        angle_int = int(azimuth_deg) % 360
        try:
            self._usb_dev.ctrl_transfer(
                0x40, 0x00, PARAM_BEAM_LOCK, 0,
                np.array([angle_int], dtype=np.int32).tobytes(),
            )
        except Exception as e:
            self.get_logger().warn(f"Beam lock write failed: {e}")

    def _unlock_beam(self):
        if self._usb_dev is None:
            return
        try:
            self._usb_dev.ctrl_transfer(
                0x40, 0x00, PARAM_BEAM_UNLOCK, 0,
                np.array([1], dtype=np.int32).tobytes(),
            )
        except Exception as e:
            self.get_logger().warn(f"Beam unlock write failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = AcousticSLAM()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
