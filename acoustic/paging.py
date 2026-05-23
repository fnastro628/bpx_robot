#!/usr/bin/env python3
"""
CAP 7 — Paging ("Find Me From Another Room")

Runs as a background listener when the robot is IDLE.
Uses openwakeword (lightweight, offline) to detect the robot's name from any
direction, then publishes a "come" command targeted at the DOA of the caller.

Pipeline:
  1. openwakeword detects wake word from beamformed audio
  2. Head turns toward DOA of the detection
  3. Whisper confirms (optional — avoids false positives)
  4. Publishes {"type": "page_approach", "doa_deg": X} to /commands
  5. behavior_manager navigates robot toward DOA via VSLAM

PASS test:
  Robot idle in living room.
  Call robot name from kitchen (no line of sight).
  Robot turns head toward kitchen, then navigates there.

Requires: pip install openwakeword
  Model: download "hey_jarvis" or train custom wake word for robot's name.
  See: https://github.com/dscripka/openWakeWord
"""

import json
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool, String, UInt8

from behaviors.gait_map import MotionState

WAKE_MODEL   = "hey_jarvis"     # replace with custom model for robot's name
SAMPLE_RATE  = 16000
CHUNK_SIZE   = 1280             # 80 ms chunks at 16 kHz (openwakeword requirement)
THRESHOLD    = 0.5              # wake word confidence threshold


class PagingNode(Node):
    def __init__(self):
        super().__init__("paging_node")

        self.declare_parameter("wake_model",  WAKE_MODEL)
        self.declare_parameter("threshold",   THRESHOLD)

        model_name  = self.get_parameter("wake_model").value
        self._thr   = self.get_parameter("threshold").value

        self._cmd_pub = self.create_publisher(String, "/commands",    10)
        self._pan_pub = self.create_publisher(Float32, "/head/pan_deg", 10)

        self._doa          = 0.0
        self._motion_state = MotionState.LYING_DOWN
        self._idle         = True

        self.create_subscription(Float32, "/acoustic/doa",    self._on_doa,   10)
        self.create_subscription(UInt8,   "/bpx/motion_state", self._on_state, 10)

        # Load openwakeword model
        try:
            from openwakeword.model import Model
            self._oww = Model(wakeword_models=[model_name], inference_framework="onnx")
            self.get_logger().info(f"openwakeword ready — model: {model_name}")
        except Exception as e:
            self.get_logger().warn(f"openwakeword not available: {e}. Paging disabled.")
            self._oww = None
            return

        # Audio capture thread
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    # ── State tracking ────────────────────────────────────────────────────────

    def _on_doa(self, msg: Float32):
        raw = msg.data
        self._doa = raw - 360.0 if raw > 180.0 else raw

    def _on_state(self, msg: UInt8):
        self._motion_state = msg.data
        self._idle = (self._motion_state in (
            MotionState.LYING_DOWN, MotionState.SIT_DOWN, MotionState.PASSIVE
        ))

    # ── Wake word listen loop ─────────────────────────────────────────────────

    def _listen_loop(self):
        import sounddevice as sd

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SIZE,
        ) as stream:
            self.get_logger().info("Paging: listening for wake word...")
            while rclpy.ok():
                chunk, _ = stream.read(CHUNK_SIZE)
                if not self._idle:
                    continue   # only page when robot is not busy

                audio = chunk[:, 0].astype(np.int16)
                prediction = self._oww.predict(audio)

                for name, score in prediction.items():
                    if score >= self._thr:
                        self.get_logger().info(
                            f"Wake word '{name}' detected "
                            f"(score={score:.2f}) from DOA={self._doa:.1f}°"
                        )
                        self._on_wake(self._doa)
                        self._oww.reset()
                        break

    def _on_wake(self, doa_deg: float):
        # Turn head toward caller
        pan = Float32(); pan.data = doa_deg
        self._pan_pub.publish(pan)

        # Publish page_approach command (behavior_manager handles navigation)
        cmd = String()
        cmd.data = json.dumps({"type": "page_approach", "doa_deg": doa_deg})
        self._cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PagingNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
