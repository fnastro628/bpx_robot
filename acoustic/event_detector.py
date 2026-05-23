#!/usr/bin/env python3
"""
CAP 8 — Acoustic Event Detection

Runs YAMNet (TensorFlow Lite) on raw mic audio to detect home-relevant
acoustic events. Uses all-directional audio (not beamformed) for maximum
sensitivity. Attaches the current DOA angle to every event for spatial context.

Published:
  /acoustic/event  std_msgs/String (JSON)
  {"event": "glass_break", "confidence": 0.94, "doa_deg": 127.0, "timestamp": 1234.5}

Event → behavior_manager command mapping (in behavior_manager.py):
  glass_break  → alert + navigate toward
  alarm        → alert owner
  baby_cry     → alert owner, navigate to source
  fall         → check on person at location
  loud_impact  → alert

PASS test:
  Play glass-break sound from phone → event published within 1 s
  Play smoke alarm sound            → alarm event published
  Normal speech does NOT trigger events
  ros2 topic echo /acoustic/event   shows JSON with class + confidence

Requires:
  pip install tensorflow   (or tflite-runtime on Jetson)
  YAMNet TFLite model: https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1
  Download to ~/.bpx/yamnet.tflite
"""

import json
import os
import queue
import threading
import time
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String

YAMNET_MODEL  = os.path.expanduser("~/.bpx/yamnet.tflite")
SAMPLE_RATE   = 16000
WINDOW_SEC    = 0.5
CHUNK_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)

# YAMNet AudioSet class indices → event key
# Full class list: https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
EVENT_CLASSES = {
    "Glass, shatter":        "glass_break",
    "Smoke detector":        "alarm",
    "Fire alarm":            "alarm",
    "Baby cry, infant cry":  "baby_cry",
    "Thud":                  "fall",
    "Gunshot, gunfire":      "loud_impact",
    "Explosion":             "loud_impact",
    "Dog bark":              "dog_bark",
    "Clapping":              "clap_audio",  # softer than impulse clap
}


class EventDetector(Node):
    def __init__(self):
        super().__init__("event_detector")

        self.declare_parameter("event_confidence_threshold", 0.75)
        self.declare_parameter("model_path", YAMNET_MODEL)

        self._threshold  = self.get_parameter("event_confidence_threshold").value
        model_path       = self.get_parameter("model_path").value

        self._event_pub = self.create_publisher(String, "/acoustic/event", 10)
        self._doa       = 0.0

        self.create_subscription(Float32, "/acoustic/doa", self._on_doa, 10)

        self._audio_q: queue.Queue = queue.Queue(maxsize=4)
        self._interpreter = self._load_model(model_path)
        self._class_names = self._load_class_names()

        if self._interpreter:
            t = threading.Thread(target=self._inference_loop, daemon=True)
            t.start()
            self._start_audio_capture()
            self.get_logger().info(
                f"Event detector ready — threshold={self._threshold}"
            )
        else:
            self.get_logger().warn("YAMNet model not loaded. Event detection disabled.")

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self, path: str):
        if not os.path.exists(path):
            self.get_logger().warn(
                f"YAMNet model not found at {path}. "
                "Download from: https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1"
            )
            return None
        try:
            import tflite_runtime.interpreter as tflite
            interp = tflite.Interpreter(model_path=path)
            interp.allocate_tensors()
            return interp
        except ImportError:
            try:
                import tensorflow as tf
                interp = tf.lite.Interpreter(model_path=path)
                interp.allocate_tensors()
                return interp
            except Exception as e:
                self.get_logger().error(f"TFLite load failed: {e}")
                return None

    def _load_class_names(self) -> list[str]:
        csv_path = os.path.expanduser("~/.bpx/yamnet_class_map.csv")
        if not os.path.exists(csv_path):
            return []
        with open(csv_path) as f:
            lines = f.readlines()
        return [line.strip().split(",")[2].strip('"') for line in lines[1:]]

    # ── Audio capture ─────────────────────────────────────────────────────────

    def _start_audio_capture(self):
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            try:
                self._audio_q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=callback,
        )
        self._stream.start()

    # ── Inference loop ────────────────────────────────────────────────────────

    def _on_doa(self, msg: Float32):
        self._doa = msg.data

    def _inference_loop(self):
        inp  = self._interpreter.get_input_details()
        outp = self._interpreter.get_output_details()

        while rclpy.ok():
            try:
                chunk = self._audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

            # YAMNet expects float32 waveform at 16 kHz
            waveform = chunk.astype(np.float32)
            self._interpreter.set_tensor(inp[0]["index"], waveform[np.newaxis, :])
            self._interpreter.invoke()
            scores = self._interpreter.get_tensor(outp[0]["index"])[0]

            top_idx   = int(np.argmax(scores))
            top_score = float(scores[top_idx])
            top_name  = (
                self._class_names[top_idx]
                if top_idx < len(self._class_names)
                else str(top_idx)
            )

            if top_name in EVENT_CLASSES and top_score >= self._threshold:
                event_key = EVENT_CLASSES[top_name]
                self._publish_event(event_key, top_score)

    def _publish_event(self, event: str, confidence: float):
        msg = String()
        msg.data = json.dumps({
            "event":      event,
            "confidence": round(confidence, 3),
            "doa_deg":    round(self._doa, 1),
            "timestamp":  time.time(),
        })
        self._event_pub.publish(msg)
        self.get_logger().info(f"EVENT: {event} ({confidence:.2f}) from {self._doa:.0f}°")


def main(args=None):
    rclpy.init(args=args)
    node = EventDetector()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
