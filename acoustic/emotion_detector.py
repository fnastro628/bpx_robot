#!/usr/bin/env python3
"""
CAP 13 — Emotional Tone Detection

Runs a speech emotion model on beamformed XVF3800 audio. Two backend options:

  Backend A (default — lighter, Jetson-friendly):
    SpeechBrain emotion-recognition (wav2vec2-IEMOCAP), exported to ONNX.
    Classes: neutral, happy, angry, sad  (→ mapped to neutral/happy/stressed/distressed)

  Backend B (fallback — uses YAMNet already loaded for event_detector):
    YAMNet AudioSet emotion-adjacent classes: laughter, crying, screaming, shouting.
    Less accurate for tone but requires no extra model.

Subscriptions:
  Reads raw audio directly via sounddevice (same beamformed channel as STT).

Published:
  /acoustic/emotion  std_msgs/String (JSON)
    {"emotion": "stressed", "confidence": 0.82, "raw_class": "angry", "timestamp": 1234.5}

Emotion → robot behaviour adjustments (in behavior_manager.py):
  stressed    → robot moves closer, speaks gently
  distressed  → robot approaches, sends alert
  happy       → more animated behaviors
  neutral     → default

PASS test:
  Speak calmly → "neutral"
  Speak with urgency / raised voice → "stressed" or "upset"
  Shout → "distressed"
  Behavior change observable (robot moves closer when stressed)

Requires:
  pip install speechbrain onnxruntime   (Backend A)
  OR: tflite-runtime + yamnet.tflite    (Backend B, uses existing event_detector model)
"""

import json
import os
import queue
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

SAMPLE_RATE   = 16000
WINDOW_SEC    = 2.0              # 2s analysis window — enough for emotion
CHUNK_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
ONNX_MODEL    = os.path.expanduser("~/.bpx/emotion_model.onnx")
YAMNET_MODEL  = os.path.expanduser("~/.bpx/yamnet.tflite")

# SpeechBrain IEMOCAP → our 4-class mapping
IEMOCAP_MAP = {
    "neu": "neutral",
    "hap": "happy",
    "exc": "happy",
    "ang": "stressed",
    "sad": "distressed",
    "dis": "distressed",
    "fea": "distressed",
    "sur": "neutral",
}

# YAMNet class names → our 4-class mapping (used in Backend B fallback)
YAMNET_EMOTION_MAP = {
    "Laughter":               "happy",
    "Baby laughter":          "happy",
    "Crying, sobbing":        "distressed",
    "Baby cry, infant cry":   "distressed",
    "Screaming":              "distressed",
    "Shouting":               "stressed",
    "Angry":                  "stressed",
}


class EmotionDetector(Node):
    def __init__(self):
        super().__init__("emotion_detector")

        self.declare_parameter("confidence_threshold", 0.60)
        self.declare_parameter("publish_neutral",      False)  # suppress neutral to reduce noise

        self._threshold       = self.get_parameter("confidence_threshold").value
        self._publish_neutral = self.get_parameter("publish_neutral").value

        self._emotion_pub = self.create_publisher(String, "/acoustic/emotion", 10)

        self._vad      = False
        self._audio_q: queue.Queue = queue.Queue(maxsize=2)

        self.create_subscription(Bool, "/acoustic/vad", self._on_vad, 10)

        # Try to load ONNX model (Backend A), fall back to YAMNet (Backend B)
        self._infer = self._load_onnx() or self._load_yamnet()

        if self._infer:
            t = threading.Thread(target=self._inference_loop, daemon=True)
            t.start()
            self._start_audio_capture()
            self.get_logger().info("Emotion detector ready.")
        else:
            self.get_logger().warn(
                "No emotion model found. Place emotion_model.onnx at ~/.bpx/ "
                "or ensure yamnet.tflite is present."
            )

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_onnx(self):
        if not os.path.exists(ONNX_MODEL):
            return None
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(ONNX_MODEL)
            self.get_logger().info("Emotion: using ONNX (SpeechBrain/wav2vec2)")

            def infer_onnx(audio: np.ndarray):
                out = sess.run(None, {"input": audio[np.newaxis, :]})[0][0]
                idx = int(np.argmax(out))
                labels = list(IEMOCAP_MAP.keys())
                raw = labels[idx] if idx < len(labels) else "neu"
                return IEMOCAP_MAP.get(raw, "neutral"), float(out[idx]), raw

            return infer_onnx
        except Exception as e:
            self.get_logger().warn(f"ONNX load failed: {e}")
            return None

    def _load_yamnet(self):
        if not os.path.exists(YAMNET_MODEL):
            return None
        try:
            try:
                import tflite_runtime.interpreter as tflite
                interp = tflite.Interpreter(model_path=YAMNET_MODEL)
            except ImportError:
                import tensorflow as tf
                interp = tf.lite.Interpreter(model_path=YAMNET_MODEL)
            interp.allocate_tensors()

            # Load class names
            csv_path = os.path.expanduser("~/.bpx/yamnet_class_map.csv")
            class_names: list[str] = []
            if os.path.exists(csv_path):
                with open(csv_path) as f:
                    lines = f.readlines()
                class_names = [ln.strip().split(",")[2].strip('"') for ln in lines[1:]]

            inp  = interp.get_input_details()
            outp = interp.get_output_details()

            self.get_logger().info("Emotion: using YAMNet fallback backend")

            def infer_yamnet(audio: np.ndarray):
                interp.set_tensor(inp[0]["index"], audio[np.newaxis, :])
                interp.invoke()
                scores   = interp.get_tensor(outp[0]["index"])[0]
                top_idx  = int(np.argmax(scores))
                top_name = class_names[top_idx] if top_idx < len(class_names) else ""
                emotion  = YAMNET_EMOTION_MAP.get(top_name, "neutral")
                return emotion, float(scores[top_idx]), top_name

            return infer_yamnet
        except Exception as e:
            self.get_logger().warn(f"YAMNet emotion fallback failed: {e}")
            return None

    # ── Audio ─────────────────────────────────────────────────────────────────

    def _on_vad(self, msg: Bool):
        self._vad = msg.data

    def _start_audio_capture(self):
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            if self._vad:
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

    def _inference_loop(self):
        while rclpy.ok():
            try:
                audio = self._audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                emotion, confidence, raw_class = self._infer(audio.astype(np.float32))
            except Exception as e:
                self.get_logger().debug(f"Emotion inference error: {e}")
                continue

            if confidence < self._threshold:
                continue
            if emotion == "neutral" and not self._publish_neutral:
                continue

            self._publish(emotion, confidence, raw_class)

    def _publish(self, emotion: str, confidence: float, raw_class: str):
        msg      = String()
        msg.data = json.dumps({
            "emotion":    emotion,
            "confidence": round(confidence, 3),
            "raw_class":  raw_class,
            "timestamp":  time.time(),
        })
        self._emotion_pub.publish(msg)
        self.get_logger().info(f"Emotion: {emotion} ({confidence:.2f}) [{raw_class}]")


def main(args=None):
    rclpy.init(args=args)
    node = EmotionDetector()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
