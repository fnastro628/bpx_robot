#!/usr/bin/env python3
"""
Speech-to-Text node using faster-whisper + energy-based VAD.

Listens continuously for speech after detecting audio above the silence
threshold. Strips the wake word then publishes the cleaned transcript to
/speech/raw as a std_msgs/String.

Runs entirely offline on Jetson CUDA — no internet required.
"""

import queue
import threading

import numpy as np
import sounddevice as sd
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

SAMPLE_RATE      = 16000   # Hz — Whisper expects 16 kHz
CHUNK_SECONDS    = 0.5     # audio chunk duration fed to the queue
SILENCE_RMS      = 0.018   # below this RMS = silence
SILENCE_CHUNKS   = 3       # consecutive silent chunks to end a recording
MAX_RECORD_SECS  = 8       # hard cap on a single utterance


class SttNode(Node):
    def __init__(self):
        super().__init__("stt_node")

        self.declare_parameter("model_size", "small")
        self.declare_parameter("device",     "cuda")
        self.declare_parameter("wake_word",  "hey robot")

        model_size = self.get_parameter("model_size").value
        device     = self.get_parameter("device").value
        self.wake_word = self.get_parameter("wake_word").value.lower()

        self.pub = self.create_publisher(String, "/speech/raw", 10)

        self.get_logger().info(f"Loading Whisper '{model_size}' on {device}…")
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device=device, compute_type="float16")
        self.get_logger().info("Whisper ready — listening for speech.")

        self._audio_q: queue.Queue = queue.Queue()
        self._start_audio_thread()

    # ── Audio capture ─────────────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        self._audio_q.put(indata.copy())

    def _start_audio_thread(self):
        chunk_size = int(SAMPLE_RATE * CHUNK_SECONDS)
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._vad_loop, daemon=True)
        self._thread.start()

    # ── VAD + recording loop ──────────────────────────────────────────────────

    def _vad_loop(self):
        buffer = []
        recording = False
        silent_streak = 0
        max_chunks = int(MAX_RECORD_SECS / CHUNK_SECONDS)

        while rclpy.ok():
            try:
                chunk = self._audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if not recording:
                if rms > SILENCE_RMS:
                    recording = True
                    buffer = [chunk]
                    silent_streak = 0
            else:
                buffer.append(chunk)
                if rms < SILENCE_RMS:
                    silent_streak += 1
                else:
                    silent_streak = 0

                if silent_streak >= SILENCE_CHUNKS or len(buffer) >= max_chunks:
                    audio = np.concatenate(buffer).flatten()
                    self._transcribe(audio)
                    buffer = []
                    recording = False
                    silent_streak = 0

    # ── Transcription ─────────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray):
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=3, vad_filter=True
        )
        text = " ".join(s.text for s in segments).strip().lower()

        if not text:
            return

        self.get_logger().info(f"Heard: '{text}'")

        # Require wake word OR allow always-on mode (no wake word set)
        if self.wake_word:
            if self.wake_word not in text:
                return
            text = text.replace(self.wake_word, "").strip()

        if text:
            msg = String()
            msg.data = text
            self.pub.publish(msg)
            self.get_logger().info(f"Published: '{text}'")


def main(args=None):
    rclpy.init(args=args)
    node = SttNode()
    try:
        rclpy.spin(node)
    finally:
        node._stream.stop()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
