#!/usr/bin/env python3
"""
Command parser node.

Subscribes to /speech/raw (raw Whisper transcript).
Converts text into a structured JSON command and publishes to /commands.

Command JSON schema:
  { "type": "stand" }
  { "type": "sit" }
  { "type": "stop" }
  { "type": "damp" }
  { "type": "velocity", "x": float, "y": float, "yaw": float }
  { "type": "gait",     "gait": int }          # MotionGait enum value
  { "type": "come" }
  { "type": "heel" }
  { "type": "fetch",    "target": str | null }
  { "type": "find",     "target": str }

Speed modifiers ("slow", "fast") scale velocity commands proportionally.
Fuzzy matching handles minor mis-transcriptions.
"""

import difflib
import json
import sys
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Allow importing from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from behaviors.gait_map import COMMAND_ALIASES, GAIT_ALIASES

# Speed words scale the default velocity magnitude
_SPEED_SCALE = {"slowly": 0.5, "slow": 0.5, "fast": 1.6, "quickly": 1.6, "run": 1.4}
_BASE_SPEED   = 0.5   # m/s for "normal" forward motion


class CommandParser(Node):
    def __init__(self):
        super().__init__("command_parser")

        self.sub = self.create_subscription(
            String, "/speech/raw", self._on_speech, 10
        )
        self.pub = self.create_publisher(String, "/commands", 10)

        # Pre-build sorted phrase list for fuzzy matching (longest first)
        self._all_phrases = sorted(
            list(COMMAND_ALIASES.keys()) + list(GAIT_ALIASES.keys()),
            key=len, reverse=True
        )

        self.get_logger().info("Command parser ready.")

    # ── Main callback ─────────────────────────────────────────────────────────

    def _on_speech(self, msg: String):
        text = msg.data.lower().strip()
        cmd  = self._parse(text)
        if cmd is None:
            self.get_logger().info(f"No command matched for: '{text}'")
            return

        out      = String()
        out.data = json.dumps(cmd)
        self.pub.publish(out)
        self.get_logger().info(f"Command: {cmd}")

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse(self, text: str) -> dict | None:
        speed = self._extract_speed(text)

        # 1. Exact substring match (longest phrase first to avoid partial hits)
        for phrase in self._all_phrases:
            if phrase in text:
                return self._resolve(phrase, speed, text)

        # 2. Fuzzy match — try multi-word windows across the transcript
        words = text.split()
        for width in range(min(4, len(words)), 0, -1):
            for start in range(len(words) - width + 1):
                window = " ".join(words[start : start + width])
                hits = difflib.get_close_matches(
                    window, self._all_phrases, n=1, cutoff=0.82
                )
                if hits:
                    return self._resolve(hits[0], speed, text)

        return None

    def _resolve(self, phrase: str, speed: float, full_text: str) -> dict:
        if phrase in COMMAND_ALIASES:
            action = COMMAND_ALIASES[phrase]
            if isinstance(action, tuple):
                cmd_type, params = action
                scaled = {k: round(v * speed, 3) for k, v in params.items()}
                return {"type": cmd_type, **scaled}
            # Fetch: try to extract target object from text
            if action == "fetch":
                target = self._extract_object(full_text)
                return {"type": "fetch", "target": target}
            if action == "find":
                target = self._extract_object(full_text)
                return {"type": "find", "target": target or "unknown"}
            return {"type": action}

        if phrase in GAIT_ALIASES:
            return {"type": "gait", "gait": GAIT_ALIASES[phrase].value}

        return None  # shouldn't reach here

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_speed(self, text: str) -> float:
        for word, scale in _SPEED_SCALE.items():
            if word in text:
                return scale
        return 1.0

    def _extract_object(self, text: str) -> str | None:
        """Very simple object extraction — grab the last noun-like word."""
        # Strip known command words and return whatever remains
        stop_words = {
            "fetch", "get", "retrieve", "bring", "find", "search",
            "for", "the", "a", "an", "me", "it", "that"
        }
        words = [w for w in text.split() if w not in stop_words]
        return words[-1] if words else None


def main(args=None):
    rclpy.init(args=args)
    node = CommandParser()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
