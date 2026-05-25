#!/usr/bin/env python3
"""
Face recognition node using face_recognition (dlib HOG + 128-d embeddings).

Replaces InsightFace which crashes on aarch64 (insightface 1.0.1 / stl_vector
bounds assertion in C++ model loader).

Topics:
  IN  /camera/left/image_raw         sensor_msgs/Image
  IN  /perception/enroll_face        std_msgs/String  (person name to enroll)
  OUT /perception/faces              std_msgs/String  (JSON list of detections)

Face detection JSON:
  [{ "name": str,    # person name or "unknown"
     "sim":  float,  # cosine similarity (0–1)
     "bbox": [x1, y1, x2, y2] }, ...]

Enrollment:
  Publish the person's name to /perception/enroll_face.
  Hold the person in front of the camera for ~1 second (ENROLL_FRAMES frames).
  The node averages the embeddings and stores them in the database.
  Publish "" (empty string) to cancel enrollment.

Install:
  pip3 install face_recognition   # compiles dlib from source ~10 min
"""

import json
import sys
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from perception.face_recognition.face_db import FaceDatabase

ENROLL_FRAMES  = 10    # frames to average per enrollment
DETECT_SCALE   = 0.5   # downsample factor for face detection (speed)
DEBUG_EVERY    = 30    # log face count every N frames


class FaceRecognizer(Node):
    def __init__(self):
        super().__init__("face_recognizer")

        # Higher threshold than ArcFace because dlib 128-d embeddings have
        # lower inter-class separation than 512-d ArcFace vectors.
        self.declare_parameter("similarity_threshold", 0.60)
        threshold = self.get_parameter("similarity_threshold").value

        self.db        = FaceDatabase()
        self.bridge    = CvBridge()
        self.threshold = threshold

        self._enrolling_name: str | None   = None
        self._enroll_buf: list[np.ndarray] = []
        self._frame_count = 0

        try:
            import face_recognition as fr
            self._fr = fr
            self.get_logger().info(
                f"face_recognition ready. DB: {self.db.list_people()}"
            )
        except ImportError as e:
            self.get_logger().error(
                f"face_recognition not installed: {e}. "
                "Run: pip3 install face_recognition"
            )
            self._fr = None

        self._img_sub = self.create_subscription(
            Image, "/camera/left/image_raw", self._on_image, 1
        )
        self._enroll_sub = self.create_subscription(
            String, "/perception/enroll_face", self._on_enroll_cmd, 10
        )
        self._face_pub = self.create_publisher(String, "/perception/faces", 10)

    # ── Enrollment trigger ────────────────────────────────────────────────────

    def _on_enroll_cmd(self, msg: String):
        name = msg.data.strip()
        if not name:
            self._enrolling_name = None
            self._enroll_buf.clear()
            self.get_logger().info("Enrollment cancelled.")
            return
        self._enrolling_name = name
        self._enroll_buf.clear()
        self.get_logger().info(f"Enrolling face for: '{name}' — hold still…")

    # ── Image callback ────────────────────────────────────────────────────────

    def _on_image(self, msg: Image):
        if self._fr is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # face_recognition expects RGB; downsample for speed on CPU
        small_rgb = cv2.cvtColor(
            cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE),
            cv2.COLOR_BGR2RGB,
        )

        locations = self._fr.face_locations(small_rgb, model="hog")
        encodings = self._fr.face_encodings(small_rgb, locations)

        self._frame_count += 1
        if self._frame_count % DEBUG_EVERY == 0:
            self.get_logger().info(
                f"frame {self._frame_count}: {len(locations)} face(s) detected "
                f"(image {small_rgb.shape[1]}×{small_rgb.shape[0]}, "
                f"enrolling={self._enrolling_name is not None}, "
                f"buf={len(self._enroll_buf)})"
            )

        scale   = int(1 / DETECT_SCALE)
        results = []

        for enc, loc in zip(encodings, locations):
            top, right, bottom, left = [v * scale for v in loc]
            bbox = [left, top, right, bottom]   # x1,y1,x2,y2
            emb  = np.array(enc, dtype=np.float32)

            if self._enrolling_name:
                self._enroll_buf.append(emb)
                if len(self._enroll_buf) >= ENROLL_FRAMES:
                    avg_emb = np.mean(self._enroll_buf, axis=0)
                    self.db.add_person(self._enrolling_name, avg_emb)
                    self.get_logger().info(
                        f"Enrolled '{self._enrolling_name}' "
                        f"(DB: {self.db.list_people()})"
                    )
                    self._enrolling_name = None
                    self._enroll_buf.clear()
                results.append({"name": "enrolling", "sim": 0.0, "bbox": bbox})
            else:
                name, sim = self.db.identify(emb, threshold=self.threshold)
                results.append({
                    "name": name or "unknown",
                    "sim":  sim,
                    "bbox": bbox,
                })

        if results:
            m      = String()
            m.data = json.dumps(results)
            self._face_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognizer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
