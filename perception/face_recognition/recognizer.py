#!/usr/bin/env python3
"""
Face recognition node using InsightFace (ArcFace, buffalo_l model).

Subscribes to the left camera image, detects and embeds all faces,
then identifies each against the FaceDatabase.

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
"""

import json
import sys
import os

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from perception.face_recognition.face_db import FaceDatabase

ENROLL_FRAMES = 10   # number of frames to average per enrollment


class FaceRecognizer(Node):
    def __init__(self):
        super().__init__("face_recognizer")

        self.declare_parameter("similarity_threshold", 0.45)
        threshold = self.get_parameter("similarity_threshold").value

        self.db        = FaceDatabase()
        self.bridge    = CvBridge()
        self.threshold = threshold

        self._enrolling_name: str | None   = None
        self._enroll_buf: list[np.ndarray] = []

        # Load InsightFace
        try:
            from insightface.app import FaceAnalysis
            self._fa = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self._fa.prepare(ctx_id=0, det_size=(640, 640))
            self.get_logger().info(
                f"InsightFace ready. DB: {self.db.list_people()}"
            )
        except Exception as exc:
            self.get_logger().error(f"InsightFace load failed: {exc}")
            self._fa = None

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
        if self._fa is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        faces = self._fa.get(frame)

        results = []

        for face in faces:
            emb = np.array(face.embedding, dtype=np.float32)

            if self._enrolling_name:
                self._enroll_buf.append(emb)
                if len(self._enroll_buf) >= ENROLL_FRAMES:
                    avg_emb = np.mean(self._enroll_buf, axis=0)
                    self.db.add_person(self._enrolling_name, avg_emb)
                    self.get_logger().info(
                        f"Enrolled '{self._enrolling_name}' ✓ "
                        f"(total DB: {self.db.list_people()})"
                    )
                    self._enrolling_name = None
                    self._enroll_buf.clear()
                results.append({
                    "name": "enrolling…",
                    "sim":  0.0,
                    "bbox": [int(v) for v in face.bbox],
                })
            else:
                name, sim = self.db.identify(emb, threshold=self.threshold)
                results.append({
                    "name": name or "unknown",
                    "sim":  sim,
                    "bbox": [int(v) for v in face.bbox],
                })

        if results:
            m = String()
            m.data = json.dumps(results)
            self._face_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognizer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
