#!/usr/bin/env python3
"""
YOLOv8 object detector node.

Input : /camera/left/image_raw        sensor_msgs/Image
Output: /perception/person_detections  std_msgs/String (JSON)
        /perception/object_detections  std_msgs/String (JSON)
        /perception/detection_viz      sensor_msgs/Image  (annotated, 30 Hz)

Detection JSON schema:
  { "class": str, "conf": float, "cx": float, "size": float,
    "bbox": [x1, y1, x2, y2] }

  cx   : normalised horizontal offset from frame centre [-1, +1]
  size : bbox height as fraction of frame height [0, 1]

For Jetson: export the model once with:
  yolo export model=yolov8n.pt format=engine device=0
Then set model_path parameter to the .engine file for full TensorRT speed.
"""

import json
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Colour palette per class index (BGR)
_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72),  (23, 204, 146), (134, 219, 61),
    (52, 147, 26),  (187, 212, 0),  (168, 153, 44), (255, 194, 0),
    (255, 152, 0),  (255, 87, 34),  (255, 55, 100), (221, 34, 118),
]


class ObjectDetector(Node):
    def __init__(self):
        super().__init__("object_detector")

        self.declare_parameter("model_path",     "yolov8n.pt")
        self.declare_parameter("conf_threshold", 0.50)
        self.declare_parameter("device",         "0")      # cuda:0
        self.declare_parameter("publish_viz",    True)

        model_path      = self.get_parameter("model_path").value
        conf            = self.get_parameter("conf_threshold").value
        device          = self.get_parameter("device").value
        self._pub_viz   = self.get_parameter("publish_viz").value

        self.bridge = CvBridge()
        self.conf   = conf

        self.get_logger().info(f"Loading YOLO model: {model_path}")
        import torch
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        if device.isdigit():
            if torch.cuda.is_available():
                target = f"cuda:{device}"
            else:
                self.get_logger().warn(
                    "CUDA not available — running on CPU. "
                    "Install Jetson PyTorch wheel for GPU speed."
                )
                target = "cpu"
        else:
            target = device
        self.model.to(target)
        self.get_logger().info(f"YOLO ready on {target}.")

        self._img_sub = self.create_subscription(
            Image, "/camera/left/image_raw", self._on_image, 1
        )
        self._person_pub = self.create_publisher(
            String, "/perception/person_detections", 10
        )
        self._object_pub = self.create_publisher(
            String, "/perception/object_detections", 10
        )
        self._viz_pub = self.create_publisher(
            Image, "/perception/detection_viz", 1
        )

    def _on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w  = frame.shape[:2]

        results = self.model(frame, conf=self.conf, verbose=False)[0]

        best_person = None
        best_obj    = None
        viz         = frame.copy() if self._pub_viz else None

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label  = results.names[cls_id]
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

            cx_norm   = ((x1 + x2) / 2 / w) * 2 - 1
            size_norm = (y2 - y1) / h

            det = {
                "class": label,
                "conf":  round(conf, 2),
                "cx":    round(cx_norm, 3),
                "size":  round(size_norm, 3),
                "bbox":  [int(x1), int(y1), int(x2), int(y2)],
            }

            if cls_id == 0:   # COCO class 0 = person
                if best_person is None or size_norm > best_person["size"]:
                    best_person = det
            else:
                if best_obj is None or size_norm > best_obj["size"]:
                    best_obj = det

            if viz is not None:
                color = _PALETTE[cls_id % len(_PALETTE)]
                cv2.rectangle(viz, (int(x1), int(y1)), (int(x2), int(y2)),
                              color, 2)
                text  = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(viz,
                              (int(x1), int(y1) - th - 6),
                              (int(x1) + tw, int(y1)), color, -1)
                cv2.putText(viz, text, (int(x1), int(y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        if best_person:
            m = String(); m.data = json.dumps(best_person)
            self._person_pub.publish(m)

        if best_obj:
            m = String(); m.data = json.dumps(best_obj)
            self._object_pub.publish(m)

        if viz is not None:
            viz_msg = self.bridge.cv2_to_imgmsg(viz, encoding="bgr8")
            viz_msg.header = msg.header
            self._viz_pub.publish(viz_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
