#!/usr/bin/env python3
"""
YOLOv8 object detector node.

Input : /camera/left/image_raw  (sensor_msgs/Image)
Output: /perception/person_detections  (std_msgs/String — JSON)
        /perception/object_detections  (std_msgs/String — JSON)

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
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ObjectDetector(Node):
    def __init__(self):
        super().__init__("object_detector")

        self.declare_parameter("model_path",      "yolov8n.pt")
        self.declare_parameter("conf_threshold",  0.50)
        self.declare_parameter("device",          "0")     # cuda:0

        model_path = self.get_parameter("model_path").value
        conf       = self.get_parameter("conf_threshold").value
        device     = self.get_parameter("device").value

        self.bridge = CvBridge()
        self.conf   = conf

        self.get_logger().info(f"Loading YOLO model: {model_path}")
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.model.to(f"cuda:{device}" if device.isdigit() else device)
        self.get_logger().info("YOLO ready.")

        self._img_sub = self.create_subscription(
            Image, "/camera/left/image_raw", self._on_image, 1
        )
        self._person_pub = self.create_publisher(
            String, "/perception/person_detections", 10
        )
        self._object_pub = self.create_publisher(
            String, "/perception/object_detections", 10
        )

    def _on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w  = frame.shape[:2]

        results = self.model(frame, conf=self.conf, verbose=False)[0]

        best_person   = None
        best_obj      = None

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label  = results.names[cls_id]
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

            cx_norm   = ((x1 + x2) / 2 / w) * 2 - 1   # -1..+1
            size_norm = (y2 - y1) / h                   #  0..1

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

        if best_person:
            m = String(); m.data = json.dumps(best_person)
            self._person_pub.publish(m)

        if best_obj:
            m = String(); m.data = json.dumps(best_obj)
            self._object_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
