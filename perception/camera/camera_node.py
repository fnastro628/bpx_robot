#!/usr/bin/env python3
"""
Stereo camera driver — Waveshare IMX219-83 on Jetson Orin Nano Super.

Opens both IMX219 CSI sensors via GStreamer nvarguscamerasrc (hardware ISP,
colour correction, lens shading — much better than raw V4L2).

Published topics:
  /camera/left/image_raw        sensor_msgs/Image
  /camera/left/camera_info      sensor_msgs/CameraInfo
  /camera/right/image_raw       sensor_msgs/Image
  /camera/right/camera_info     sensor_msgs/CameraInfo

Parameters:
  width          (int)   — capture width,  default 1280
  height         (int)   — capture height, default 720
  fps            (int)   — frame rate,     default 30
  left_sensor_id (int)   — nvargus sensor id for left  camera, default 0
  right_sensor_id(int)   — nvargus sensor id for right camera, default 1
  calibration_file (str) — path to stereo calibration YAML (optional)
                           Loaded by camera_info_manager when provided.

GStreamer pipeline used per camera:
  nvarguscamerasrc sensor-id=N
    ! video/x-raw(memory:NVMM),width=W,height=H,framerate=F/1
    ! nvvidconv
    ! video/x-raw,format=BGRx
    ! videoconvert
    ! video/x-raw,format=BGR
    ! appsink drop=1

PASS test (no ROS2 needed):
  python3 camera_node.py --test      # opens both cameras, shows frames in window
  python3 camera_node.py --test --right   # right camera only
"""

from __future__ import annotations

import argparse
import threading
import time

import cv2
import numpy as np

# ROS2 imports are deferred — test mode works without them
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import Header
    from cv_bridge import CvBridge
    _HAS_ROS = True
    _HAS_CV_BRIDGE = True
except ImportError:
    _HAS_ROS = False
    _HAS_CV_BRIDGE = False
    class Node:  # dummy base so class definition succeeds without ROS2
        pass

DEFAULT_WIDTH  = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS    = 30


def _gst_pipeline(sensor_id: int, width: int, height: int, fps: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} "
        f"! video/x-raw(memory:NVMM),width={width},height={height},"
        f"framerate={fps}/1 "
        f"! nvvidconv "
        f"! video/x-raw,format=BGRx "
        f"! videoconvert "
        f"! video/x-raw,format=BGR "
        f"! appsink drop=1"
    )


class StereoCamera:
    """Thin wrapper around two OpenCV VideoCaptures using nvarguscamerasrc."""

    def __init__(
        self,
        left_id:  int = 0,
        right_id: int = 1,
        width:    int = DEFAULT_WIDTH,
        height:   int = DEFAULT_HEIGHT,
        fps:      int = DEFAULT_FPS,
    ):
        self.width  = width
        self.height = height
        self.fps    = fps

        self._left  = cv2.VideoCapture(
            _gst_pipeline(left_id,  width, height, fps), cv2.CAP_GSTREAMER
        )
        self._right = cv2.VideoCapture(
            _gst_pipeline(right_id, width, height, fps), cv2.CAP_GSTREAMER
        )

        if not self._left.isOpened():
            raise RuntimeError(
                f"Left camera (sensor-id={left_id}) failed to open. "
                "Check nvarguscamerasrc is available and sensor is connected."
            )
        if not self._right.isOpened():
            raise RuntimeError(
                f"Right camera (sensor-id={right_id}) failed to open."
            )

    def read(self) -> tuple[bool, np.ndarray, np.ndarray]:
        """Read one frame from each camera. Returns (ok, left_bgr, right_bgr)."""
        ok_l, left  = self._left.read()
        ok_r, right = self._right.read()
        return (ok_l and ok_r), left, right

    def release(self):
        self._left.release()
        self._right.release()


class StereoCameraNode(Node):
    def __init__(self):
        super().__init__("stereo_camera")

        self.declare_parameter("width",           DEFAULT_WIDTH)
        self.declare_parameter("height",          DEFAULT_HEIGHT)
        self.declare_parameter("fps",             DEFAULT_FPS)
        self.declare_parameter("left_sensor_id",  0)
        self.declare_parameter("right_sensor_id", 1)
        self.declare_parameter("calibration_file", "")
        self.declare_parameter("frame_id_left",   "camera_left_optical")
        self.declare_parameter("frame_id_right",  "camera_right_optical")

        w      = self.get_parameter("width").value
        h      = self.get_parameter("height").value
        fps    = self.get_parameter("fps").value
        l_id   = self.get_parameter("left_sensor_id").value
        r_id   = self.get_parameter("right_sensor_id").value
        cal    = self.get_parameter("calibration_file").value
        self._fid_l = self.get_parameter("frame_id_left").value
        self._fid_r = self.get_parameter("frame_id_right").value

        if not _HAS_ROS:
            raise RuntimeError(
                "ROS2 not found — run: source /opt/ros/humble/setup.bash"
            )
        if not _HAS_CV_BRIDGE:
            raise RuntimeError("cv_bridge not found — install ros-humble-cv-bridge")

        self._bridge = CvBridge()

        # ── Publishers ────────────────────────────────────────────────────────
        self._lpub  = self.create_publisher(Image,      "/camera/left/image_raw",    1)
        self._rpub  = self.create_publisher(Image,      "/camera/right/image_raw",   1)
        self._lcpub = self.create_publisher(CameraInfo, "/camera/left/camera_info",  1)
        self._rcpub = self.create_publisher(CameraInfo, "/camera/right/camera_info", 1)

        # ── Camera info (identity until calibration loaded) ───────────────────
        self._linfo = self._default_camera_info(w, h, self._fid_l)
        self._rinfo = self._default_camera_info(w, h, self._fid_r)

        if cal:
            self._load_calibration(cal, w, h)

        # ── Open cameras ──────────────────────────────────────────────────────
        try:
            self._cam = StereoCamera(l_id, r_id, w, h, fps)
            self.get_logger().info(
                f"Stereo camera open — {w}×{h}@{fps}fps  "
                f"left=sensor-{l_id}  right=sensor-{r_id}"
            )
        except RuntimeError as e:
            self.get_logger().error(str(e))
            raise

        # ── Capture loop in background thread ─────────────────────────────────
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    # ── Capture loop ──────────────────────────────────────────────────────────

    def _capture_loop(self):
        while self._running and rclpy.ok():
            ok, left, right = self._cam.read()
            if not ok:
                self.get_logger().warn("Camera read failed — retrying")
                time.sleep(0.05)
                continue

            stamp = self.get_clock().now().to_msg()
            self._publish_frame(left,  self._lpub,  self._lcpub,
                                self._linfo, stamp, self._fid_l)
            self._publish_frame(right, self._rpub,  self._rcpub,
                                self._rinfo, stamp, self._fid_r)

    def _publish_frame(
        self,
        frame:     np.ndarray,
        img_pub,
        info_pub,
        info_msg:  CameraInfo,
        stamp,
        frame_id:  str,
    ):
        img_msg             = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img_msg.header.stamp    = stamp
        img_msg.header.frame_id = frame_id
        img_pub.publish(img_msg)

        info_msg.header.stamp    = stamp
        info_msg.header.frame_id = frame_id
        info_pub.publish(info_msg)

    # ── CameraInfo helpers ────────────────────────────────────────────────────

    @staticmethod
    def _default_camera_info(w: int, h: int, frame_id: str) -> CameraInfo:
        """Identity camera info — no distortion, focal length = width."""
        ci             = CameraInfo()
        ci.width       = w
        ci.height      = h
        ci.header.frame_id = frame_id
        ci.distortion_model = "plumb_bob"
        ci.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        # K: simple pinhole with cx=w/2, cy=h/2, f=w
        f = float(w)
        ci.k = [f, 0.0, w/2.0,
                0.0, f,  h/2.0,
                0.0, 0.0, 1.0]
        ci.r = [1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0]
        ci.p = [f, 0.0, w/2.0, 0.0,
                0.0, f,  h/2.0, 0.0,
                0.0, 0.0, 1.0,  0.0]
        return ci

    def _load_calibration(self, path: str, w: int, h: int):
        """Load calibration from a camera_info_manager-compatible YAML."""
        import yaml, os
        if not os.path.exists(path):
            self.get_logger().warn(f"Calibration file not found: {path}")
            return
        with open(path) as f:
            data = yaml.safe_load(f)

        def _fill(ci, section):
            s = data.get(section, {})
            ci.k = s.get("camera_matrix", {}).get("data", ci.k)
            ci.d = s.get("distortion_coefficients", {}).get("data", ci.d)
            ci.r = s.get("rectification_matrix", {}).get("data", ci.r)
            ci.p = s.get("projection_matrix", {}).get("data", ci.p)
            ci.distortion_model = s.get("distortion_model", "plumb_bob")

        _fill(self._linfo, "left")
        _fill(self._rinfo, "right")
        self.get_logger().info(f"Loaded calibration from {path}")

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self._cam.release()
        super().destroy_node()


# ── Standalone test mode ──────────────────────────────────────────────────────

def _test(right_only: bool = False):
    """Open cameras and show live feed — no ROS2 required."""
    import sys
    cam = StereoCamera()
    print("Press Q to quit.")
    while True:
        ok, left, right = cam.read()
        if not ok:
            print("Read failed"); break
        if not right_only:
            cv2.imshow("Left",  left)
        cv2.imshow("Right", right)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()


# ── Entry points ──────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = StereoCameraNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",       action="store_true",
                        help="Show live camera feed without ROS2")
    parser.add_argument("--right",      action="store_true",
                        help="Show right camera only (use with --test)")
    a = parser.parse_args()
    if a.test:
        _test(right_only=a.right)
    else:
        main()
