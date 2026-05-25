#!/usr/bin/env python3
"""
Live debug viewer — run on the Jetson with a monitor connected.

Shows camera feed with all active perception overlays in a single window:
  - YOLO object/person bounding boxes
  - Face recognition boxes + names
  - Enrollment progress bar
  - System status panel (FPS, detection counts, topic health)

Usage:
  DISPLAY=:0 python3 debug_viewer.py

Press Q in the window to quit.
"""

import json
import time
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ── Colours (BGR) ────────────────────────────────────────────────────────────
C_PERSON   = (50,  205,  50)   # green
C_OBJECT   = (255, 165,   0)   # orange
C_FACE_KNW = ( 30, 144, 255)   # blue  — known face
C_FACE_UNK = ( 80,  80, 220)   # purple — unknown
C_ENROLL   = (  0, 215, 255)   # gold  — enrolling
C_PANEL    = ( 30,  30,  30)   # dark grey panel bg
C_TEXT     = (220, 220, 220)   # light text
C_OK       = ( 50, 205,  50)
C_WARN     = (  0, 165, 255)
C_ERR      = ( 50,  50, 220)

PANEL_W    = 300   # right-hand status panel width
FONT       = cv2.FONT_HERSHEY_SIMPLEX
ENROLL_FRAMES = 10  # must match recognizer.py


class DebugViewer(Node):
    def __init__(self):
        super().__init__("debug_viewer")
        self.bridge = CvBridge()
        self._lock  = threading.Lock()

        # Latest data from each topic
        self._frame        = None
        self._person_det   = None   # dict or None
        self._object_det   = None   # dict or None
        self._faces        = []     # list of dicts
        self._enroll_name  = None   # currently enrolling name
        self._enroll_count = 0      # frames collected so far

        # FPS tracking per topic
        self._fps_cam    = _FPSTracker()
        self._fps_person = _FPSTracker()
        self._fps_face   = _FPSTracker()

        # Subscriptions
        self.create_subscription(
            Image,  "/camera/left/image_raw",        self._on_image,  1)
        self.create_subscription(
            String, "/perception/person_detections",  self._on_person, 10)
        self.create_subscription(
            String, "/perception/object_detections",  self._on_object, 10)
        self.create_subscription(
            String, "/perception/faces",              self._on_faces,  10)
        self.create_subscription(
            String, "/perception/enroll_face",        self._on_enroll, 10)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self._fps_cam.tick()
        with self._lock:
            self._frame = frame

    def _on_person(self, msg: String):
        self._fps_person.tick()
        with self._lock:
            self._person_det = json.loads(msg.data)

    def _on_object(self, msg: String):
        with self._lock:
            self._object_det = json.loads(msg.data)

    def _on_faces(self, msg: String):
        self._fps_face.tick()
        faces = json.loads(msg.data)
        with self._lock:
            self._faces = faces
            # Track enrollment progress
            enrolling = [f for f in faces if f["name"] == "enrolling"]
            if enrolling:
                self._enroll_count += 1
            else:
                if self._enroll_count > 0:
                    self._enroll_count  = 0
                    self._enroll_name   = None

    def _on_enroll(self, msg: String):
        name = msg.data.strip()
        with self._lock:
            self._enroll_name  = name or None
            self._enroll_count = 0

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self) -> np.ndarray | None:
        with self._lock:
            if self._frame is None:
                return None
            frame        = self._frame.copy()
            person_det   = self._person_det
            object_det   = self._object_det
            faces        = list(self._faces)
            enroll_name  = self._enroll_name
            enroll_count = self._enroll_count

        h, w = frame.shape[:2]

        # ── YOLO person box ───────────────────────────────────────────────────
        if person_det:
            x1, y1, x2, y2 = person_det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_PERSON, 2)
            _label(frame, f"person {person_det['conf']:.0%}", x1, y1, C_PERSON)

        # ── YOLO object box ───────────────────────────────────────────────────
        if object_det:
            x1, y1, x2, y2 = object_det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_OBJECT, 2)
            _label(frame, f"{object_det['class']} {object_det['conf']:.0%}",
                   x1, y1, C_OBJECT)

        # ── Face boxes ────────────────────────────────────────────────────────
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            name = face["name"]
            if name == "enrolling":
                color = C_ENROLL
                lbl   = f"enrolling ({enroll_count}/{ENROLL_FRAMES})"
            elif name == "unknown":
                color = C_FACE_UNK
                lbl   = f"unknown ({face['sim']:.2f})"
            else:
                color = C_FACE_KNW
                lbl   = f"{name} ({face['sim']:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            _label(frame, lbl, x1, y1, color)

        # ── Enrollment progress bar ───────────────────────────────────────────
        if enroll_name and enroll_count > 0:
            bar_w = int(w * enroll_count / ENROLL_FRAMES)
            cv2.rectangle(frame, (0, h - 20), (w, h), (40, 40, 40), -1)
            cv2.rectangle(frame, (0, h - 20), (bar_w, h), C_ENROLL, -1)
            cv2.putText(frame, f"Enrolling '{enroll_name}'…",
                        (10, h - 5), FONT, 0.55, (0, 0, 0), 2)

        # ── Status panel ─────────────────────────────────────────────────────
        panel = self._make_panel(h, person_det, object_det, faces, enroll_name)

        return np.hstack([frame, panel])

    def _make_panel(self, h, person_det, object_det, faces, enroll_name) -> np.ndarray:
        panel = np.full((h, PANEL_W, 3), C_PANEL, dtype=np.uint8)
        y = 30

        def row(text, color=C_TEXT, scale=0.5, bold=False):
            nonlocal y
            cv2.putText(panel, text, (12, y), FONT, scale, color,
                        2 if bold else 1, cv2.LINE_AA)
            y += int(scale * 40 + 4)

        row("BPX DEBUG VIEWER", C_OK, 0.6, bold=True)
        y += 6
        cv2.line(panel, (8, y), (PANEL_W - 8, y), (80, 80, 80), 1); y += 12

        row("CAMERA", C_TEXT, 0.45, bold=True)
        row(f"  FPS:    {self._fps_cam.fps:.1f}", _fps_color(self._fps_cam.fps, 25))
        y += 6

        row("YOLO DETECTOR", C_TEXT, 0.45, bold=True)
        row(f"  topic:  {_hz_str(self._fps_person.fps)}",
            _fps_color(self._fps_person.fps, 5))
        if person_det:
            row(f"  person: cx={person_det['cx']:+.2f}  sz={person_det['size']:.2f}", C_OK)
        else:
            row("  person: none", (120, 120, 120))
        if object_det:
            row(f"  object: {object_det['class']} {object_det['conf']:.0%}", C_WARN)
        else:
            row("  object: none", (120, 120, 120))
        y += 6

        row("FACE RECOGNIZER", C_TEXT, 0.45, bold=True)
        row(f"  topic:  {_hz_str(self._fps_face.fps)}",
            _fps_color(self._fps_face.fps, 1))
        if faces:
            for face in faces[:4]:   # show up to 4
                n = face["name"]
                s = face["sim"]
                c = C_ENROLL if n == "enrolling" else \
                    C_FACE_KNW if n not in ("unknown",) else C_FACE_UNK
                row(f"  {n:<14} {s:.2f}", c)
        else:
            row("  no faces detected", (120, 120, 120))

        if enroll_name:
            y += 4
            row(f"ENROLLING: {enroll_name}", C_ENROLL, 0.5, bold=True)

        # Tips at bottom
        y = h - 50
        cv2.line(panel, (8, y), (PANEL_W - 8, y), (80, 80, 80), 1); y += 14
        row("Q — quit", (100, 100, 100), 0.4)
        row("Enroll: ros2 topic pub --once", (100, 100, 100), 0.4)

        return panel


# ── Helpers ───────────────────────────────────────────────────────────────────

def _label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    by = max(y - 4, th + 4)
    cv2.rectangle(img, (x, by - th - 4), (x + tw + 4, by + 2), color, -1)
    cv2.putText(img, text, (x + 2, by - 2), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def _fps_color(fps, ok_thresh):
    if fps >= ok_thresh:       return C_OK
    if fps >= ok_thresh * 0.3: return C_WARN
    return C_ERR


def _hz_str(fps):
    return f"{fps:.1f} Hz" if fps > 0.1 else "waiting…"


class _FPSTracker:
    def __init__(self, window=2.0):
        self._window = window
        self._times  = []
        self.fps     = 0.0

    def tick(self):
        now = time.monotonic()
        self._times.append(now)
        cutoff = now - self._window
        self._times = [t for t in self._times if t > cutoff]
        self.fps = len(self._times) / self._window


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = DebugViewer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    cv2.namedWindow("BPX Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BPX Debug", 1580, 720)

    print("Debug viewer running. Press Q in the window to quit.")

    while rclpy.ok():
        frame = node.render()
        if frame is not None:
            cv2.imshow("BPX Debug", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
