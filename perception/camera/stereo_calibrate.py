#!/usr/bin/env python3
"""
Stereo calibration for the Waveshare IMX219-83 camera pair.

Run on the Jetson — no ROS2 needed.

Usage:
  # Step 1 — capture calibration frames (hold checkerboard in ~20 positions)
  python3 stereo_calibrate.py --capture --outdir /tmp/cal_frames

  # Step 2 — solve calibration from saved frames
  python3 stereo_calibrate.py --solve --indir /tmp/cal_frames --out ~/.bpx/stereo_cal.yaml

  # Both steps in one go
  python3 stereo_calibrate.py --capture --solve --outdir /tmp/cal_frames --out ~/.bpx/stereo_cal.yaml

Controls during capture:
  SPACE  — capture current frame pair (only if checkerboard detected in both)
  R      — discard last capture
  Q      — finish capture and proceed

Checkerboard: 9×6 inner corners (10×7 squares), ~30 mm per square.
Print, measure the actual square size with a ruler, and pass --square-mm.

Output YAML is compatible with camera_node.py calibration_file parameter
and with rtabmap stereo calibration format.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from perception.camera.camera_node import StereoCamera

# ── Defaults ─────────────────────────────────────────────────────────────────
BOARD_COLS   = 9       # inner corners wide
BOARD_ROWS   = 6       # inner corners tall
SQUARE_MM    = 30.0    # physical square size in millimetres
MIN_FRAMES   = 20      # minimum captures before solve is allowed
TARGET_FRAMES = 25     # recommended


def _object_points(cols: int, rows: int, sq_mm: float) -> np.ndarray:
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    pts *= sq_mm
    return pts


# ── Capture ───────────────────────────────────────────────────────────────────

def capture(outdir: str, cols: int, rows: int, sq_mm: float):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    cam   = StereoCamera()
    board = (cols, rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    idx   = 0

    print(f"\nCapturing stereo calibration frames.")
    print(f"Board: {cols}×{rows} inner corners, {sq_mm} mm squares.")
    print(f"Target: {TARGET_FRAMES} frames.  Min: {MIN_FRAMES}.")
    print("SPACE=capture  R=undo  Q=quit\n")

    # Try to open display — Jetson may be headless
    has_display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")

    while True:
        ok, left, right = cam.read()
        if not ok:
            print("Camera read error"); continue

        gl = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        ok_l, corners_l = cv2.findChessboardCorners(gl, board, flags)
        ok_r, corners_r = cv2.findChessboardCorners(gr, board, flags)

        detected = ok_l and ok_r

        if has_display:
            vis_l = left.copy();  vis_r = right.copy()
            cv2.drawChessboardCorners(vis_l, board, corners_l, ok_l)
            cv2.drawChessboardCorners(vis_r, board, corners_r, ok_r)
            status = f"{'BOTH' if detected else ('L-only' if ok_l else 'R-only' if ok_r else 'NONE')}  [{idx}/{TARGET_FRAMES}]"
            cv2.putText(vis_l, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0) if detected else (0, 0, 255), 2)
            combined = np.hstack([
                cv2.resize(vis_l, (640, 360)),
                cv2.resize(vis_r, (640, 360)),
            ])
            cv2.imshow("Stereo Calibration — SPACE=capture  R=undo  Q=quit", combined)

        key = cv2.waitKey(1) & 0xFF if has_display else _headless_key(detected, idx)

        if key == ord(" "):
            if not detected:
                print("  Checkerboard not detected in both cameras — skipping")
                continue
            # Sub-pixel refinement
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_l = cv2.cornerSubPix(gl, corners_l, (11,11), (-1,-1), crit)
            corners_r = cv2.cornerSubPix(gr, corners_r, (11,11), (-1,-1), crit)
            cv2.imwrite(f"{outdir}/left_{idx:03d}.jpg",    left)
            cv2.imwrite(f"{outdir}/right_{idx:03d}.jpg",   right)
            np.save(f"{outdir}/corners_l_{idx:03d}.npy", corners_l)
            np.save(f"{outdir}/corners_r_{idx:03d}.npy", corners_r)
            idx += 1
            print(f"  Captured frame {idx}/{TARGET_FRAMES}")
            if idx >= TARGET_FRAMES:
                print("Target reached.")
                break

        elif key == ord("r") and idx > 0:
            idx -= 1
            for f in Path(outdir).glob(f"*_{idx:03d}.*"):
                f.unlink()
            print(f"  Discarded frame {idx+1}")

        elif key == ord("q"):
            if idx < MIN_FRAMES:
                print(f"  Only {idx} frames — need at least {MIN_FRAMES}. Keep going.")
            else:
                print(f"  Done capturing ({idx} frames).")
                break

    cam.release()
    if has_display:
        cv2.destroyAllWindows()
    return idx


def _headless_key(detected: bool, idx: int) -> int:
    """Headless mode: auto-capture with a 2-second interval when detected."""
    if detected:
        time.sleep(2.0)
        print(f"  Auto-capturing (headless) — move board now...")
        return ord(" ")
    time.sleep(0.1)
    return 0xFF


# ── Solve ─────────────────────────────────────────────────────────────────────

def solve(indir: str, outfile: str, cols: int, rows: int, sq_mm: float):
    board    = (cols, rows)
    obj_pts  = _object_points(cols, rows, sq_mm)
    objpoints, imgpoints_l, imgpoints_r = [], [], []

    frame_files = sorted(Path(indir).glob("left_*.jpg"))
    if not frame_files:
        print(f"No frames found in {indir}"); sys.exit(1)

    img_size = None
    for lf in frame_files:
        idx = lf.stem.split("_")[1]
        rf  = Path(indir) / f"right_{idx}.jpg"
        cl  = Path(indir) / f"corners_l_{idx}.npy"
        cr  = Path(indir) / f"corners_r_{idx}.npy"
        if not (rf.exists() and cl.exists() and cr.exists()):
            continue
        left  = cv2.imread(str(lf))
        if img_size is None:
            img_size = (left.shape[1], left.shape[0])
        objpoints.append(obj_pts)
        imgpoints_l.append(np.load(str(cl)))
        imgpoints_r.append(np.load(str(cr)))

    print(f"Solving with {len(objpoints)} frame pairs…")

    cal_flags = cv2.CALIB_RATIONAL_MODEL   # better for wide-angle lenses

    # Individual calibrations first (better initialisation for stereo)
    rms_l, K_l, D_l, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_l, img_size, None, None, flags=cal_flags
    )
    rms_r, K_r, D_r, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_r, img_size, None, None, flags=cal_flags
    )
    print(f"  Left  RMS: {rms_l:.4f} px")
    print(f"  Right RMS: {rms_r:.4f} px")

    # Stereo calibration
    stereo_flags = (
        cv2.CALIB_FIX_INTRINSIC |   # use the individually calibrated intrinsics
        cv2.CALIB_RATIONAL_MODEL
    )
    rms_s, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        K_l, D_l, K_r, D_r,
        img_size,
        flags=stereo_flags,
    )
    print(f"  Stereo RMS: {rms_s:.4f} px")
    print(f"  Baseline (T[0]): {abs(T[0][0]):.2f} mm  (expect ~60 mm)")

    # Stereo rectification
    R_l, R_r, P_l, P_r, Q, roi_l, roi_r = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )

    _save_yaml(outfile, img_size, K_l, D_l, R_l, P_l, K_r, D_r, R_r, P_r,
               R, T, Q, sq_mm, rms_s)
    print(f"\nCalibration saved to {outfile}")
    print(f"Pass to camera_node: --ros-args -p calibration_file:={outfile}")

    return rms_s


def _save_yaml(
    path: str, img_size, K_l, D_l, R_l, P_l,
    K_r, D_r, R_r, P_r, R, T, Q, sq_mm, rms
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _mat(name: str, rows: int, cols: int, data) -> str:
        flat = np.array(data).flatten().tolist()
        data_str = ", ".join(f"{v:.8f}" for v in flat)
        return f"  {name}:\n    rows: {rows}\n    cols: {cols}\n    data: [{data_str}]\n"

    w, h = img_size
    with open(path, "w") as f:
        f.write(f"# Stereo calibration — IMX219-83  RMS={rms:.4f} sq={sq_mm}mm\n")
        f.write(f"image_width: {w}\nimage_height: {h}\n\n")
        f.write("left:\n")
        f.write(f"  distortion_model: rational_poly\n")
        f.write(_mat("camera_matrix",          3, 3, K_l))
        f.write(_mat("distortion_coefficients",1, len(D_l.flatten()), D_l))
        f.write(_mat("rectification_matrix",   3, 3, R_l))
        f.write(_mat("projection_matrix",      3, 4, P_l))
        f.write("\nright:\n")
        f.write(f"  distortion_model: rational_poly\n")
        f.write(_mat("camera_matrix",          3, 3, K_r))
        f.write(_mat("distortion_coefficients",1, len(D_r.flatten()), D_r))
        f.write(_mat("rectification_matrix",   3, 3, R_r))
        f.write(_mat("projection_matrix",      3, 4, P_r))
        f.write("\nstereo:\n")
        f.write(_mat("R", 3, 3, R))
        f.write(_mat("T", 3, 1, T))
        f.write(_mat("Q", 4, 4, Q))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="IMX219-83 stereo calibration")
    p.add_argument("--capture",    action="store_true")
    p.add_argument("--solve",      action="store_true")
    p.add_argument("--outdir",     default="/tmp/cal_frames")
    p.add_argument("--indir",      default="/tmp/cal_frames")
    p.add_argument("--out",        default=os.path.expanduser("~/.bpx/stereo_cal.yaml"))
    p.add_argument("--cols",       type=int,   default=BOARD_COLS)
    p.add_argument("--rows",       type=int,   default=BOARD_ROWS)
    p.add_argument("--square-mm",  type=float, default=SQUARE_MM)
    a = p.parse_args()

    if not a.capture and not a.solve:
        p.print_help(); sys.exit(1)

    n = 0
    if a.capture:
        n = capture(a.outdir, a.cols, a.rows, a.square_mm)

    if a.solve:
        if a.capture and n < MIN_FRAMES:
            print(f"Only {n} frames captured — need {MIN_FRAMES}. Skipping solve.")
        else:
            solve(a.indir, a.out, a.cols, a.rows, a.square_mm)


if __name__ == "__main__":
    main()
