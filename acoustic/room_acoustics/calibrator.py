#!/usr/bin/env python3
"""
CAP 9 — Guided Acoustic Calibration

Interactive workflow that walks the user through recording room acoustic
fingerprints.  Run standalone (no robot needed) or via a ROS2 service.

Standalone usage:
  python calibrator.py --room living_room --barks 7
  python calibrator.py --room bathroom    --barks 5
  python calibrator.py --list            # show all stored rooms + vector counts
  python calibrator.py --delete bedroom  # remove a room from the DB

  # CAP 11 — within-room grid calibration
  python calibrator.py --room living_room --grid 3x3
    → prompts for 9 positions (A1 … C3), 5 barks per position

ROS2 service usage:
  ros2 service call /acoustic/calibrate_room bpx_interfaces/srv/CalibrateRoom \
      "{room_name: 'bedroom', n_barks: 5}"

Each calibration session:
  1. Prompts operator to stand robot in position.
  2. Sounds a ready beep (optional).
  3. Fires N barks (default 5), records RIR features for each.
  4. Stores all vectors in RoomDatabase under the given room name.
  5. Reports feature statistics and flags if the new vectors are consistent
     with previously stored ones (cosine distance < 0.15).
"""

import argparse
import sys
import time

import numpy as np

from acoustic.room_acoustics.bark_signal import BarkSignal
from acoustic.room_acoustics.rir_extractor import RIRExtractor
from acoustic.room_acoustics.room_db import RoomDatabase


BARKS_PER_POSITION = 5
INTER_BARK_SEC     = 2.0    # pause between consecutive barks


# ── Core calibration ──────────────────────────────────────────────────────────

def calibrate_room(
    room_name: str,
    n_barks: int = BARKS_PER_POSITION,
    db: RoomDatabase | None = None,
    vslam_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    verbose: bool = True,
) -> list[np.ndarray]:
    """
    Record n_barks RIR feature vectors for room_name and store them in the DB.
    Returns the list of feature vectors measured.
    """
    if db is None:
        db = RoomDatabase()

    bark = BarkSignal()
    rx   = RIRExtractor(bark)

    features_collected: list[np.ndarray] = []

    for i in range(n_barks):
        if verbose:
            print(f"  Bark {i+1}/{n_barks} …", end=" ", flush=True)
        t0 = time.time()
        try:
            feat = rx.measure()
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        features_collected.append(feat)
        db.add_room(room_name, feat, vslam_pos=vslam_pos)
        elapsed = time.time() - t0

        if verbose:
            print(f"done ({elapsed:.2f}s)  T60={feat[0]:.3f}s  C80={feat[2]:.1f}dB")

        if i < n_barks - 1:
            time.sleep(INTER_BARK_SEC)

    return features_collected


def _check_consistency(features: list[np.ndarray]) -> float:
    """Return mean pairwise cosine similarity (higher = more consistent)."""
    if len(features) < 2:
        return 1.0
    vecs = np.stack([RoomDatabase._normalize(f) for f in features])
    gram = vecs @ vecs.T
    n    = len(vecs)
    off_diag = gram[np.triu_indices(n, k=1)]
    return float(off_diag.mean())


# ── Grid calibration (CAP 11) ─────────────────────────────────────────────────

def calibrate_grid(
    room_name: str,
    grid_rows: int = 3,
    grid_cols: int = 3,
    n_barks: int = BARKS_PER_POSITION,
    db: RoomDatabase | None = None,
):
    """Record grid_rows × grid_cols positions within a room (CAP 11)."""
    if db is None:
        db = RoomDatabase()

    row_labels = "ABCDEFGH"[:grid_rows]
    print(f"\nGrid calibration: {room_name}  ({grid_rows}×{grid_cols})")
    print(f"Positions: {', '.join(r+str(c+1) for r in row_labels for c in range(grid_cols))}\n")

    for row in row_labels:
        for col in range(1, grid_cols + 1):
            cell  = f"{row}{col}"
            label = f"{room_name}/{cell}"

            existing = db.count_vectors(label)
            if existing > 0:
                ans = input(
                    f"  Cell {cell}: {existing} vectors already stored. "
                    "Re-record? [y/N] "
                ).strip().lower()
                if ans != "y":
                    print(f"  Skipping {cell}.")
                    continue
                db.remove_room(label)

            input(f"\n  Move robot to position {cell}, then press ENTER …")
            print(f"  Recording {n_barks} barks for {label} …")
            feats = calibrate_room(label, n_barks=n_barks, db=db)

            consistency = _check_consistency(feats)
            flag = "" if consistency >= 0.85 else "  ⚠ low consistency"
            print(f"  {cell}: {len(feats)} stored  consistency={consistency:.3f}{flag}")

    print(f"\nGrid calibration complete for {room_name}.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="Acoustic room calibration tool",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--room",   metavar="NAME", help="Room name to calibrate")
    group.add_argument("--list",   action="store_true", help="List stored rooms")
    group.add_argument("--delete", metavar="NAME", help="Remove a room from DB")

    parser.add_argument("--barks", type=int, default=BARKS_PER_POSITION,
                        help=f"Barks per position (default {BARKS_PER_POSITION})")
    parser.add_argument("--grid",  metavar="RxC",
                        help="Grid calibration, e.g. --grid 3x3 (CAP 11)")
    args = parser.parse_args()

    db = RoomDatabase()

    # ── List ──────────────────────────────────────────────────────────────────
    if args.list:
        rooms = db.list_rooms()
        if not rooms:
            print("No rooms in database.")
            return
        print(f"{'Room':<30}  Vectors")
        print("-" * 42)
        for r in rooms:
            print(f"  {r:<28}  {db.count_vectors(r)}")
        return

    # ── Delete ────────────────────────────────────────────────────────────────
    if args.delete:
        existing = db.count_vectors(args.delete)
        if existing == 0:
            print(f"Room '{args.delete}' not found.")
            sys.exit(1)
        ans = input(f"Delete room '{args.delete}' ({existing} vectors)? [y/N] ").strip().lower()
        if ans == "y":
            db.remove_room(args.delete)
            print("Deleted.")
        return

    # ── Grid calibration ──────────────────────────────────────────────────────
    if args.grid:
        try:
            rows, cols = [int(x) for x in args.grid.lower().split("x")]
        except ValueError:
            print("--grid must be in RxC format, e.g. 3x3")
            sys.exit(1)
        calibrate_grid(args.room, grid_rows=rows, grid_cols=cols,
                       n_barks=args.barks, db=db)
        return

    # ── Single-room calibration ───────────────────────────────────────────────
    existing = db.count_vectors(args.room)
    if existing > 0:
        ans = input(
            f"Room '{args.room}' has {existing} stored vectors. "
            "Add more? [Y/n] "
        ).strip().lower()
        if ans == "n":
            return

    print(f"\nCalibrating room: {args.room}")
    print(f"Place robot in the centre of the room, then press ENTER …")
    input()

    print(f"Recording {args.barks} barks …\n")
    feats = calibrate_room(args.room, n_barks=args.barks, db=db)

    consistency = _check_consistency(feats)
    flag = "" if consistency >= 0.85 else "  ⚠ low consistency — reposition robot and try again"
    total = db.count_vectors(args.room)
    print(f"\nDone. {len(feats)} new vectors stored ({total} total).")
    print(f"Consistency: {consistency:.3f}{flag}")

    if total < 5:
        print(f"Tip: collect at least 5 vectors for reliable classification "
              f"({5 - total} more recommended).")


if __name__ == "__main__":
    _cli()
