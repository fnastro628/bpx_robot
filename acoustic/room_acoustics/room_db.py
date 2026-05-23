"""
CAP 10 — Room Acoustic Fingerprint Database

SQLite-backed storage for 15-dimensional RIR feature vectors.
Follows the same pattern as perception/face_recognition/face_db.py.

Multiple feature vectors per room are supported (minimum 5 recommended).
Identification uses cosine similarity averaged over all stored vectors for
that room against the query vector.

Schema:
  rooms(id, name, feature_blob, vslam_x, vslam_y, vslam_yaw, recorded_at)

Usage:
    db = RoomDatabase()
    db.add_room("living_room", feature_vec)
    name, score = db.identify(query_vec)
    db.list_rooms()    # → ["bathroom", "bedroom", "living_room"]
    db.remove_room("bathroom")
"""

import sqlite3
from pathlib import Path

import numpy as np

_DEFAULT_DB = Path.home() / ".bpx" / "rooms.db"
FEATURE_DIM = 15


class RoomDatabase:
    def __init__(self, db_path: str | Path = _DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self):
        with self._conn() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS rooms (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT    NOT NULL,
                    feature_blob BLOB   NOT NULL,
                    vslam_x     REAL    DEFAULT 0.0,
                    vslam_y     REAL    DEFAULT 0.0,
                    vslam_yaw   REAL    DEFAULT 0.0,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_room(
        self,
        name: str,
        features: np.ndarray,
        vslam_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """Store one feature vector for a named room.

        Call this multiple times (≥5) per room for reliable classification.
        """
        if len(features) != FEATURE_DIM:
            raise ValueError(f"Expected {FEATURE_DIM}-d feature vector, got {len(features)}")
        blob = features.astype(np.float32).tobytes()
        x, y, yaw = vslam_pos
        with self._conn() as con:
            con.execute(
                "INSERT INTO rooms (name, feature_blob, vslam_x, vslam_y, vslam_yaw) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, blob, float(x), float(y), float(yaw)),
            )

    def remove_room(self, name: str):
        with self._conn() as con:
            con.execute("DELETE FROM rooms WHERE name = ?", (name,))

    # ── Read ──────────────────────────────────────────────────────────────────

    def identify(
        self,
        features: np.ndarray,
        threshold: float = 0.80,
    ) -> tuple[str | None, float]:
        """Return (room_name, similarity) or (None, best_score) below threshold.

        Each room's similarity is the average cosine similarity across all its
        stored feature vectors. The room with the highest average wins.
        """
        rows = self._all_features()
        if not rows:
            return None, 0.0

        q = self._normalize(features)

        # Accumulate per-room sums and counts
        room_scores: dict[str, list[float]] = {}
        for name, blob in rows:
            stored = np.frombuffer(blob, dtype=np.float32)
            sim    = float(np.dot(q, self._normalize(stored)))
            room_scores.setdefault(name, []).append(sim)

        best_name  = max(room_scores, key=lambda n: np.mean(room_scores[n]))
        best_score = float(np.mean(room_scores[best_name]))

        if best_score < threshold:
            return None, round(best_score, 4)
        return best_name, round(best_score, 4)

    def list_rooms(self) -> list[str]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT DISTINCT name FROM rooms ORDER BY name"
            ).fetchall()
        return [r[0] for r in rows]

    def count_vectors(self, name: str | None = None) -> int:
        """Total stored feature vectors; optionally filtered by room name."""
        with self._conn() as con:
            if name:
                return con.execute(
                    "SELECT COUNT(*) FROM rooms WHERE name = ?", (name,)
                ).fetchone()[0]
            return con.execute("SELECT COUNT(*) FROM rooms").fetchone()[0]

    def room_centroid(self, name: str) -> np.ndarray:
        """Return the average feature vector for a room (for diagnostics)."""
        blobs = self._features_for_room(name)
        if not blobs:
            raise KeyError(f"Room '{name}' not in database")
        vecs = np.stack([np.frombuffer(b, dtype=np.float32) for b in blobs])
        return vecs.mean(axis=0)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _all_features(self) -> list[tuple[str, bytes]]:
        with self._conn() as con:
            return con.execute("SELECT name, feature_blob FROM rooms").fetchall()

    def _features_for_room(self, name: str) -> list[bytes]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT feature_blob FROM rooms WHERE name = ?", (name,)
            ).fetchall()
        return [r[0] for r in rows]

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-8)
