"""
SQLite-backed face database.

Stores 512-d ArcFace embeddings (float32) as binary blobs alongside person names.
Multiple embeddings per person are supported; identification uses cosine similarity
against every stored embedding and returns the best match above the threshold.

Usage:
    db = FaceDatabase()
    db.add_person("Alice", embedding_array)
    name, score = db.identify(query_embedding)
    db.list_people()         # → ["Alice", "Bob"]
    db.remove_person("Bob")
"""

import sqlite3
from pathlib import Path

import numpy as np

_DEFAULT_DB = Path.home() / ".bpx" / "faces.db"


class FaceDatabase:
    def __init__(self, db_path: str | Path = _DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self):
        with self._conn() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    name       TEXT    NOT NULL,
                    embedding  BLOB    NOT NULL,
                    added_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_person(self, name: str, embedding: np.ndarray):
        blob = embedding.astype(np.float32).tobytes()
        with self._conn() as con:
            con.execute(
                "INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, blob)
            )

    def remove_person(self, name: str):
        with self._conn() as con:
            con.execute("DELETE FROM faces WHERE name = ?", (name,))

    # ── Read ──────────────────────────────────────────────────────────────────

    def identify(
        self, embedding: np.ndarray, threshold: float = 0.45
    ) -> tuple[str | None, float]:
        """Return (name, similarity) or (None, best_score) if below threshold."""
        rows = self._all_embeddings()
        if not rows:
            return None, 0.0

        q = self._normalize(embedding)
        best_name, best_sim = None, -1.0

        for name, stored_bytes in rows:
            stored = np.frombuffer(stored_bytes, dtype=np.float32)
            sim    = float(np.dot(q, self._normalize(stored)))
            if sim > best_sim:
                best_sim  = sim
                best_name = name

        if best_sim < threshold:
            return None, round(best_sim, 4)
        return best_name, round(best_sim, 4)

    def list_people(self) -> list[str]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT DISTINCT name FROM faces ORDER BY name"
            ).fetchall()
        return [r[0] for r in rows]

    def count(self) -> int:
        with self._conn() as con:
            return con.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _all_embeddings(self) -> list[tuple[str, bytes]]:
        with self._conn() as con:
            return con.execute("SELECT name, embedding FROM faces").fetchall()

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-8)
