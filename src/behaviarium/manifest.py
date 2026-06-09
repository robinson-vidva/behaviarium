"""Single sqlite manifest (WAL mode) — the one source of job state, per project.

Primary identity is a stable ``video_id`` (Phase 7; replaces the old (Type,Class,Filename)
triple). Two tables:
- ``videos``   : one row per video — ``video_id``, ``filename``, ``source_path`` (original),
                 ``current_path`` (where it is now; == source until reorg), corrected fps,
                 ``include`` flag, and ``tag`` (JSON: one design-factor level per factor).
- ``manifest`` : one row per ``(video_id, stage)`` with status / params / approval / error.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

# Reserved video_id for project-level (aggregate) stage rows.
PROJECT_ID = "__project__"


class Status(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"  # runner result for excluded/untagged videos (not persisted)


class Approval(str, Enum):
    """Human review state for a reviewable stage (e.g. boundary ROI)."""

    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id     TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    current_path TEXT NOT NULL,
    frame_count  INTEGER,
    fps          REAL,
    include      INTEGER NOT NULL DEFAULT 1,
    tag          TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS manifest (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id    TEXT NOT NULL,
    stage       TEXT NOT NULL,
    status      TEXT NOT NULL,
    params      TEXT,
    approval    TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    UNIQUE(video_id, stage)
);
"""


class Manifest:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # -- videos -----------------------------------------------------------------
    def upsert_video(
        self,
        video_id: str,
        filename: str,
        source_path: Path | str,
        current_path: Path | str,
        frame_count: int | None,
        fps: float | None,
    ) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO videos (video_id, filename, source_path, current_path,
                                    frame_count, fps, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    filename=excluded.filename,
                    source_path=excluded.source_path,
                    current_path=excluded.current_path,
                    frame_count=excluded.frame_count,
                    fps=excluded.fps,
                    updated_at=excluded.updated_at
                """,
                (video_id, filename, str(source_path), str(current_path), frame_count, fps, now, now),
            )

    def get_video(self, video_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM videos WHERE video_id=?", (video_id,)).fetchone()
        return self._video_dict(row) if row else None

    def get_video_by_source_path(self, source_path: Path | str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE source_path=?", (str(source_path),)
            ).fetchone()
        return self._video_dict(row) if row else None

    def list_videos(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM videos ORDER BY video_id").fetchall()
        return [self._video_dict(r) for r in rows]

    @staticmethod
    def _video_dict(row) -> dict:
        d = dict(row)
        d["tag"] = json.loads(d["tag"]) if d.get("tag") else None
        return d

    def set_include(self, video_id: str, include: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE videos SET include=?, updated_at=? WHERE video_id=?",
                (1 if include else 0, _now(), video_id),
            )

    def set_tag(self, video_id: str, tag: dict | None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE videos SET tag=?, updated_at=? WHERE video_id=?",
                (json.dumps(tag) if tag else None, _now(), video_id),
            )

    def get_tag(self, video_id: str) -> dict | None:
        rec = self.get_video(video_id)
        return rec["tag"] if rec else None

    def set_current_path(self, video_id: str, current_path: Path | str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE videos SET current_path=?, updated_at=? WHERE video_id=?",
                (str(current_path), _now(), video_id),
            )

    # -- manifest (video_id, stage) ---------------------------------------------
    def upsert(
        self, video_id: str, stage: str, status: Status = Status.PENDING, params: dict | None = None
    ) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO manifest (video_id, stage, status, params, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id, stage) DO UPDATE SET
                    status=excluded.status, params=excluded.params, updated_at=excluded.updated_at
                """,
                (video_id, stage, Status(status).value,
                 json.dumps(params) if params is not None else None, now, now),
            )

    def set_status(self, video_id: str, stage: str, status: Status, error: str | None = None) -> None:
        now = _now()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE manifest SET status=?, error=?, updated_at=? WHERE video_id=? AND stage=?",
                (Status(status).value, error, now, video_id, stage),
            )
            if cur.rowcount == 0:
                conn.execute(
                    """
                    INSERT INTO manifest (video_id, stage, status, error, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (video_id, stage, Status(status).value, error, now, now),
                )

    def get_status(self, video_id: str, stage: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM manifest WHERE video_id=? AND stage=?", (video_id, stage)
            ).fetchone()
        return row["status"] if row else None

    def get_row(self, video_id: str, stage: str) -> dict | None:
        """Full manifest row for one (video_id, stage); ``params`` is parsed JSON (or None)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM manifest WHERE video_id=? AND stage=?", (video_id, stage)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["params"] = json.loads(d["params"]) if d["params"] else None
        return d

    def set_params(self, video_id: str, stage: str, params: dict) -> None:
        """Update only the params JSON for a (video_id, stage) row, leaving status untouched."""
        now = _now()
        payload = json.dumps(params)
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE manifest SET params=?, updated_at=? WHERE video_id=? AND stage=?",
                (payload, now, video_id, stage),
            )
            if cur.rowcount == 0:
                conn.execute(
                    """
                    INSERT INTO manifest (video_id, stage, status, params, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (video_id, stage, Status.PENDING.value, payload, now, now),
                )

    def set_approval(self, video_id: str, stage: str, approval: Approval | str) -> None:
        now = _now()
        value = Approval(approval).value
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE manifest SET approval=?, updated_at=? WHERE video_id=? AND stage=?",
                (value, now, video_id, stage),
            )
            if cur.rowcount == 0:
                conn.execute(
                    """
                    INSERT INTO manifest (video_id, stage, status, approval, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (video_id, stage, Status.PENDING.value, value, now, now),
                )

    def get_approval(self, video_id: str, stage: str) -> str | None:
        row = self.get_row(video_id, stage)
        return row["approval"] if row else None

    def query(
        self, video_id: str | None = None, stage: str | None = None, status: str | None = None
    ) -> list[dict]:
        clauses, args = [], []
        for col, val in (("video_id", video_id), ("stage", stage), ("status", status)):
            if val is not None:
                clauses.append(f"{col}=?")
                args.append(val)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM manifest{where} ORDER BY video_id, stage", args
            ).fetchall()
        return [dict(r) for r in rows]
