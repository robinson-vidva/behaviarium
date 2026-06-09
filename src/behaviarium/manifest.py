"""Single sqlite manifest (WAL mode) — the one source of job state.

Two tables in one database file:
- ``videos``   : one row per video (the ``(type, class, filename)`` join key), holds the
                 corrected fps single source of truth.
- ``manifest`` : one row per ``(video, stage)`` with status / timestamps / params / error.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class Status(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"  # runner result for excluded videos (not persisted)


class Approval(str, Enum):
    """Human review state for a reviewable stage (e.g. boundary ROI)."""

    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class VideoKey:
    """The (Type, Class, Filename) join key. ``klass`` maps to DB column ``class``."""

    type: str
    klass: str
    filename: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    type        TEXT NOT NULL,
    class       TEXT NOT NULL,
    filename    TEXT NOT NULL,
    path        TEXT,
    frame_count INTEGER,
    fps         REAL,
    include     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    UNIQUE(type, class, filename)
);
CREATE TABLE IF NOT EXISTS manifest (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    type        TEXT NOT NULL,
    class       TEXT NOT NULL,
    filename    TEXT NOT NULL,
    stage       TEXT NOT NULL,
    status      TEXT NOT NULL,
    params      TEXT,
    approval    TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    UNIQUE(type, class, filename, stage)
);
"""

# (table, column, column declaration) — added to pre-existing databases on init().
_MIGRATIONS = [
    ("videos", "include", "INTEGER NOT NULL DEFAULT 1"),
    ("manifest", "approval", "TEXT"),
]


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
            for table, column, decl in _MIGRATIONS:
                cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
                if column not in cols:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    # -- videos -----------------------------------------------------------------
    def upsert_video(
        self, key: VideoKey, path: Path | str, frame_count: int | None, fps: float | None
    ) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO videos (type, class, filename, path, frame_count, fps,
                                    created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(type, class, filename) DO UPDATE SET
                    path=excluded.path,
                    frame_count=excluded.frame_count,
                    fps=excluded.fps,
                    updated_at=excluded.updated_at
                """,
                (key.type, key.klass, key.filename, str(path), frame_count, fps, now, now),
            )

    def get_video(self, key: VideoKey) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE type=? AND class=? AND filename=?",
                (key.type, key.klass, key.filename),
            ).fetchone()
        return dict(row) if row else None

    def list_videos(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM videos ORDER BY type, class, filename"
            ).fetchall()
        return [dict(r) for r in rows]

    def set_include(self, key: VideoKey, include: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE videos SET include=?, updated_at=?
                WHERE type=? AND class=? AND filename=?
                """,
                (1 if include else 0, _now(), key.type, key.klass, key.filename),
            )

    # -- manifest (video, stage) ------------------------------------------------
    def upsert(
        self,
        key: VideoKey,
        stage: str,
        status: Status = Status.PENDING,
        params: dict | None = None,
    ) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO manifest (type, class, filename, stage, status, params,
                                      created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(type, class, filename, stage) DO UPDATE SET
                    status=excluded.status,
                    params=excluded.params,
                    updated_at=excluded.updated_at
                """,
                (
                    key.type,
                    key.klass,
                    key.filename,
                    stage,
                    Status(status).value,
                    json.dumps(params) if params is not None else None,
                    now,
                    now,
                ),
            )

    def set_status(
        self, key: VideoKey, stage: str, status: Status, error: str | None = None
    ) -> None:
        now = _now()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE manifest SET status=?, error=?, updated_at=?
                WHERE type=? AND class=? AND filename=? AND stage=?
                """,
                (Status(status).value, error, now, key.type, key.klass, key.filename, stage),
            )
            if cur.rowcount == 0:
                conn.execute(
                    """
                    INSERT INTO manifest (type, class, filename, stage, status, error,
                                          created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key.type,
                        key.klass,
                        key.filename,
                        stage,
                        Status(status).value,
                        error,
                        now,
                        now,
                    ),
                )

    def get_status(self, key: VideoKey, stage: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT status FROM manifest
                WHERE type=? AND class=? AND filename=? AND stage=?
                """,
                (key.type, key.klass, key.filename, stage),
            ).fetchone()
        return row["status"] if row else None

    def get_row(self, key: VideoKey, stage: str) -> dict | None:
        """Full manifest row for one (video, stage); ``params`` is parsed JSON (or None)."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM manifest
                WHERE type=? AND class=? AND filename=? AND stage=?
                """,
                (key.type, key.klass, key.filename, stage),
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["params"] = json.loads(d["params"]) if d["params"] else None
        return d

    def set_params(self, key: VideoKey, stage: str, params: dict) -> None:
        """Update only the params JSON for a (video, stage) row, leaving status untouched."""
        now = _now()
        payload = json.dumps(params)
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE manifest SET params=?, updated_at=?
                WHERE type=? AND class=? AND filename=? AND stage=?
                """,
                (payload, now, key.type, key.klass, key.filename, stage),
            )
            if cur.rowcount == 0:
                conn.execute(
                    """
                    INSERT INTO manifest (type, class, filename, stage, status, params,
                                          created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (key.type, key.klass, key.filename, stage, Status.PENDING.value,
                     payload, now, now),
                )

    def set_approval(self, key: VideoKey, stage: str, approval: Approval | str) -> None:
        now = _now()
        value = Approval(approval).value
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE manifest SET approval=?, updated_at=?
                WHERE type=? AND class=? AND filename=? AND stage=?
                """,
                (value, now, key.type, key.klass, key.filename, stage),
            )
            if cur.rowcount == 0:
                conn.execute(
                    """
                    INSERT INTO manifest (type, class, filename, stage, status, approval,
                                          created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (key.type, key.klass, key.filename, stage, Status.PENDING.value,
                     value, now, now),
                )

    def get_approval(self, key: VideoKey, stage: str) -> str | None:
        row = self.get_row(key, stage)
        return row["approval"] if row else None

    def query(
        self,
        type: str | None = None,
        klass: str | None = None,
        filename: str | None = None,
        stage: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        clauses, args = [], []
        for col, val in (
            ("type", type),
            ("class", klass),
            ("filename", filename),
            ("stage", stage),
            ("status", status),
        ):
            if val is not None:
                clauses.append(f"{col}=?")
                args.append(val)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM manifest{where} ORDER BY type, class, filename, stage", args
            ).fetchall()
        return [dict(r) for r in rows]
