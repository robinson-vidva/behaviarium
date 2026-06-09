"""Reorganize a source video into its per-video folder — an EXPLICIT user action (Phase 7).

Mode: ``copy`` (default, originals untouched) | ``move`` | ``symlink``. Idempotent (detects
already-reorganized), never overwrites an existing file, and verifies the copy/move succeeded
before updating the manifest's ``current_path``. If not reorganized, the manifest keeps
pointing at the original ``source_path``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import Config
from .manifest import Manifest
from .paths import video_dir

MODES = ("copy", "move", "symlink")


def reorg_video(cfg: Config, manifest: Manifest, video_id: str, mode: str = "copy") -> str:
    if mode not in MODES:
        raise ValueError(f"reorg mode must be one of {MODES}, got {mode!r}")
    rec = manifest.get_video(video_id)
    if rec is None:
        raise RuntimeError(f"unknown video_id {video_id!r}; run ingest first")

    current = Path(rec["current_path"])
    dest = video_dir(cfg, video_id) / rec["filename"]

    if dest.exists():
        if current.resolve() == dest.resolve():
            return "already-reorganized"  # idempotent
        raise FileExistsError(f"refusing to overwrite existing file: {dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(current, dest)
    elif mode == "move":
        shutil.move(str(current), str(dest))
    else:  # symlink
        dest.symlink_to(current.resolve())

    if not dest.exists():
        raise RuntimeError(f"reorg {mode} failed to produce {dest}")
    manifest.set_current_path(video_id, dest)
    return mode
