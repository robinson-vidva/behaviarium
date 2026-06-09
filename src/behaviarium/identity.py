"""Stable video identity. ``video_id`` is a filename slug, deduped on collision — no reliance
on folder layout (Phase 7)."""

from __future__ import annotations

import re
from pathlib import Path

_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    s = _SLUG.sub("-", text.lower()).strip("-")
    return s or "video"


def make_video_id(filename: str, existing: set[str]) -> str:
    """Slug of the filename stem, with ``-2``, ``-3`` … appended on collision."""
    base = slugify(Path(filename).stem)
    vid, i = base, 2
    while vid in existing:
        vid = f"{base}-{i}"
        i += 1
    return vid
