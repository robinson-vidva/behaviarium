from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


def write_synthetic_video(path: Path, n_frames: int, size: tuple[int, int] = (64, 64)) -> Path:
    """Write a tiny MJPG/AVI clip with a known frame count (no real video needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, size)
    assert writer.isOpened(), f"VideoWriter failed to open {path}"
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def write_rect_video(
    path: Path,
    n_frames: int = 5,
    size: tuple[int, int] = (200, 160),
    rect: tuple[int, int, int, int] = (40, 30, 100, 80),
    bright: int = 255,
) -> Path:
    """Write a clip with a known bright rectangle (the 'arena') on a dark background.

    ``size`` is (width, height); ``rect`` is (x, y, w, h)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    x, y, rw, rh = rect
    base = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(base, (x, y), (x + rw, y + rh), (bright, bright, bright), -1)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h))
    assert writer.isOpened(), f"VideoWriter failed to open {path}"
    for _ in range(n_frames):
        writer.write(base.copy())
    writer.release()
    return path


@pytest.fixture
def make_video():
    return write_synthetic_video


@pytest.fixture
def make_rect_video():
    return write_rect_video
