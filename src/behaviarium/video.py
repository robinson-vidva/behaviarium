"""Cross-platform OpenCV video I/O and frame transforms. No assay logic.

Writes MJPG/AVI (portable on Mac + Windows). Container fps is preserved from the source for
playability; it is NOT the analysis fps — corrected fps lives in the manifest (see ingest).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

_FOURCC = cv2.VideoWriter_fourcc(*"MJPG")

_FLIP_CODES = {"horizontal": 1, "vertical": 0, "both": -1}


def probe_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def probe_dims(path: Path) -> tuple[int, int]:
    """(width, height) of the video in pixels."""
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()


def read_frame(path: Path, index: int = 0) -> np.ndarray:
    """Read a single frame (clamped to the valid range). Raises if the video can't be read."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = max(0, min(index, total - 1)) if total > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {index} from {path}")
        return frame
    finally:
        cap.release()


def _rotate_bound(frame: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate by an arbitrary angle, expanding the canvas so nothing is cropped."""
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), degrees, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    m[0, 2] += nw / 2.0 - cx
    m[1, 2] += nh / 2.0 - cy
    return cv2.warpAffine(frame, m, (nw, nh))


def apply_rotation(frame: np.ndarray, degrees: float, flip: str | None = None) -> np.ndarray:
    """Rotate (clockwise) then optionally flip. 90/180/270 are exact; others expand the canvas."""
    out = frame
    deg = degrees % 360
    if deg == 90:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    elif deg == 180:
        out = cv2.rotate(out, cv2.ROTATE_180)
    elif deg == 270:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif deg != 0:
        out = _rotate_bound(out, deg)
    if flip:
        if flip not in _FLIP_CODES:
            raise ValueError(f"Unsupported flip: {flip!r}")
        out = cv2.flip(out, _FLIP_CODES[flip])
    return out


def process_video(
    src: Path, dst: Path, fn: Callable[[np.ndarray], np.ndarray]
) -> int:
    """Apply ``fn`` to every frame of ``src``, writing results to ``dst``. Returns frame count.

    The writer is sized from the first transformed frame, so transforms that change the frame
    size (rotate, crop) are fine as long as they are consistent across frames.
    """
    src, dst = Path(src), Path(dst)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = fn(frame)
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(str(dst), _FOURCC, fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open VideoWriter for {dst}")
            writer.write(out)
            n += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return n
