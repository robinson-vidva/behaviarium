"""Generic arena-ROI geometry. One representation serves every assay.

Geometry is a plain dict so it round-trips through the manifest as JSON:
    rect   -> {"shape": "rect",   "x": int, "y": int, "w": int, "h": int}
    circle -> {"shape": "circle", "cx": int, "cy": int, "r": int}
"""

from __future__ import annotations

import cv2
import numpy as np


def normalize_geom(geom: dict) -> dict:
    """Validate and coerce a geometry dict to ints. Raises ValueError on bad input."""
    shape = geom.get("shape")
    if shape == "rect":
        return {"shape": "rect", **{k: int(round(float(geom[k]))) for k in ("x", "y", "w", "h")}}
    if shape == "circle":
        return {"shape": "circle", **{k: int(round(float(geom[k]))) for k in ("cx", "cy", "r")}}
    raise ValueError(f"Unsupported ROI shape: {shape!r}")


def roi_to_mask(geom: dict, shape_hw: tuple[int, int]) -> np.ndarray:
    """Binary mask (uint8, 255 inside the ROI) of size ``shape_hw`` = (height, width)."""
    g = normalize_geom(geom)
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if g["shape"] == "rect":
        cv2.rectangle(mask, (g["x"], g["y"]), (g["x"] + g["w"], g["y"] + g["h"]), 255, -1)
    else:
        cv2.circle(mask, (g["cx"], g["cy"]), g["r"], 255, -1)
    return mask


def draw_roi(frame: np.ndarray, geom: dict, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Return a copy of ``frame`` with the ROI outline drawn on it (for preview overlays)."""
    g = normalize_geom(geom)
    out = frame.copy()
    if g["shape"] == "rect":
        cv2.rectangle(out, (g["x"], g["y"]), (g["x"] + g["w"], g["y"] + g["h"]), color, thickness)
    else:
        cv2.circle(out, (g["cx"], g["cy"]), g["r"], color, thickness)
    return out


def bbox_of(geom: dict, shape_hw: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
    """Bounding box (x, y, w, h). For circles, the enclosing square; clamped to the frame
    when ``shape_hw`` is given."""
    g = normalize_geom(geom)
    if g["shape"] == "rect":
        x, y, w, h = g["x"], g["y"], g["w"], g["h"]
    else:
        x, y, w, h = g["cx"] - g["r"], g["cy"] - g["r"], 2 * g["r"], 2 * g["r"]
    if shape_hw is not None:
        H, W = shape_hw
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        x, y, w, h = x0, y0, max(0, x1 - x0), max(0, y1 - y0)
    return x, y, w, h


def apply_mask(frame: np.ndarray, geom: dict, fill_value: int = 0, crop: bool = False) -> np.ndarray:
    """Zero (or ``fill_value``) every pixel outside the ROI; optionally crop to the bbox."""
    mask = roi_to_mask(geom, frame.shape[:2])
    if fill_value == 0:
        out = cv2.bitwise_and(frame, frame, mask=mask)
    else:
        out = np.full_like(frame, fill_value)
        out[mask == 255] = frame[mask == 255]
    if crop:
        x, y, w, h = bbox_of(geom, frame.shape[:2])
        out = out[y:y + h, x:x + w]
    return out
