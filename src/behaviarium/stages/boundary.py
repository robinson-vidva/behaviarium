"""Boundary stage — auto-detect the arena ROI with OpenCV. Assay-agnostic.

Params (shape hint, threshold, area bounds, pixel range) come from per-project config. Stores
generic ROI geometry (rect or circle) as JSON in the (video, boundary) manifest params, writes
a preview overlay PNG for human review, and sets the approval state to ``pending_review``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import BoundaryParams
from ..manifest import Approval
from ..paths import boundary_preview, video_output
from ..registry import register
from ..roi import draw_roi, normalize_geom
from ..stage import Stage, StageContext
from ..video import read_frame


def detect_roi(frame: np.ndarray, p: BoundaryParams) -> dict | None:
    """Detect the arena ROI on a single frame. Returns generic geometry or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = p.blur_ksize
    if k and k > 1:
        k = k + 1 if k % 2 == 0 else k  # kernel must be odd
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    _, binary = cv2.threshold(gray, p.threshold, 255, cv2.THRESH_BINARY)
    if p.pixel_min > 0 or p.pixel_max < 255:
        binary = cv2.bitwise_and(binary, cv2.inRange(gray, p.pixel_min, p.pixel_max))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = gray.shape
    area = float(h * w)
    lo, hi = p.min_area_frac * area, p.max_area_frac * area
    candidates = [c for c in contours if lo <= cv2.contourArea(c) <= hi]
    if not candidates:
        return None
    c = max(candidates, key=cv2.contourArea)
    if p.shape == "circle":
        (cx, cy), r = cv2.minEnclosingCircle(c)
        return {"shape": "circle", "cx": int(round(cx)), "cy": int(round(cy)), "r": int(round(r))}
    x, y, bw, bh = cv2.boundingRect(c)
    return {"shape": "rect", "x": int(x), "y": int(y), "w": int(bw), "h": int(bh)}


def write_preview(frame: np.ndarray, geom: dict, dst: Path) -> None:
    """Draw the ROI on the frame and save a PNG for human review."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), draw_roi(frame, geom))


@register("boundary")
class BoundaryStage(Stage):
    def inputs(self, ctx: StageContext) -> list[Path]:
        return [video_output(ctx.cfg, ctx.video, "rotate")]

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [boundary_preview(ctx.cfg, ctx.video)]

    def run(self, ctx: StageContext) -> None:
        p = ctx.cfg.project.boundary
        src = video_output(ctx.cfg, ctx.video, "rotate")
        if not src.exists():
            raise RuntimeError(f"rotate output missing; run rotate first: {src}")
        frame = read_frame(src, p.sample_frame)
        geom = detect_roi(frame, p)
        if geom is None:
            raise RuntimeError("boundary: no arena contour found within the configured area bounds")
        geom = normalize_geom(geom)
        preview = boundary_preview(ctx.cfg, ctx.video)
        write_preview(frame, geom, preview)
        ctx.manifest.set_params(
            ctx.video, self.name, {"roi": geom, "preview": str(preview), "shape": p.shape}
        )
        ctx.manifest.set_approval(ctx.video, self.name, Approval.PENDING_REVIEW)
