"""Chamber stage — spatial occupancy from DLC tracks. Assay-agnostic.

Reads the dlc stage output (raw or filtered, honoring the existing naming) for a configured
tracking bodypart BY NAME, assigns each frame to a config-declared region (relative to the
approved boundary ROI), and computes per-region frame counts + time-in-region.

Decision #3: ALL time math uses corrected_fps = actual_frame_count/600, read from the manifest
video row. No 84 / 30 / any literal framerate appears here.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .. import dlc_io
from ..class_parser import parse_class
from ..config import ChamberRegion
from ..paths import chamber_csv, chamber_parquet, dlc_output_path, video_output
from ..registry import register
from ..roi import bbox_of
from ..stage import Stage, StageContext
from ..video import probe_dims


def _in_region(nx: np.ndarray, ny: np.ndarray, region: ChamberRegion) -> np.ndarray:
    """Boolean mask of points inside ``region`` (coords are fractions of the ROI bbox)."""
    if region.shape == "circle":
        return (nx - region.cx) ** 2 + (ny - region.cy) ** 2 <= region.r ** 2
    return (nx >= region.x) & (nx < region.x + region.w) & (ny >= region.y) & (ny < region.y + region.h)


def assign_regions(nx: np.ndarray, ny: np.ndarray, regions: list[ChamberRegion]) -> np.ndarray:
    """Assign each point to the first region that contains it; NaN/unmatched -> 'none'."""
    labels = np.full(nx.shape, "none", dtype=object)
    assigned = np.isnan(nx) | np.isnan(ny)
    for reg in regions:
        m = _in_region(nx, ny, reg) & ~assigned
        labels[m] = reg.name
        assigned |= m
    return labels


def _roi_bbox_in_analysis(cfg, key, manifest, frame_dims) -> tuple[float, float, float, float]:
    """ROI bbox in the analysis (dlc/mask) frame's coordinate space.

    With mask.crop the analysis frame IS the ROI bbox -> (0,0,W,H). Without crop, the ROI sits
    at its detected position in the full frame, so use the stored boundary geometry."""
    fw, fh = frame_dims
    if cfg.project.mask.crop:
        return 0.0, 0.0, float(fw), float(fh)
    brow = manifest.get_row(key, "boundary")
    geom = (brow.get("params") or {}).get("roi") if brow else None
    if not geom:
        raise RuntimeError("chamber needs the boundary ROI (no crop); run/approve boundary first")
    x, y, w, h = bbox_of(geom, (fh, fw))
    return float(x), float(y), float(w), float(h)


@register("chamber")
class ChamberStage(Stage):
    def inputs(self, ctx: StageContext) -> list[Path]:
        return [dlc_output_path(ctx.cfg, ctx.video, ctx.cfg.project.dlc.filter.enabled)]

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [chamber_parquet(ctx.cfg, ctx.video), chamber_csv(ctx.cfg, ctx.video)]

    def run(self, ctx: StageContext) -> None:
        cfg, key = ctx.cfg, ctx.video
        if cfg.project.chamber is None:
            raise RuntimeError("no chamber region scheme configured for this project")

        dlc_path = dlc_output_path(cfg, key, cfg.project.dlc.filter.enabled)
        if not dlc_path.exists():
            raise RuntimeError(f"chamber requires the dlc output; run dlc first: {dlc_path}")

        rec = ctx.video_record()
        fps = rec.get("fps") if rec else None  # corrected_fps = frame_count/600 (decision #3)
        if not fps or fps <= 0:
            raise RuntimeError(f"missing corrected_fps for {key}; run ingest first")

        # tracking point BY NAME via the multiindex reader (never positional x.1/x.11)
        df = dlc_io.read_dlc_csv(dlc_path)
        bp = dlc_io.get_bodypart(df, cfg.project.chamber.tracking_bodypart)
        px, py = bp["x"].to_numpy(dtype=float), bp["y"].to_numpy(dtype=float)

        mask_video = video_output(cfg, key, "mask")
        rx, ry, rw, rh = _roi_bbox_in_analysis(cfg, key, ctx.manifest, probe_dims(mask_video))
        nx = (px - rx) / rw
        ny = (py - ry) / rh

        regions = cfg.project.chamber.regions
        labels = assign_regions(nx, ny, regions)
        counts = Counter(labels.tolist())
        total = int(len(labels))

        factors = parse_class(key.klass, cfg.project.class_parser)

        names = [r.name for r in regions]
        if counts.get("none", 0):
            names = names + ["none"]
        rows = []
        for name in names:
            c = int(counts.get(name, 0))
            rows.append(
                {
                    "Type": key.type,
                    "Class": key.klass,
                    "Filename": key.filename,
                    **factors,
                    "region": name,
                    "frame_count": c,
                    "time_s": c / fps,
                    "fraction": (c / total) if total else 0.0,
                }
            )
        factor_cols = list(factors.keys())
        long_df = pd.DataFrame(rows)[
            ["Type", "Class", "Filename", *factor_cols, "region", "frame_count", "time_s", "fraction"]
        ]

        pq, csv = chamber_parquet(cfg, key), chamber_csv(cfg, key)
        pq.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_parquet(pq, index=False)  # for R (separate manual step); Python never calls R
        long_df.to_csv(csv, index=False)

        ctx.manifest.set_params(
            key,
            self.name,
            {
                "tracking_bodypart": cfg.project.chamber.tracking_bodypart,
                "corrected_fps": fps,
                "total_frames": total,
                "factors": factors,
                "regions": {n: int(counts.get(n, 0)) for n in names},
                "parquet": str(pq),
                "csv": str(csv),
            },
        )
