"""Postprocess stage — aggregate per-video tidy outputs into project-level long tables.

PROJECT scope: runs once across all INCLUDED videos, producing one bsoid-clusters long table
and one chamber-occupancy long table (each Parquet + CSV), keyed by (Type,Class,Filename) +
parsed factor columns. Idempotent. Clear error if a video's upstream output is missing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..manifest import VideoKey
from ..paths import (
    bsoid_clusters_parquet,
    chamber_parquet,
    postprocess_bsoid_long,
    postprocess_chamber_long,
)
from ..registry import register
from ..stage import Stage, StageContext, StageScope


def _included_keys(ctx: StageContext) -> list[VideoKey]:
    return [
        VideoKey(v["type"], v["class"], v["filename"])
        for v in ctx.manifest.list_videos()
        if v.get("include", 1)
    ]


def _aggregate(keys, per_video_path, cfg, what: str) -> pd.DataFrame:
    frames = []
    for key in keys:
        p = per_video_path(cfg, key)
        if not p.exists():
            raise RuntimeError(
                f"postprocess: missing {what} output for {key.type}/{key.klass}/{key.filename}; "
                f"run {what} first ({p})"
            )
        frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True)


@register("postprocess")
class PostprocessStage(Stage):
    scope = StageScope.PROJECT

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [
            postprocess_bsoid_long(ctx.cfg, ".parquet"),
            postprocess_bsoid_long(ctx.cfg, ".csv"),
            postprocess_chamber_long(ctx.cfg, ".parquet"),
            postprocess_chamber_long(ctx.cfg, ".csv"),
        ]

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.cfg
        keys = _included_keys(ctx)
        if not keys:
            raise RuntimeError("postprocess: no included videos to aggregate")

        bsoid_long = _aggregate(keys, bsoid_clusters_parquet, cfg, "bsoid")
        chamber_long = _aggregate(keys, chamber_parquet, cfg, "chamber")

        bl_pq, bl_csv = postprocess_bsoid_long(cfg, ".parquet"), postprocess_bsoid_long(cfg, ".csv")
        cl_pq, cl_csv = postprocess_chamber_long(cfg, ".parquet"), postprocess_chamber_long(cfg, ".csv")
        bl_pq.parent.mkdir(parents=True, exist_ok=True)
        bsoid_long.to_parquet(bl_pq, index=False)
        bsoid_long.to_csv(bl_csv, index=False)
        chamber_long.to_parquet(cl_pq, index=False)
        chamber_long.to_csv(cl_csv, index=False)

        ctx.manifest.set_params(
            ctx.video,
            self.name,
            {
                "n_videos": len(keys),
                "bsoid_long_parquet": str(bl_pq),
                "bsoid_long_csv": str(bl_csv),
                "chamber_long_parquet": str(cl_pq),
                "chamber_long_csv": str(cl_csv),
            },
        )
