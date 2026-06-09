"""Postprocess stage — aggregate per-video tidy outputs into project-level long tables.

PROJECT scope: runs once across all INCLUDED videos, producing one bsoid-clusters long table
and one chamber-occupancy long table (each Parquet + CSV), keyed by (Type,Class,Filename) +
parsed factor columns. Idempotent. Clear error if a video's upstream output is missing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import (
    bsoid_clusters_parquet,
    chamber_parquet,
    postprocess_bsoid_long,
    postprocess_chamber_long,
)
from ..registry import register
from ..runner import eligible_video_ids
from ..stage import Stage, StageContext, StageScope


def _aggregate(video_ids, per_video_path, cfg, manifest, what: str) -> pd.DataFrame:
    """Concatenate per-video outputs, attaching design-factor columns from each video's CURRENT
    tag (authoritative — so a video processed pre-tag and tagged later aggregates correctly)."""
    factor_names = cfg.project.design.factor_names()
    frames = []
    for vid in video_ids:
        p = per_video_path(cfg, vid)
        if not p.exists():
            raise RuntimeError(f"postprocess: missing {what} output for {vid}; run {what} first ({p})")
        df = pd.read_parquet(p)
        tag = manifest.get_tag(vid) or {}
        for name in factor_names:
            df[name] = tag.get(name)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    lead = [c for c in ["video_id", "filename", *factor_names] if c in out.columns]
    return out[lead + [c for c in out.columns if c not in lead]]


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
        video_ids = eligible_video_ids(cfg, ctx.manifest, self.name)  # included AND tagged
        if not video_ids:
            raise RuntimeError("postprocess: no included+tagged videos to aggregate")

        bsoid_long = _aggregate(video_ids, bsoid_clusters_parquet, cfg, ctx.manifest, "bsoid")
        chamber_long = _aggregate(video_ids, chamber_parquet, cfg, ctx.manifest, "chamber")

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
                "n_videos": len(video_ids),
                "bsoid_long_parquet": str(bl_pq),
                "bsoid_long_csv": str(bl_csv),
                "chamber_long_parquet": str(cl_pq),
                "chamber_long_csv": str(cl_csv),
            },
        )
