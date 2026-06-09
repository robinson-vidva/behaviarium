"""Stats stage — the REAL significance path (PROJECT scope).

Consumes the postprocess long tables and runs, for B-SOiD clusters AND chamber regions, a
between-group Wasserstein permutation test + Benjamini-Hochberg FDR. The output's p_value /
q_value / significant are the ONLY such columns in the pipeline (decision #4). Groups come
from the config design matrix via ``stats.group_factor``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import (
    postprocess_bsoid_long,
    postprocess_chamber_long,
    stats_bsoid_table,
    stats_chamber_table,
)
from ..registry import register
from ..stage import Stage, StageContext, StageScope
from ..stats import group_comparison


@register("stats")
class StatsStage(Stage):
    scope = StageScope.PROJECT

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [
            stats_bsoid_table(ctx.cfg, ".parquet"),
            stats_bsoid_table(ctx.cfg, ".csv"),
            stats_chamber_table(ctx.cfg, ".parquet"),
            stats_chamber_table(ctx.cfg, ".csv"),
        ]

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.cfg
        sp = cfg.project.stats
        if sp is None:
            raise RuntimeError("no stats params configured for this project")

        bsoid_long_pq = postprocess_bsoid_long(cfg, ".parquet")
        chamber_long_pq = postprocess_chamber_long(cfg, ".parquet")
        if not bsoid_long_pq.exists() or not chamber_long_pq.exists():
            raise RuntimeError("stats requires the postprocess aggregates; run postprocess first")

        bsoid_long = pd.read_parquet(bsoid_long_pq)
        chamber_long = pd.read_parquet(chamber_long_pq)

        cluster_stats = group_comparison(
            bsoid_long, "cluster", sp.group_factor, sp.metric, sp.n_permutations, sp.alpha, sp.seed
        )
        region_stats = group_comparison(
            chamber_long, "region", sp.group_factor, sp.metric, sp.n_permutations, sp.alpha, sp.seed
        )

        sb_pq, sb_csv = stats_bsoid_table(cfg, ".parquet"), stats_bsoid_table(cfg, ".csv")
        sc_pq, sc_csv = stats_chamber_table(cfg, ".parquet"), stats_chamber_table(cfg, ".csv")
        sb_pq.parent.mkdir(parents=True, exist_ok=True)
        cluster_stats.to_parquet(sb_pq, index=False)
        cluster_stats.to_csv(sb_csv, index=False)
        region_stats.to_parquet(sc_pq, index=False)
        region_stats.to_csv(sc_csv, index=False)

        ctx.manifest.set_params(
            ctx.video,
            self.name,
            {
                "group_factor": sp.group_factor,
                "metric": sp.metric,
                "n_permutations": sp.n_permutations,
                "alpha": sp.alpha,
                "seed": sp.seed,
                "n_significant_clusters": int(cluster_stats["significant"].sum()),
                "n_significant_regions": int(region_stats["significant"].sum()),
                "bsoid_stats_parquet": str(sb_pq),
                "bsoid_stats_csv": str(sb_csv),
                "chamber_stats_parquet": str(sc_pq),
                "chamber_stats_csv": str(sc_csv),
            },
        )
