"""B-SOiD stage — behavioral clustering. Engine-aware, one interface, two backends.

- ``real``: lazily imports the configured B-SOiD module ONLY here (never breaks the Mac base
  install). Loads the trained model, predicts the per-offset streams, reconstructs per-frame
  labels with the shared frameshift module.
- ``stub``: synthetic per-10Hz cluster predictions over range(n_clusters); no heavy deps, so
  the whole reconstruction path is exercised and tested on Mac.

If ``engine=real`` but the B-SOiD import fails, it falls back to the stub (import failure only;
real runtime errors are surfaced). Per-frame labels come from the frameshift + flatten('F')
reconstruction (decision #2), never block-repeat. n_shift is per-video from corrected_fps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import dlc_io
from ..bsoid_reconstruct import n_shift_for_fps, offset_lengths, reconstruct_labels
from ..class_parser import parse_class
from ..config import resolve_bsoid_model
from ..paths import (
    bsoid_clusters_csv,
    bsoid_clusters_parquet,
    bsoid_labels_csv,
    bsoid_labels_parquet,
    dlc_output_path,
)
from ..registry import register
from ..stage import Stage, StageContext


def _run_stub(n_frames: int, n_shift: int, n_clusters: int) -> list[np.ndarray]:
    """Synthetic per-offset ~10Hz prediction streams over range(n_clusters)."""
    rng = np.random.default_rng(0)
    return [
        rng.integers(0, n_clusters, size=length)
        for length in offset_lengths(n_frames, n_shift)
    ]


def _run_real(cfg, p, pose_df: pd.DataFrame, fps: float) -> list[np.ndarray]:
    """Real B-SOiD backend. INTEGRATION POINT — confirm against your B-SOiD fork on Windows.

    Confirmed from YttriLab/B-SOID source: ``bsoid_predict(feats, scaler, model)`` returns a
    list of per-offset label streams. The feature extraction call and the saved-model bundle
    format are fork/version specific — confirm before the first real run."""
    import importlib  # lazy

    import joblib  # lazy

    module = importlib.import_module(p.module)
    model_path = resolve_bsoid_model(cfg)
    if model_path is None or not model_path.exists():
        raise RuntimeError(
            f"B-SOiD model not found: {model_path} (set BEHAVIARIUM_BSOID_MODEL or bsoid.model_path)"
        )
    bundle = joblib.load(model_path)
    scaler, model = bundle if isinstance(bundle, (tuple, list)) else (bundle.get("scaler"), bundle.get("model"))

    # CONFIRM ON WINDOWS: pose array shape expected by bsoid_extract and the model bundle layout.
    feats = module.bsoid_extract([pose_df.to_numpy()], fps)
    streams = module.bsoid_predict(feats, scaler, model)
    return [np.asarray(s) for s in streams]


@register("bsoid")
class BsoidStage(Stage):
    def inputs(self, ctx: StageContext) -> list[Path]:
        return [dlc_output_path(ctx.cfg, ctx.video, ctx.cfg.project.dlc.filter.enabled)]

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [
            bsoid_labels_parquet(ctx.cfg, ctx.video),
            bsoid_labels_csv(ctx.cfg, ctx.video),
            bsoid_clusters_parquet(ctx.cfg, ctx.video),
            bsoid_clusters_csv(ctx.cfg, ctx.video),
        ]

    def run(self, ctx: StageContext) -> None:
        cfg, key = ctx.cfg, ctx.video
        p = cfg.project.bsoid
        if p is None:
            raise RuntimeError("no bsoid params configured for this project")

        dlc_path = dlc_output_path(cfg, key, cfg.project.dlc.filter.enabled)
        if not dlc_path.exists():
            raise RuntimeError(f"bsoid requires the dlc output; run dlc first: {dlc_path}")

        rec = ctx.video_record()
        fps = rec.get("fps") if rec else None  # corrected_fps (decision #3); never a literal
        if not fps or fps <= 0:
            raise RuntimeError(f"missing corrected_fps for {key}; run ingest first")

        pose = dlc_io.read_dlc_csv(dlc_path)
        n_frames = int(len(pose))
        n_shift = n_shift_for_fps(fps)  # round(fps/10), per-video
        n_clusters = p.n_clusters

        if p.engine == "stub":
            backend = "stub"
            streams = _run_stub(n_frames, n_shift, n_clusters)
        elif p.engine == "real":
            try:
                streams = _run_real(cfg, p, pose, fps)
                backend = "real"
            except ImportError as exc:
                backend = f"stub (fallback: B-SOiD import failed: {exc})"
                streams = _run_stub(n_frames, n_shift, n_clusters)
        else:
            raise ValueError(f"Unknown bsoid engine: {p.engine!r}")

        labels = reconstruct_labels(streams, n_frames)  # frameshift + flatten('F'); len == n_frames
        n_shift_used = len(streams)

        factors = parse_class(key.klass, cfg.project.class_parser)
        base = {"Type": key.type, "Class": key.klass, "Filename": key.filename, **factors}
        factor_cols = list(factors.keys())

        # per-frame labels (tidy long)
        labels_df = pd.DataFrame(
            {**{k: [v] * n_frames for k, v in base.items()}, "frame": np.arange(n_frames), "label": labels}
        )[["Type", "Class", "Filename", *factor_cols, "frame", "label"]]

        # per-cluster summary over range(n_clusters)
        counts = np.bincount(labels[labels >= 0], minlength=n_clusters)
        cluster_rows = [
            {
                **base,
                "cluster": c,
                "frame_count": int(counts[c]),
                "time_s": int(counts[c]) / fps,
                "fraction": (int(counts[c]) / n_frames) if n_frames else 0.0,
            }
            for c in range(n_clusters)
        ]
        clusters_df = pd.DataFrame(cluster_rows)[
            ["Type", "Class", "Filename", *factor_cols, "cluster", "frame_count", "time_s", "fraction"]
        ]

        lp, lc = bsoid_labels_parquet(cfg, key), bsoid_labels_csv(cfg, key)
        cp, cc = bsoid_clusters_parquet(cfg, key), bsoid_clusters_csv(cfg, key)
        lp.parent.mkdir(parents=True, exist_ok=True)
        labels_df.to_parquet(lp, index=False)  # for R (separate manual step); Python never calls R
        labels_df.to_csv(lc, index=False)
        clusters_df.to_parquet(cp, index=False)
        clusters_df.to_csv(cc, index=False)

        ctx.manifest.set_params(
            key,
            self.name,
            {
                "engine_requested": p.engine,
                "backend": backend,
                "n_clusters": n_clusters,
                "n_shift": n_shift_used,
                "corrected_fps": fps,
                "n_frames": n_frames,
                "factors": factors,
                "labels_parquet": str(lp),
                "labels_csv": str(lc),
                "clusters_parquet": str(cp),
                "clusters_csv": str(cc),
            },
        )
