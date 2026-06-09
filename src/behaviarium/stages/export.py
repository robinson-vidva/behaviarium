"""Export stage — assemble the pipeline's tidy outputs into one portable, self-describing
bundle. TERMINAL stage. PROJECT scope (one bundle per project, not per-video).

It does NOT recompute or transform data — it copies the canonical Parquet+CSV already written
by postprocess (aggregates) and stats, plus the per-video B-SOiD per-frame labels, into
``outputs/export/<project>/`` with stable names, and writes a machine-readable
``export_manifest.json`` + a minimal ``data_dictionary.md``. Downstream platforms (R, etc.)
consume the bundle as a separate manual step.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..paths import (
    bsoid_labels_csv,
    bsoid_labels_parquet,
    export_dir,
    postprocess_bsoid_long,
    postprocess_chamber_long,
    stats_bsoid_table,
    stats_chamber_table,
)
from ..registry import register
from ..runner import eligible_video_ids
from ..stage import Stage, StageContext, StageScope

# (bundle name, producing stage, source-path function). Each is one canonical file (parquet+csv).
_SINGLE_DATASETS = [
    ("chamber_long", "postprocess", postprocess_chamber_long),
    ("bsoid_clusters", "postprocess", postprocess_bsoid_long),
    ("cluster_stats", "stats", stats_bsoid_table),
    ("region_stats", "stats", stats_chamber_table),
]

_BASE_COLS = {"video_id", "filename"}

_COLUMN_DOCS = {
    "video_id": "Primary identity — stable filename slug (the join key across all datasets)",
    "filename": "Original source video filename",
    "cluster": "B-SOiD behavioral cluster index (0..n_clusters-1)",
    "region": "Arena region name (from the per-project region scheme)",
    "frame": "Frame index (0-based)",
    "label": "Per-frame B-SOiD cluster label",
    "frame_count": "Number of frames in this cluster/region",
    "time_s": "Time in seconds = frame_count / corrected_fps",
    "fraction": "Fraction of the video's frames in this cluster/region",
    "group_a": "First compared group (a level of the stats group_factor)",
    "group_b": "Second compared group",
    "n_a": "Number of videos in group A",
    "n_b": "Number of videos in group B",
    "wasserstein_stat": "Wasserstein-1 distance between the groups' per-video fraction distributions",
    "p_value": "Permutation p-value (label-shuffle null for the Wasserstein statistic)",
    "q_value": "Benjamini-Hochberg FDR-adjusted p-value across units",
    "significant": "q_value < alpha",
}
_FALLBACK_DOC = "Design factor column (from tagging) / value column"


def _require(path: Path, producer: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"export: missing {producer} output; run {producer} first ({path})")
    return path


def _schema(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }


@register("export")
class ExportStage(Stage):
    scope = StageScope.PROJECT

    def outputs(self, ctx: StageContext) -> list[Path]:
        bundle = export_dir(ctx.cfg)
        return [bundle / "export_manifest.json", bundle / "data_dictionary.md"]

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.cfg
        bundle = export_dir(cfg)
        bundle.mkdir(parents=True, exist_ok=True)

        datasets: dict = {}

        # 1) single-file aggregate + stats datasets (copy verbatim to stable names)
        for name, producer, src_fn in _SINGLE_DATASETS:
            src_pq = _require(src_fn(cfg, ".parquet"), producer)
            src_csv = _require(src_fn(cfg, ".csv"), producer)
            shutil.copyfile(src_pq, bundle / f"{name}.parquet")
            shutil.copyfile(src_csv, bundle / f"{name}.csv")
            entry = {"producer": producer, "parquet": f"{name}.parquet", "csv": f"{name}.csv"}
            entry.update(_schema(pd.read_parquet(bundle / f"{name}.parquet")))
            datasets[name] = entry

        # 2) per-video B-SOiD per-frame labels (copied, never concatenated/transformed)
        labels_dir = bundle / "bsoid_labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        video_ids = eligible_video_ids(cfg, ctx.manifest, self.name)
        parts, total_rows, schema = [], 0, None
        for vid in video_ids:
            src_pq = _require(bsoid_labels_parquet(cfg, vid), "bsoid")
            src_csv = _require(bsoid_labels_csv(cfg, vid), "bsoid")
            base = f"{vid}__bsoid_labels"
            shutil.copyfile(src_pq, labels_dir / f"{base}.parquet")
            shutil.copyfile(src_csv, labels_dir / f"{base}.csv")
            df = pd.read_parquet(labels_dir / f"{base}.parquet")
            schema = _schema(df) if schema is None else schema
            total_rows += int(len(df))
            parts.append(
                {
                    "video_id": vid,
                    "parquet": f"bsoid_labels/{base}.parquet",
                    "csv": f"bsoid_labels/{base}.csv",
                    "rows": int(len(df)),
                }
            )
        datasets["bsoid_labels"] = {
            "producer": "bsoid",
            "kind": "per_video",
            "columns": schema["columns"] if schema else [],
            "dtypes": schema["dtypes"] if schema else {},
            "rows": total_rows,
            "parts": parts,
        }

        # factor columns = design-factor tags carried into the chamber aggregate
        chamber_cols = datasets["chamber_long"]["columns"]
        known = _BASE_COLS | {"region", "frame_count", "time_s", "fraction"}
        factors = [c for c in chamber_cols if c not in known]
        n_clusters = cfg.project.bsoid.n_clusters if cfg.project.bsoid else cfg.n_clusters

        export_manifest = {
            "project": cfg.project.name,
            "assay": cfg.assay,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "join_key": ["video_id"],
            "corrected_fps": {
                "formula": "frame_count / recording_duration_s",
                "recording_duration_s": cfg.recording_duration_s,
            },
            "design_factors": [{"name": f.name, "levels": f.levels} for f in cfg.project.design.factors],
            "factors": factors,
            "n_clusters": n_clusters,
            "datasets": datasets,
        }
        (bundle / "export_manifest.json").write_text(json.dumps(export_manifest, indent=2))
        (bundle / "data_dictionary.md").write_text(_data_dictionary(export_manifest, cfg))

        ctx.manifest.set_params(
            ctx.video,
            self.name,
            {
                "bundle_dir": str(bundle),
                "datasets": {k: (v.get("rows")) for k, v in datasets.items()},
                "n_videos": len(video_ids),
                "manifest": str(bundle / "export_manifest.json"),
            },
        )


def _doc(col: str) -> str:
    return _COLUMN_DOCS.get(col, _FALLBACK_DOC)


def _data_dictionary(m: dict, cfg) -> str:
    lines = [
        f"# {m['project']} ({m['assay']}) — data dictionary",
        "",
        "Join key: **video_id** — present in every dataset; join on it.",
        f"corrected_fps = frame_count / recording_duration_s "
        f"(recording_duration_s = {m['corrected_fps']['recording_duration_s']}).",
        f"Design-factor columns (from tagging): {', '.join(m['factors']) or '(none)'}.  "
        f"n_clusters = {m['n_clusters']}.",
        "",
    ]
    for name, d in m["datasets"].items():
        loc = d.get("parquet") or "bsoid_labels/ (one file per video)"
        lines.append(f"## {name}  ·  producer: {d['producer']}  ·  {loc}")
        for col in d["columns"]:
            lines.append(f"- `{col}` — {_doc(col)} ({d['dtypes'].get(col, '')})")
        lines.append("")
    lines += [
        "## How to load",
        "- Python: `pandas.read_parquet(\"chamber_long.parquet\")` (or `read_csv`).",
        "- R: `arrow::read_parquet(\"chamber_long.parquet\")` or `readr::read_csv(\"chamber_long.csv\")`.",
        "- Join any datasets on `video_id`.",
        "",
    ]
    return "\n".join(lines)
