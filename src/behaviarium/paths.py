"""Known output paths for stage artifacts. Pure pathlib — no I/O, no assay logic.

Layout mirrors the join key: ``<output_root>/<stage>/<Type>/<Class>/<stem>__<stage>.<ext>``.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .manifest import VideoKey


def stage_dir(cfg: Config, key: VideoKey, stage: str) -> Path:
    return cfg.output_root / stage / key.type / key.klass


def _stem(key: VideoKey) -> str:
    return Path(key.filename).stem


def video_output(cfg: Config, key: VideoKey, stage: str, ext: str = ".avi") -> Path:
    return stage_dir(cfg, key, stage) / f"{_stem(key)}__{stage}{ext}"


def boundary_preview(cfg: Config, key: VideoKey) -> Path:
    return stage_dir(cfg, key, "boundary") / f"{_stem(key)}__boundary_preview.png"


def dlc_raw_path(cfg: Config, key: VideoKey) -> Path:
    return stage_dir(cfg, key, "dlc") / f"{_stem(key)}__dlc_raw.csv"


def dlc_filtered_path(cfg: Config, key: VideoKey) -> Path:
    return stage_dir(cfg, key, "dlc") / f"{_stem(key)}__dlc_filtered.csv"


def dlc_output_path(cfg: Config, key: VideoKey, filtered: bool) -> Path:
    """The active DLC tracking output: _filtered when filtering is enabled, else _raw."""
    return dlc_filtered_path(cfg, key) if filtered else dlc_raw_path(cfg, key)


def chamber_parquet(cfg: Config, key: VideoKey) -> Path:
    return stage_dir(cfg, key, "chamber") / f"{_stem(key)}__chamber.parquet"


def chamber_csv(cfg: Config, key: VideoKey) -> Path:
    return stage_dir(cfg, key, "chamber") / f"{_stem(key)}__chamber.csv"


def bsoid_labels_parquet(cfg: Config, key: VideoKey) -> Path:
    return stage_dir(cfg, key, "bsoid") / f"{_stem(key)}__bsoid_labels.parquet"


def bsoid_labels_csv(cfg: Config, key: VideoKey) -> Path:
    return stage_dir(cfg, key, "bsoid") / f"{_stem(key)}__bsoid_labels.csv"


def bsoid_clusters_parquet(cfg: Config, key: VideoKey) -> Path:
    return stage_dir(cfg, key, "bsoid") / f"{_stem(key)}__bsoid_clusters.parquet"


def bsoid_clusters_csv(cfg: Config, key: VideoKey) -> Path:
    return stage_dir(cfg, key, "bsoid") / f"{_stem(key)}__bsoid_clusters.csv"


# --- project-level aggregates (Phase 5). One set per project; no video in the path. ---
def postprocess_bsoid_long(cfg: Config, ext: str) -> Path:
    return cfg.output_root / "postprocess" / f"bsoid_clusters_long{ext}"


def postprocess_chamber_long(cfg: Config, ext: str) -> Path:
    return cfg.output_root / "postprocess" / f"chamber_occupancy_long{ext}"


def stats_bsoid_table(cfg: Config, ext: str) -> Path:
    return cfg.output_root / "stats" / f"bsoid_cluster_stats{ext}"


def stats_chamber_table(cfg: Config, ext: str) -> Path:
    return cfg.output_root / "stats" / f"chamber_region_stats{ext}"


# --- export bundle (Phase 6). One portable, self-describing bundle per project. ---
def export_dir(cfg: Config) -> Path:
    return cfg.output_root / "export" / cfg.project.name
