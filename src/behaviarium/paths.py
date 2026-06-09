"""Known output paths (Phase 7). Per-video artifacts live in the project's per-video folder
``<project_dir>/videos/<video_id>/``; project-level aggregates live in ``<project_dir>/outputs/``.
Pure pathlib — no I/O, no assay logic.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config


def video_dir(cfg: Config, video_id: str) -> Path:
    """Per-video home for sidecars + that video's stage outputs/metadata."""
    return cfg.videos_dir / video_id


def video_output(cfg: Config, video_id: str, stage: str, ext: str = ".avi") -> Path:
    return video_dir(cfg, video_id) / f"{video_id}__{stage}{ext}"


def boundary_preview(cfg: Config, video_id: str) -> Path:
    return video_dir(cfg, video_id) / f"{video_id}__boundary_preview.png"


def dlc_raw_path(cfg: Config, video_id: str) -> Path:
    return video_dir(cfg, video_id) / f"{video_id}__dlc_raw.csv"


def dlc_filtered_path(cfg: Config, video_id: str) -> Path:
    return video_dir(cfg, video_id) / f"{video_id}__dlc_filtered.csv"


def dlc_output_path(cfg: Config, video_id: str, filtered: bool) -> Path:
    return dlc_filtered_path(cfg, video_id) if filtered else dlc_raw_path(cfg, video_id)


def chamber_parquet(cfg: Config, video_id: str) -> Path:
    return video_dir(cfg, video_id) / f"{video_id}__chamber.parquet"


def chamber_csv(cfg: Config, video_id: str) -> Path:
    return video_dir(cfg, video_id) / f"{video_id}__chamber.csv"


def bsoid_labels_parquet(cfg: Config, video_id: str) -> Path:
    return video_dir(cfg, video_id) / f"{video_id}__bsoid_labels.parquet"


def bsoid_labels_csv(cfg: Config, video_id: str) -> Path:
    return video_dir(cfg, video_id) / f"{video_id}__bsoid_labels.csv"


def bsoid_clusters_parquet(cfg: Config, video_id: str) -> Path:
    return video_dir(cfg, video_id) / f"{video_id}__bsoid_clusters.parquet"


def bsoid_clusters_csv(cfg: Config, video_id: str) -> Path:
    return video_dir(cfg, video_id) / f"{video_id}__bsoid_clusters.csv"


# --- project-level aggregates ---
def postprocess_bsoid_long(cfg: Config, ext: str) -> Path:
    return cfg.outputs_dir / "postprocess" / f"bsoid_clusters_long{ext}"


def postprocess_chamber_long(cfg: Config, ext: str) -> Path:
    return cfg.outputs_dir / "postprocess" / f"chamber_occupancy_long{ext}"


def stats_bsoid_table(cfg: Config, ext: str) -> Path:
    return cfg.outputs_dir / "stats" / f"bsoid_cluster_stats{ext}"


def stats_chamber_table(cfg: Config, ext: str) -> Path:
    return cfg.outputs_dir / "stats" / f"chamber_region_stats{ext}"


def export_dir(cfg: Config) -> Path:
    return cfg.outputs_dir / "export"
