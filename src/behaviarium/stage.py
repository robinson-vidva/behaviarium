"""Stage contract: a uniform, idempotent unit of work.

Every stage declares its outputs (and optionally inputs) as known paths, runs, and reports
status through the manifest. Stages are assay-agnostic by default; assay-specific behaviour is
provided by registering a stage variant for a given assay (see ``behaviarium.registry``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar

from .config import Config
from .manifest import Manifest, VideoKey


class StageScope(str, Enum):
    VIDEO = "video"  # runs once per video
    DATASET = "dataset"  # runs once over the whole dataset, manages its own rows (e.g. ingest)
    PROJECT = "project"  # runs once across videos, ONE aggregate output (postprocess/stats)


@dataclass
class StageContext:
    """Everything a stage needs: config, manifest, and (for video-scoped stages) the video."""

    cfg: Config
    manifest: Manifest
    video: VideoKey | None = None

    @property
    def data_root(self) -> Path:
        return self.cfg.data_root

    @property
    def output_root(self) -> Path:
        return self.cfg.output_root

    def video_record(self) -> dict | None:
        return self.manifest.get_video(self.video) if self.video else None

    def source_path(self) -> Path:
        """Absolute path to the original source video (recorded by ingest)."""
        rec = self.video_record()
        if not rec or not rec.get("path"):
            raise RuntimeError(f"No ingested source path for {self.video}; run ingest first")
        return Path(rec["path"])


class Stage(ABC):
    """Abstract stage. Subclasses set ``name`` (via the registry) and implement ``run``."""

    name: ClassVar[str]
    assay: ClassVar[str | None] = None
    scope: ClassVar[StageScope] = StageScope.VIDEO

    def inputs(self, ctx: StageContext) -> list[Path]:
        """Declared input paths. Default: none."""
        return []

    @abstractmethod
    def outputs(self, ctx: StageContext) -> list[Path]:
        """Declared output paths. Used for the default idempotent skip-if-done check."""

    @abstractmethod
    def run(self, ctx: StageContext) -> None:
        """Do the work. Must be idempotent: re-running produces the same outputs."""

    def is_done(self, ctx: StageContext) -> bool:
        """Idempotent skip check. Default: all declared outputs already exist."""
        outs = self.outputs(ctx)
        return bool(outs) and all(p.exists() for p in outs)
