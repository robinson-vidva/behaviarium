"""Ingest stage — discover videos and seed the manifest. Assay-agnostic.

Discovers video files under ``data_root`` using the generic layout
``data_root/<Type>/<Class>/<filename>``, probes the actual frame count via OpenCV, and stores
the corrected fps (= frame_count / recording_duration_s) per video — the single source of
truth for fps. Interpreting the ``Class`` string is a later, assay-specific stage; ingest only
records the raw join key.
"""

from __future__ import annotations

from pathlib import Path

from ..manifest import Status, VideoKey
from ..registry import register
from ..stage import Stage, StageContext, StageScope
from ..video import probe_frame_count


def video_key_from_path(path: Path, data_root: Path) -> VideoKey:
    """Derive (Type, Class, Filename) from the path relative to data_root.

    Layout convention: ``data_root/<Type>/<Class>/<filename>``. Shallower trees degrade
    gracefully (missing levels become "").
    """
    parts = path.relative_to(data_root).parts
    filename = parts[-1]
    parents = parts[:-1]
    type_ = parents[0] if len(parents) >= 1 else ""
    klass = parents[1] if len(parents) >= 2 else ""
    return VideoKey(type=type_, klass=klass, filename=filename)


@register("ingest")
class IngestStage(Stage):
    scope = StageScope.DATASET

    def outputs(self, ctx: StageContext) -> list[Path]:
        return []  # outputs are manifest rows, not files

    def is_done(self, ctx: StageContext) -> bool:
        return False  # always re-scan; upserts make this idempotent

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.cfg
        root = cfg.data_root
        exts = {e.lower() for e in cfg.video_extensions}
        duration = cfg.recording_duration_s

        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path.suffix.lower() not in exts:
                continue
            key = video_key_from_path(path, root)
            frame_count = probe_frame_count(path)
            fps = (frame_count / duration) if duration else None
            ctx.manifest.upsert_video(key, path, frame_count, fps)
            ctx.manifest.upsert(
                key,
                self.name,
                status=Status.DONE,
                params={"frame_count": frame_count, "fps": fps, "path": str(path)},
            )
