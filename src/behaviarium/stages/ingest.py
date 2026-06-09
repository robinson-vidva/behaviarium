"""Ingest stage — layout-agnostic discovery + stable identity (Phase 7). Assay-agnostic.

Scans ``data_path`` RECURSIVELY for videos regardless of flat/nested layout. Each video gets
a stable ``video_id`` (filename slug, deduped on collision) plus its original ``source_path``.
Re-ingest is idempotent: a file already known by source_path keeps its video_id. A per-video
folder ``videos/<video_id>/`` is created for every video. NO folder-derived Type/Class, NO
Class-string parsing — design factors come from tagging.
"""

from __future__ import annotations

from pathlib import Path

from ..identity import make_video_id
from ..manifest import Status
from ..paths import video_dir
from ..registry import register
from ..stage import Stage, StageContext, StageScope
from ..video import probe_frame_count


@register("ingest")
class IngestStage(Stage):
    scope = StageScope.DATASET

    def outputs(self, ctx: StageContext) -> list[Path]:
        return []  # outputs are manifest rows + per-video folders

    def is_done(self, ctx: StageContext) -> bool:
        return False  # always re-scan; upserts make this idempotent

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.cfg
        root = cfg.data_path
        exts = {e.lower() for e in cfg.video_extensions}
        duration = cfg.recording_duration_s

        existing = {v["video_id"] for v in ctx.manifest.list_videos()}
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path.suffix.lower() not in exts:
                continue
            source_path = path.resolve()
            known = ctx.manifest.get_video_by_source_path(source_path)
            if known:
                video_id = known["video_id"]
                current_path = known["current_path"]  # preserve a prior reorg
            else:
                video_id = make_video_id(path.name, existing)
                existing.add(video_id)
                current_path = source_path

            frame_count = probe_frame_count(current_path)
            fps = (frame_count / duration) if duration else None
            video_dir(cfg, video_id).mkdir(parents=True, exist_ok=True)
            ctx.manifest.upsert_video(
                video_id, path.name, source_path, current_path, frame_count, fps
            )
            ctx.manifest.upsert(
                video_id,
                self.name,
                status=Status.DONE,
                params={"frame_count": frame_count, "fps": fps, "source_path": str(source_path)},
            )
