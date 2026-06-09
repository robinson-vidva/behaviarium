"""Runner — execute one stage and drive its manifest status.

For video-scoped stages: skip-if-done (idempotent), else mark running -> done/failed.
For dataset-scoped stages (e.g. ingest): run and let the stage write its own rows.
For project-scoped stages (postprocess/stats): one aggregate output, status tracked under a
synthetic project-level key.
"""

from __future__ import annotations

import traceback

from .config import Config
from .manifest import Manifest, Status, VideoKey
from .registry import get_stage
from .stage import StageContext, StageScope


def project_key(cfg: Config) -> VideoKey:
    """Synthetic manifest key for project-level (aggregate) stage rows."""
    return VideoKey(type="__project__", klass="__all__", filename=cfg.project.name)


def _run_tracked(stage, ctx, manifest, name, key) -> Status:
    if stage.is_done(ctx) and manifest.get_status(key, name) == Status.DONE.value:
        return Status.DONE
    manifest.set_status(key, name, Status.RUNNING)
    try:
        stage.run(ctx)
    except Exception as exc:  # noqa: BLE001 — record failure, surface to caller
        manifest.set_status(key, name, Status.FAILED, error="".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip())
        raise
    manifest.set_status(key, name, Status.DONE)
    return Status.DONE


def run_stage(
    name: str, cfg: Config, manifest: Manifest, video: VideoKey | None = None
) -> Status:
    stage = get_stage(name, cfg.assay)()

    if stage.scope == StageScope.DATASET:
        # Discovery-style stage: it manages its own per-video rows.
        stage.run(StageContext(cfg=cfg, manifest=manifest, video=video))
        return Status.DONE

    if stage.scope == StageScope.PROJECT:
        key = project_key(cfg)
        ctx = StageContext(cfg=cfg, manifest=manifest, video=key)
        return _run_tracked(stage, ctx, manifest, name, key)

    if video is None:
        raise ValueError(f"Stage {name!r} is video-scoped but no video was given")

    ctx = StageContext(cfg=cfg, manifest=manifest, video=video)

    rec = manifest.get_video(video)
    if rec is not None and not rec.get("include", 1):
        return Status.SKIPPED  # excluded video — runner skips it, status untouched

    return _run_tracked(stage, ctx, manifest, name, video)
