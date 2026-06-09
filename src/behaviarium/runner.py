"""Runner — execute one stage and drive its manifest status.

VIDEO scope: skip-if-done (idempotent); excluded or untagged videos are SKIPPED.
DATASET scope (ingest): run, manages its own rows.
PROJECT scope (postprocess/stats/export): one aggregate output, status under ``PROJECT_ID``.
"""

from __future__ import annotations

import traceback

from .config import Config
from .manifest import PROJECT_ID, Manifest, Status
from .pipeline import TAG_REQUIRED_STAGES
from .registry import get_stage
from .stage import StageContext, StageScope


def is_included(manifest: Manifest, video_id: str) -> bool:
    rec = manifest.get_video(video_id)
    return rec is not None and bool(rec.get("include", 1))


def is_fully_tagged(tag: dict | None, cfg: Config) -> bool:
    """A video is tagged when it has a level for every declared design factor (trivially true
    when the project declares no factors)."""
    names = cfg.project.design.factor_names()
    if not names:
        return True
    return bool(tag) and all(n in tag and tag[n] not in (None, "") for n in names)


def is_processable(cfg: Config, manifest: Manifest, video_id: str) -> bool:
    """Eligible for the tag-required grouping stages: included AND fully tagged. (Per-video
    prep/analysis stages run on any included video — tag optional.)"""
    rec = manifest.get_video(video_id)
    if rec is None or not rec.get("include", 1):
        return False
    return is_fully_tagged(rec.get("tag"), cfg)


def eligible_video_ids(cfg: Config, manifest: Manifest, stage_name: str) -> list[str]:
    """Videos a given stage will process: always included; tagged too iff the stage is in
    TAG_REQUIRED_STAGES. One rule, driven by the single TAG_REQUIRED_STAGES list."""
    require_tag = stage_name in TAG_REQUIRED_STAGES
    out = []
    for v in manifest.list_videos():
        if not v.get("include", 1):
            continue
        if require_tag and not is_fully_tagged(v.get("tag"), cfg):
            continue
        out.append(v["video_id"])
    return out


# Back-compat alias: the grouping stages aggregate exactly the processable (include+tagged) set.
def processable_video_ids(cfg: Config, manifest: Manifest) -> list[str]:
    return [v["video_id"] for v in manifest.list_videos() if is_processable(cfg, manifest, v["video_id"])]


def _run_tracked(stage, ctx, manifest, name, video_id) -> Status:
    if stage.is_done(ctx) and manifest.get_status(video_id, name) == Status.DONE.value:
        return Status.DONE
    manifest.set_status(video_id, name, Status.RUNNING)
    try:
        stage.run(ctx)
    except Exception as exc:  # noqa: BLE001 — record failure, surface to caller
        manifest.set_status(video_id, name, Status.FAILED, error="".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip())
        raise
    manifest.set_status(video_id, name, Status.DONE)
    return Status.DONE


def run_stage(name: str, cfg: Config, manifest: Manifest, video: str | None = None) -> Status:
    stage = get_stage(name, cfg.assay)()

    if stage.scope == StageScope.DATASET:
        stage.run(StageContext(cfg=cfg, manifest=manifest, video=video))
        return Status.DONE

    if stage.scope == StageScope.PROJECT:
        ctx = StageContext(cfg=cfg, manifest=manifest, video=PROJECT_ID)
        return _run_tracked(stage, ctx, manifest, name, PROJECT_ID)

    if video is None:
        raise ValueError(f"Stage {name!r} is video-scoped but no video_id was given")

    # Per-video stages gate on inclusion ONLY — the design tag is required just for the
    # TAG_REQUIRED_STAGES (grouping), which aggregate eligible_video_ids(). So an untagged
    # video still runs rotate..bsoid and joins the aggregates once tagged.
    if not is_included(manifest, video):
        return Status.SKIPPED  # excluded — runner skips it, status untouched

    ctx = StageContext(cfg=cfg, manifest=manifest, video=video)
    return _run_tracked(stage, ctx, manifest, name, video)
