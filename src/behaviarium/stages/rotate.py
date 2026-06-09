"""Rotate stage — config-driven orientation correction (Bonsai replacement). Assay-agnostic.

Degrees and optional flip come from per-project config. Reads the ingested source video and
writes an orientation-corrected video to a known outputs path. Idempotent.
"""

from __future__ import annotations

from pathlib import Path

from ..paths import video_output
from ..registry import register
from ..stage import Stage, StageContext
from ..video import apply_rotation, process_video


@register("rotate")
class RotateStage(Stage):
    def inputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.source_path()]

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [video_output(ctx.cfg, ctx.video, self.name)]

    def run(self, ctx: StageContext) -> None:
        p = ctx.cfg.project.rotate
        # Hook for future auto-detection of orientation; not implemented this phase.
        if p.auto:
            raise NotImplementedError("rotate.auto is a future hook; set explicit degrees/flip")
        src = ctx.source_path()
        dst = self.outputs(ctx)[0]
        process_video(src, dst, lambda f: apply_rotation(f, p.degrees, p.flip))
        ctx.manifest.set_params(
            ctx.video, self.name, {"degrees": p.degrees, "flip": p.flip, "output": str(dst)}
        )
