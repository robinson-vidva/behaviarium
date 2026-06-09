"""Mask stage — apply the APPROVED arena ROI to make the DLC-ready video. Assay-agnostic.

Refuses to run unless the (video, boundary) row is ``approved``. Zeros (or fills) pixels
outside the ROI and optionally crops to the ROI bounding box. Idempotent.
"""

from __future__ import annotations

from pathlib import Path

from ..manifest import Approval
from ..paths import video_output
from ..registry import register
from ..roi import apply_mask
from ..stage import Stage, StageContext
from ..video import process_video


@register("mask")
class MaskStage(Stage):
    def inputs(self, ctx: StageContext) -> list[Path]:
        return [video_output(ctx.cfg, ctx.video, "rotate")]

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [video_output(ctx.cfg, ctx.video, self.name)]

    def run(self, ctx: StageContext) -> None:
        row = ctx.manifest.get_row(ctx.video, "boundary")
        approval = row.get("approval") if row else None
        if approval != Approval.APPROVED.value:
            raise RuntimeError(
                f"mask requires an APPROVED boundary ROI (approval={approval!r}); "
                "approve it in the control plane first"
            )
        geom = (row.get("params") or {}).get("roi")
        if not geom:
            raise RuntimeError("boundary row has no ROI geometry")

        mp = ctx.cfg.project.mask
        src = video_output(ctx.cfg, ctx.video, "rotate")
        if not src.exists():
            raise RuntimeError(f"rotate output missing; run rotate first: {src}")
        dst = self.outputs(ctx)[0]
        process_video(
            src, dst, lambda f: apply_mask(f, geom, fill_value=mp.fill_value, crop=mp.crop)
        )
        ctx.manifest.set_params(
            ctx.video, self.name, {"roi": geom, "crop": mp.crop, "output": str(dst)}
        )
