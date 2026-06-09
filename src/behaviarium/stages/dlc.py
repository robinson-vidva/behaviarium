"""DLC stage — engine-aware pose estimation. One interface, two backends.

Backends:
- ``tensorflow`` (real): lazily imports DeepLabCut and runs ``analyze_videos`` on the mask-stage
  output. Imported ONLY when used, so the base package installs on Mac (no TF/DLC).
- ``stub`` (synthetic): writes a DLC-schema CSV with the configured bodyparts and plausible
  coords — no TF. First-class so Mac dev never blocks.

If ``engine=tensorflow`` but the DeepLabCut import fails (e.g. on Mac), it falls back to the
stub. Real DLC runtime errors are NOT swallowed — only import failure triggers fallback.

Decision #1: filtering is honest. Disabled (default) => ``_raw``, no filter. Enabled => actually
filter (pandas median; arima optional) => ``_filtered``. A ``_filtered`` file is never written
unless filtering actually ran.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .. import dlc_io
from ..config import resolve_dlc_model_config
from ..paths import dlc_filtered_path, dlc_output_path, dlc_raw_path, stage_dir, video_output
from ..registry import register
from ..stage import Stage, StageContext
from ..video import probe_dims, probe_frame_count


def _synth_coords(n_frames: int, n_bodyparts: int, dims: tuple[int, int]) -> np.ndarray:
    """Plausible synthetic tracks: bounded random walks with high likelihood. Deterministic."""
    w, h = dims
    rng = np.random.default_rng(0)
    data = np.zeros((max(n_frames, 1), n_bodyparts * 3), dtype=float)
    for b in range(n_bodyparts):
        x = np.clip(np.cumsum(rng.normal(0, 1.5, data.shape[0])) + rng.uniform(0.2 * w, 0.8 * w), 0, max(w - 1, 0))
        y = np.clip(np.cumsum(rng.normal(0, 1.5, data.shape[0])) + rng.uniform(0.2 * h, 0.8 * h), 0, max(h - 1, 0))
        data[:, b * 3 + 0] = x
        data[:, b * 3 + 1] = y
        data[:, b * 3 + 2] = rng.uniform(0.9, 1.0, data.shape[0])
    return data


def _run_stub(p, raw_path: Path, n_frames: int, dims: tuple[int, int], model_label: str) -> str:
    if not p.bodyparts:
        raise RuntimeError("stub DLC backend requires dlc.bodyparts in the project config")
    scorer = f"DLC_stub_{model_label}_sh{p.shuffle}"
    coords = _synth_coords(n_frames, len(p.bodyparts), dims)
    df = dlc_io.build_dlc_dataframe(scorer, list(p.bodyparts), coords)
    dlc_io.write_dlc_csv(df, raw_path)
    return scorer


def _find_produced_csv(dest_dir: Path, vstem: str) -> Path:
    """Locate the CSV DLC just wrote (``<vstem><DLCscorer>.csv``), excluding our own names."""
    cands = [
        p
        for p in sorted(dest_dir.glob(f"{vstem}*.csv"))
        if not p.name.endswith(("__dlc_raw.csv", "__dlc_filtered.csv"))
    ]
    if not cands:
        raise RuntimeError(f"DLC produced no CSV in {dest_dir} for {vstem!r}")
    return cands[0]


def _run_tensorflow(cfg, p, video: Path, dest_dir: Path, raw_path: Path) -> str:
    import deeplabcut  # lazy: only present in the Windows DLC env

    model_config = resolve_dlc_model_config(cfg)
    if model_config is None or not model_config.exists():
        raise RuntimeError(
            f"DLC model config not found: {model_config} "
            "(set BEHAVIARIUM_DLC_MODEL_CONFIG or dlc.model_config_path)"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Pass an explicit file path so the renamed-across-versions ``videotype`` arg isn't needed.
    deeplabcut.analyze_videos(
        str(model_config),
        [str(video)],
        shuffle=p.shuffle,
        trainingsetindex=p.trainingsetindex,
        save_as_csv=True,
        destfolder=str(dest_dir),
    )
    shutil.copyfile(_find_produced_csv(dest_dir, video.stem), raw_path)
    return "tensorflow"


def _dlc_filterpredictions(cfg, p, video: Path, dest_dir: Path, filtered_path: Path) -> None:
    """DLC-exact filtering. Reads the analyze_videos .h5 left in ``dest_dir`` and writes
    ``<stem><scorer>_filtered.csv`` there; we relocate it to our known _filtered path."""
    import deeplabcut  # lazy

    model_config = resolve_dlc_model_config(cfg)
    if model_config is None or not model_config.exists():
        raise RuntimeError(
            f"DLC model config not found: {model_config} "
            "(set BEHAVIARIUM_DLC_MODEL_CONFIG or dlc.model_config_path)"
        )
    deeplabcut.filterpredictions(
        str(model_config),
        [str(video)],
        shuffle=p.shuffle,
        trainingsetindex=p.trainingsetindex,
        filtertype=p.filter.type,
        windowlength=p.filter.windowlength,
        save_as_csv=True,
        destfolder=str(dest_dir),
    )
    produced = [
        f
        for f in sorted(dest_dir.glob(f"{video.stem}*_filtered.csv"))
        if not f.name.endswith("__dlc_filtered.csv")
    ]
    if not produced:
        raise RuntimeError(f"filterpredictions produced no _filtered.csv in {dest_dir}")
    shutil.copyfile(produced[0], filtered_path)


@register("dlc")
class DlcStage(Stage):
    def inputs(self, ctx: StageContext) -> list[Path]:
        return [video_output(ctx.cfg, ctx.video, "mask")]

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [dlc_output_path(ctx.cfg, ctx.video, ctx.cfg.project.dlc.filter.enabled)]

    def run(self, ctx: StageContext) -> None:
        p = ctx.cfg.project.dlc
        mask_video = video_output(ctx.cfg, ctx.video, "mask")
        if not mask_video.exists():
            raise RuntimeError(f"dlc requires the mask output; run mask first: {mask_video}")

        n_frames = probe_frame_count(mask_video)
        dims = probe_dims(mask_video)
        dest_dir = stage_dir(ctx.cfg, ctx.video, self.name)
        raw_path = dlc_raw_path(ctx.cfg, ctx.video)

        # --- backend selection (engine-aware, one interface) ---
        if p.engine == "stub":
            backend = "stub"
            model = "stub"
            scorer = _run_stub(p, raw_path, n_frames, dims, model_label="stub")
        elif p.engine == "tensorflow":
            try:
                backend = _run_tensorflow(ctx.cfg, p, mask_video, dest_dir, raw_path)
                model = str(resolve_dlc_model_config(ctx.cfg))
                scorer = dlc_io.scorer_name(dlc_io.read_dlc_csv(raw_path))
            except ImportError as exc:
                backend = f"stub (fallback: deeplabcut import failed: {exc})"
                model = "stub"
                scorer = _run_stub(p, raw_path, n_frames, dims, model_label="tf_fallback")
        else:
            raise ValueError(f"Unknown dlc engine: {p.engine!r}")

        # --- decision #1: honest, config-driven filtering ---
        filtered_path = dlc_filtered_path(ctx.cfg, ctx.video)
        filter_path = None
        if p.filter.enabled:
            if p.filter.delegate_to_dlc:
                # DLC-exact: only meaningful when the REAL tensorflow backend ran. Never
                # silently fall back to pandas — that would defeat the purpose.
                if backend != "tensorflow":
                    raise RuntimeError(
                        "dlc.filter.delegate_to_dlc=true requires the tensorflow backend, but "
                        f"backend={backend!r}. DLC-exact filtering needs real DeepLabCut; use "
                        "filter.delegate_to_dlc=false (pandas median) for the stub/fallback."
                    )
                _dlc_filterpredictions(ctx.cfg, p, mask_video, dest_dir, filtered_path)
                filter_path = "dlc-filterpredictions"
            else:
                df = dlc_io.read_dlc_csv(raw_path)
                if p.filter.type == "median":
                    df = dlc_io.median_filter(df, p.filter.windowlength)
                elif p.filter.type == "arima":
                    raise NotImplementedError(
                        "arima filtering is not implemented in the pandas path; use type=median, "
                        "or set filter.delegate_to_dlc=true to use DLC filterpredictions (Windows)"
                    )
                else:
                    raise ValueError(f"Unknown dlc.filter.type: {p.filter.type!r}")
                dlc_io.write_dlc_csv(df, filtered_path)
                filter_path = "pandas-median"
            out_path, filtered = filtered_path, True
        else:
            # never leave a stale/misleading _filtered when filtering is off
            filtered_path.unlink(missing_ok=True)
            out_path, filtered = raw_path, False

        bodyparts = dlc_io.list_bodyparts(dlc_io.read_dlc_csv(out_path))  # round-trip proof
        ctx.manifest.set_params(
            ctx.video,
            self.name,
            {
                "engine_requested": p.engine,
                "backend": backend,
                "model": model,
                "scorer": scorer,
                "shuffle": p.shuffle,
                "trainingsetindex": p.trainingsetindex,
                "bodyparts": bodyparts,
                "filtered": filtered,
                "filter_type": p.filter.type if filtered else None,
                "filter_path": filter_path,  # pandas-median | dlc-filterpredictions | None
                "raw": str(raw_path),
                "output": str(out_path),
            },
        )
