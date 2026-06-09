"""Command-line entry point. Used directly and by the Streamlit shell's background subprocesses."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import stages  # noqa: F401  ensures stages are registered
from .config import load_config
from .manifest import Manifest, VideoKey
from .paths import boundary_preview, video_output
from .runner import run_stage
from .stages.boundary import write_preview
from .video import read_frame


def _load(args: argparse.Namespace) -> tuple:
    cfg = load_config(args.project, Path(args.config_dir) if args.config_dir else None)
    manifest = Manifest(cfg.manifest_path)
    manifest.init()
    return cfg, manifest


def _key(args: argparse.Namespace) -> VideoKey:
    return VideoKey(type=args.type, klass=args.klass, filename=args.filename)


def _cmd_ingest(args: argparse.Namespace) -> int:
    cfg, manifest = _load(args)
    run_stage("ingest", cfg, manifest)
    print(f"ingest: {len(manifest.list_videos())} video(s) registered in {cfg.manifest_path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cfg, manifest = _load(args)
    key = _key(args)
    result = run_stage(args.stage, cfg, manifest, key)
    print(f"{args.stage} {key.type}/{key.klass}/{key.filename} -> {result.value}")
    return 0


def _cmd_run_all(args: argparse.Namespace) -> int:
    cfg, manifest = _load(args)
    for v in manifest.list_videos():
        key = VideoKey(v["type"], v["class"], v["filename"])
        result = run_stage(args.stage, cfg, manifest, key)
        print(f"{args.stage} {key.type}/{key.klass}/{key.filename} -> {result.value}")
    return 0


def _cmd_run_project(args: argparse.Namespace) -> int:
    cfg, manifest = _load(args)
    result = run_stage(args.stage, cfg, manifest)  # project-scoped: no video
    print(f"{args.stage} ({cfg.project.name}) -> {result.value}")
    return 0


def _cmd_render_preview(args: argparse.Namespace) -> int:
    """Redraw the boundary overlay from the stored geometry (e.g. after a manual override)."""
    cfg, manifest = _load(args)
    key = _key(args)
    row = manifest.get_row(key, "boundary")
    geom = (row.get("params") or {}).get("roi") if row else None
    if not geom:
        print("render-preview: no stored ROI geometry for this video")
        return 1
    frame = read_frame(video_output(cfg, key, "rotate"), cfg.project.boundary.sample_frame)
    write_preview(frame, geom, boundary_preview(cfg, key))
    print(f"preview re-rendered: {boundary_preview(cfg, key)}")
    return 0


def _add_video_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--type", required=True)
    p.add_argument("--class", dest="klass", required=True)
    p.add_argument("--filename", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="behaviarium")
    parser.add_argument("--config-dir", default=None, help="override config directory")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="discover videos and seed the manifest")
    p_ingest.add_argument("--project", required=True)
    p_ingest.set_defaults(func=_cmd_ingest)

    p_run = sub.add_parser("run", help="run one stage for one video")
    p_run.add_argument("--project", required=True)
    p_run.add_argument("--stage", required=True)
    _add_video_args(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_all = sub.add_parser("run-all", help="run one stage for every video")
    p_all.add_argument("--project", required=True)
    p_all.add_argument("--stage", required=True)
    p_all.set_defaults(func=_cmd_run_all)

    p_proj = sub.add_parser("run-project", help="run one project-scoped stage (postprocess/stats)")
    p_proj.add_argument("--project", required=True)
    p_proj.add_argument("--stage", required=True)
    p_proj.set_defaults(func=_cmd_run_project)

    p_prev = sub.add_parser("render-preview", help="redraw boundary overlay from stored geometry")
    p_prev.add_argument("--project", required=True)
    _add_video_args(p_prev)
    p_prev.set_defaults(func=_cmd_render_preview)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
