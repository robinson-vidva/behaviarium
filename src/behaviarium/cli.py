"""Command-line entry point. Used directly and by the Streamlit shell's background subprocesses.

Phase 7: ``--project`` is an external PROJECT DIRECTORY (not a name); videos are addressed by
``--video-id``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import stages  # noqa: F401  ensures stages are registered
from .config import init_project, load_project
from .manifest import Manifest
from .paths import boundary_preview, video_output
from .reorg import MODES, reorg_video
from .runner import run_stage
from .stages.boundary import write_preview
from .video import read_frame


def _open(args: argparse.Namespace) -> tuple:
    cfg = load_project(args.project)
    manifest = Manifest(cfg.manifest_path)
    manifest.init()
    return cfg, manifest


def _cmd_init_project(args: argparse.Namespace) -> int:
    cfg = init_project(args.project, args.data, template=args.template)
    print(f"initialized project '{cfg.project.name}' ({cfg.assay}) at {cfg.project_dir}")
    print(f"  data_path: {cfg.data_path}")
    print(f"  manifest:  {cfg.manifest_path}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    cfg, manifest = _open(args)
    run_stage("ingest", cfg, manifest)
    print(f"ingest: {len(manifest.list_videos())} video(s) discovered under {cfg.data_path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cfg, manifest = _open(args)
    result = run_stage(args.stage, cfg, manifest, args.video_id)
    print(f"{args.stage} {args.video_id} -> {result.value}")
    return 0


def _cmd_run_all(args: argparse.Namespace) -> int:
    cfg, manifest = _open(args)
    for v in manifest.list_videos():
        result = run_stage(args.stage, cfg, manifest, v["video_id"])
        print(f"{args.stage} {v['video_id']} -> {result.value}")
    return 0


def _cmd_run_project(args: argparse.Namespace) -> int:
    cfg, manifest = _open(args)
    result = run_stage(args.stage, cfg, manifest)
    print(f"{args.stage} ({cfg.project.name}) -> {result.value}")
    return 0


def _cmd_tag(args: argparse.Namespace) -> int:
    cfg, manifest = _open(args)
    tag = dict(kv.split("=", 1) for kv in args.set)
    manifest.set_tag(args.video_id, tag)
    print(f"tagged {args.video_id}: {tag}")
    return 0


def _cmd_reorg(args: argparse.Namespace) -> int:
    cfg, manifest = _open(args)
    ids = [args.video_id] if args.video_id else [v["video_id"] for v in manifest.list_videos()]
    for vid in ids:
        result = reorg_video(cfg, manifest, vid, mode=args.mode)
        print(f"reorg {args.mode} {vid} -> {result}")
    return 0


def _cmd_render_preview(args: argparse.Namespace) -> int:
    cfg, manifest = _open(args)
    row = manifest.get_row(args.video_id, "boundary")
    geom = (row.get("params") or {}).get("roi") if row else None
    if not geom:
        print("render-preview: no stored ROI geometry for this video")
        return 1
    frame = read_frame(video_output(cfg, args.video_id, "rotate"), cfg.project.boundary.sample_frame)
    write_preview(frame, geom, boundary_preview(cfg, args.video_id))
    print(f"preview re-rendered: {boundary_preview(cfg, args.video_id)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="behaviarium")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-project", help="scaffold a NEW external project folder")
    p_init.add_argument("project", help="project directory to create")
    p_init.add_argument("--data", required=True, help="path to the raw video root (any layout)")
    p_init.add_argument("--template", default="pt_social", help="repo template to copy")
    p_init.set_defaults(func=_cmd_init_project)

    p_ingest = sub.add_parser("ingest", help="discover videos and seed the manifest")
    p_ingest.add_argument("--project", required=True, help="project directory")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_run = sub.add_parser("run", help="run one stage for one video")
    p_run.add_argument("--project", required=True)
    p_run.add_argument("--stage", required=True)
    p_run.add_argument("--video-id", required=True)
    p_run.set_defaults(func=_cmd_run)

    p_all = sub.add_parser("run-all", help="run one stage for every video")
    p_all.add_argument("--project", required=True)
    p_all.add_argument("--stage", required=True)
    p_all.set_defaults(func=_cmd_run_all)

    p_proj = sub.add_parser("run-project", help="run one project-scoped stage (postprocess/stats/export)")
    p_proj.add_argument("--project", required=True)
    p_proj.add_argument("--stage", required=True)
    p_proj.set_defaults(func=_cmd_run_project)

    p_tag = sub.add_parser("tag", help="tag a video into a design cell")
    p_tag.add_argument("--project", required=True)
    p_tag.add_argument("--video-id", required=True)
    p_tag.add_argument("--set", required=True, nargs="+", metavar="factor=level",
                       help="one factor=level per design factor")
    p_tag.set_defaults(func=_cmd_tag)

    p_reorg = sub.add_parser("reorg", help="move a source video into its per-video folder")
    p_reorg.add_argument("--project", required=True)
    p_reorg.add_argument("--video-id", default=None, help="omit to reorg all videos")
    p_reorg.add_argument("--mode", choices=MODES, default="copy")
    p_reorg.set_defaults(func=_cmd_reorg)

    p_prev = sub.add_parser("render-preview", help="redraw boundary overlay from stored geometry")
    p_prev.add_argument("--project", required=True)
    p_prev.add_argument("--video-id", required=True)
    p_prev.set_defaults(func=_cmd_render_preview)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
