from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import behaviarium.stages  # noqa: F401  registers stages
from behaviarium.config import init_project
from behaviarium.manifest import Approval, Manifest
from behaviarium.runner import run_stage


def write_rect_video(path: Path, n_frames: int = 8, size=(80, 64), rect=(0, 0, 80, 64), bright: int = 255) -> Path:
    """Tiny MJPG/AVI clip with a known bright rectangle 'arena'. ``size`` = (w, h)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    x, y, rw, rh = rect
    base = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(base, (x, y), (x + rw, y + rh), (bright, bright, bright), -1)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h))
    assert wr.isOpened(), f"VideoWriter failed to open {path}"
    for _ in range(n_frames):
        wr.write(base.copy())
    wr.release()
    return path


def write_synthetic_video(path: Path, n_frames: int, size=(64, 64)) -> Path:
    return write_rect_video(path, n_frames=n_frames, size=size, rect=(0, 0, size[0], size[1]))


@pytest.fixture
def make_rect_video():
    return write_rect_video


@pytest.fixture
def make_video():
    return write_synthetic_video


@pytest.fixture
def new_project(tmp_path, monkeypatch):
    """Create synthetic video data + an EXTERNAL project, run ingest. Returns (cfg, manifest)."""
    monkeypatch.setenv("BEHAVIARIUM_RECORDING_DURATION_S", "1")  # tiny clips -> realistic fps

    def _make(template="pt_social", specs=None, nested=False, data_subdir="raw"):
        data = tmp_path / data_subdir
        specs = specs or [("m1.avi", 80), ("m2.avi", 80)]
        for i, (fn, n) in enumerate(specs):
            sub = data / "nested" if (nested and i % 2) else data
            write_rect_video(sub / fn, n_frames=n)
        cfg = init_project(tmp_path / "proj", data, template=template)
        manifest = Manifest(cfg.manifest_path)
        run_stage("ingest", cfg, manifest)
        return cfg, manifest

    return _make


def tag_round_robin(cfg, manifest):
    """Tag each video into a design cell (cycling through cells); returns [(video_id, tag), ...]."""
    cells = cfg.project.design.cells() or [{}]
    out = []
    for i, v in enumerate(manifest.list_videos()):
        cell = cells[i % len(cells)]
        manifest.set_tag(v["video_id"], cell or None)
        out.append((v["video_id"], cell))
    return out


def run_chain(cfg, manifest, video_id):
    """rotate -> boundary -> approve -> mask -> dlc -> chamber -> bsoid for one tagged video."""
    run_stage("rotate", cfg, manifest, video_id)
    run_stage("boundary", cfg, manifest, video_id)
    manifest.set_approval(video_id, "boundary", Approval.APPROVED)
    for s in ("mask", "dlc", "chamber", "bsoid"):
        run_stage(s, cfg, manifest, video_id)
