import pytest

from behaviarium import stages  # noqa: F401  registers stages
from behaviarium.config import load_config
from behaviarium.manifest import Approval, Manifest, Status, VideoKey
from behaviarium.paths import boundary_preview, video_output
from behaviarium.runner import run_stage


def _setup(monkeypatch, tmp_path, make_rect_video, project="pt_social"):
    data = tmp_path / "data"
    make_rect_video(data / "T" / "C" / "v.avi", n_frames=4, size=(200, 160), rect=(40, 30, 100, 80))
    monkeypatch.setenv("BEHAVIARIUM_DATA_ROOT", str(data))
    monkeypatch.setenv("BEHAVIARIUM_OUTPUT_ROOT", str(tmp_path / "out"))
    cfg = load_config(project)
    manifest = Manifest(cfg.manifest_path)
    manifest.init()
    run_stage("ingest", cfg, manifest)
    return cfg, manifest, VideoKey("T", "C", "v.avi")


def test_rotate_boundary_mask_chain(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, key = _setup(monkeypatch, tmp_path, make_rect_video)

    # rotate (pt_social degrees=0): output exists, status done
    run_stage("rotate", cfg, manifest, key)
    assert video_output(cfg, key, "rotate").exists()
    assert manifest.get_status(key, "rotate") == Status.DONE.value

    # boundary: geometry stored, approval pending_review, preview PNG written
    run_stage("boundary", cfg, manifest, key)
    brow = manifest.get_row(key, "boundary")
    assert brow["approval"] == Approval.PENDING_REVIEW.value
    geom = brow["params"]["roi"]
    assert geom["shape"] == "rect"
    assert abs(geom["x"] - 40) <= 6 and abs(geom["w"] - 100) <= 8
    assert boundary_preview(cfg, key).exists()

    # mask refuses without approval
    with pytest.raises(RuntimeError):
        run_stage("mask", cfg, manifest, key)
    assert manifest.get_status(key, "mask") == Status.FAILED.value

    # approve, then mask succeeds and writes the DLC-ready video
    manifest.set_approval(key, "boundary", Approval.APPROVED)
    run_stage("mask", cfg, manifest, key)
    assert video_output(cfg, key, "mask").exists()
    assert manifest.get_status(key, "mask") == Status.DONE.value


def test_rotate_is_idempotent_skip(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, key = _setup(monkeypatch, tmp_path, make_rect_video)
    assert run_stage("rotate", cfg, manifest, key) == Status.DONE
    # second call short-circuits via skip-if-done (output exists + status done)
    assert run_stage("rotate", cfg, manifest, key) == Status.DONE


def test_excluded_video_is_skipped_by_runner(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, key = _setup(monkeypatch, tmp_path, make_rect_video)
    manifest.set_include(key, False)
    result = run_stage("rotate", cfg, manifest, key)
    assert result == Status.SKIPPED
    assert not video_output(cfg, key, "rotate").exists()
    assert manifest.get_status(key, "rotate") is None  # untouched
