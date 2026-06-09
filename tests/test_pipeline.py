import pandas as pd

from behaviarium.manifest import Status
from behaviarium.paths import bsoid_clusters_parquet, chamber_parquet, video_output
from behaviarium.runner import run_stage

from .conftest import run_chain


def test_full_per_video_chain(new_project):
    cfg, manifest = new_project(template="pt_social", specs=[("m1.avi", 80)])
    vid = "m1"
    manifest.set_tag(vid, {"treatment": "Sal-N", "housing": "PT"})
    run_chain(cfg, manifest, vid)

    for s in ("rotate", "boundary", "mask", "dlc", "chamber", "bsoid"):
        assert manifest.get_status(vid, s) == Status.DONE.value
    assert video_output(cfg, vid, "rotate").exists()
    assert chamber_parquet(cfg, vid).exists()
    assert bsoid_clusters_parquet(cfg, vid).exists()
    # all per-video artifacts live in the per-video folder
    assert chamber_parquet(cfg, vid).parent == cfg.videos_dir / vid


def test_mask_requires_approval(new_project):
    import pytest

    cfg, manifest = new_project(specs=[("m1.avi", 30)])
    vid = "m1"
    manifest.set_tag(vid, {"treatment": "Sal-N", "housing": "PT"})
    run_stage("rotate", cfg, manifest, vid)
    run_stage("boundary", cfg, manifest, vid)
    with pytest.raises(RuntimeError, match="APPROVED"):
        run_stage("mask", cfg, manifest, vid)


def test_idempotent_skip(new_project):
    cfg, manifest = new_project(specs=[("m1.avi", 30)])
    vid = "m1"
    manifest.set_tag(vid, {"treatment": "Sal-N", "housing": "PT"})
    assert run_stage("rotate", cfg, manifest, vid) == Status.DONE
    assert run_stage("rotate", cfg, manifest, vid) == Status.DONE  # skip-if-done
