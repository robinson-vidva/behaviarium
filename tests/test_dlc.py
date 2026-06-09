import pytest

from behaviarium import dlc_io
from behaviarium.manifest import Approval, Status
from behaviarium.paths import dlc_filtered_path, dlc_raw_path
from behaviarium.runner import run_stage


def _to_mask(cfg, manifest, vid, tag):
    manifest.set_tag(vid, tag)
    run_stage("rotate", cfg, manifest, vid)
    run_stage("boundary", cfg, manifest, vid)
    manifest.set_approval(vid, "boundary", Approval.APPROVED)
    run_stage("mask", cfg, manifest, vid)


def test_stub_engine_filtered(new_project):
    # oft_demo: engine=stub, filter enabled -> _filtered
    cfg, manifest = new_project(template="oft_demo", specs=[("m1.avi", 40)])
    _to_mask(cfg, manifest, "m1", {"group": "A"})
    run_stage("dlc", cfg, manifest, "m1")

    assert dlc_raw_path(cfg, "m1").exists()
    assert dlc_filtered_path(cfg, "m1").exists()
    p = manifest.get_row("m1", "dlc")["params"]
    assert p["engine_requested"] == "stub" and p["backend"] == "stub"
    assert p["filtered"] is True and p["filter_path"] == "pandas-median"
    # readable by bodypart name, in the per-video folder
    df = dlc_io.read_dlc_csv(dlc_filtered_path(cfg, "m1"))
    assert dlc_io.list_bodyparts(df) == cfg.project.dlc.bodyparts


def test_tensorflow_falls_back_to_stub_raw(new_project):
    # pt_social: engine=tensorflow -> stub fallback on Mac, filter disabled -> _raw only
    cfg, manifest = new_project(template="pt_social", specs=[("m1.avi", 40)])
    _to_mask(cfg, manifest, "m1", {"treatment": "Sal-N", "housing": "PT"})
    run_stage("dlc", cfg, manifest, "m1")

    assert dlc_raw_path(cfg, "m1").exists()
    assert not dlc_filtered_path(cfg, "m1").exists()
    p = manifest.get_row("m1", "dlc")["params"]
    assert p["engine_requested"] == "tensorflow"
    assert p["backend"].startswith("stub (fallback")
    assert p["filtered"] is False and p["filter_path"] is None


def test_dlc_requires_mask(new_project):
    cfg, manifest = new_project(template="oft_demo", specs=[("m1.avi", 40)])
    manifest.set_tag("m1", {"group": "A"})
    with pytest.raises(RuntimeError, match="run mask first"):
        run_stage("dlc", cfg, manifest, "m1")
    assert manifest.get_status("m1", "dlc") == Status.FAILED.value
