import pandas as pd

from behaviarium.manifest import PROJECT_ID, Status
from behaviarium.paths import chamber_parquet, postprocess_bsoid_long
from behaviarium.runner import eligible_video_ids, is_processable, run_stage

from .conftest import run_chain


def test_untagged_video_runs_per_video_stages(new_project):
    # decoupled: an untagged (but included) video runs rotate -> chamber, not skipped
    cfg, manifest = new_project(specs=[("m1.avi", 80)])
    vid = "m1"
    assert is_processable(cfg, manifest, vid) is False  # untagged -> not eligible for grouping
    run_chain(cfg, manifest, vid)
    for s in ("rotate", "boundary", "mask", "dlc", "chamber"):
        assert manifest.get_status(vid, s) == Status.DONE.value
    # output keyed by video_id, with NO factor columns yet (untagged)
    occ = pd.read_parquet(chamber_parquet(cfg, vid))
    assert "video_id" in occ.columns and "treatment" not in occ.columns


def test_untagged_excluded_from_tag_required_stages(new_project):
    cfg, manifest = new_project(specs=[("m1.avi", 80), ("m2.avi", 80)])
    manifest.set_tag("m1", {"treatment": "Sal-N", "housing": "PT"})  # m2 left untagged
    for s in ("postprocess", "stats", "export"):
        elig = eligible_video_ids(cfg, manifest, s)
        assert "m1" in elig and "m2" not in elig
    # per-video stages see both
    assert set(eligible_video_ids(cfg, manifest, "chamber")) == {"m1", "m2"}


def test_pretag_processing_flows_into_grouping_after_tagging(new_project):
    # m1 tagged from the start; m2 processed UNTAGGED, tagged later
    cfg, manifest = new_project(specs=[("m1.avi", 80), ("m2.avi", 80)])
    manifest.set_tag("m1", {"treatment": "Sal-N", "housing": "PT"})
    run_chain(cfg, manifest, "m1")
    run_chain(cfg, manifest, "m2")  # untagged, still runs

    # postprocess excludes the untagged m2
    run_stage("postprocess", cfg, manifest)
    blong = pd.read_parquet(postprocess_bsoid_long(cfg, ".parquet"))
    assert set(blong["video_id"]) == {"m1"}

    # tag m2 and re-run ONLY the project stage (reset status, as the UI "Re-run" does) —
    # it now flows in with correct factor columns
    manifest.set_tag("m2", {"treatment": "Sal-N", "housing": "EE"})
    manifest.set_status(PROJECT_ID, "postprocess", Status.PENDING)
    run_stage("postprocess", cfg, manifest)
    blong = pd.read_parquet(postprocess_bsoid_long(cfg, ".parquet"))
    assert set(blong["video_id"]) == {"m1", "m2"}
    m2 = blong[blong["video_id"] == "m2"]
    assert (m2["treatment"] == "Sal-N").all() and (m2["housing"] == "EE").all()


def test_excluded_video_gated_for_all_stages(new_project):
    cfg, manifest = new_project(specs=[("m1.avi", 30)])
    manifest.set_tag("m1", {"treatment": "Sal-N", "housing": "PT"})
    manifest.set_include("m1", False)
    assert run_stage("rotate", cfg, manifest, "m1") == Status.SKIPPED  # include gates per-video too
    assert "m1" not in eligible_video_ids(cfg, manifest, "chamber")
    assert "m1" not in eligible_video_ids(cfg, manifest, "postprocess")


def test_partial_tag_runs_per_video_but_not_grouping(new_project):
    cfg, manifest = new_project(specs=[("m1.avi", 30)])
    manifest.set_tag("m1", {"treatment": "Sal-N"})  # missing housing
    assert is_processable(cfg, manifest, "m1") is False
    assert "m1" in eligible_video_ids(cfg, manifest, "rotate")  # per-video runs
    assert "m1" not in eligible_video_ids(cfg, manifest, "postprocess")  # grouping does not
