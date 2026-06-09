import sqlite3

from behaviarium.manifest import Approval, Manifest, Status


def test_video_crud_tag_and_current_path(tmp_path):
    mf = Manifest(tmp_path / "m.db")
    mf.init()
    mf.upsert_video("vid1", "v.avi", "/data/v.avi", "/data/v.avi", 60, 0.1)
    rec = mf.get_video("vid1")
    assert rec["filename"] == "v.avi" and rec["frame_count"] == 60 and rec["include"] == 1
    assert rec["tag"] is None

    # lookup by source path (used for idempotent re-ingest)
    assert mf.get_video_by_source_path("/data/v.avi")["video_id"] == "vid1"

    mf.set_tag("vid1", {"treatment": "Sal-N", "housing": "PT"})
    assert mf.get_tag("vid1") == {"treatment": "Sal-N", "housing": "PT"}
    mf.set_tag("vid1", None)
    assert mf.get_tag("vid1") is None

    mf.set_current_path("vid1", "/proj/videos/vid1/v.avi")
    assert mf.get_video("vid1")["current_path"] == "/proj/videos/vid1/v.avi"

    mf.set_include("vid1", False)
    assert mf.get_video("vid1")["include"] == 0


def test_stage_status_params_approval(tmp_path):
    mf = Manifest(tmp_path / "m.db")
    mf.init()
    mf.set_status("vid1", "rotate", Status.RUNNING)
    assert mf.get_status("vid1", "rotate") == "running"
    mf.set_status("vid1", "rotate", Status.DONE)
    assert mf.get_status("vid1", "rotate") == "done"

    mf.set_params("vid1", "boundary", {"roi": {"shape": "rect"}})
    mf.set_approval("vid1", "boundary", Approval.APPROVED)
    row = mf.get_row("vid1", "boundary")
    assert row["params"]["roi"]["shape"] == "rect"
    assert row["approval"] == "approved"

    mf.set_status("vid1", "dlc", Status.FAILED, error="boom")
    assert len(mf.query(status="failed")) == 1
    assert len(mf.query(video_id="vid1")) == 3


def test_wal_mode(tmp_path):
    mf = Manifest(tmp_path / "m.db")
    mf.init()
    con = sqlite3.connect(mf.path)
    mode = con.execute("PRAGMA journal_mode").fetchone()[0]
    con.close()
    assert mode.lower() == "wal"
