import sqlite3

from behaviarium.manifest import Manifest, Status, VideoKey


def test_upsert_video_and_stage_status(tmp_path):
    mf = Manifest(tmp_path / "m.db")
    mf.init()
    key = VideoKey("A", "B", "v.avi")

    mf.upsert_video(key, "/x/v.avi", 60, 0.1)
    assert mf.get_video(key)["frame_count"] == 60

    # upsert is idempotent on the (type, class, filename) key
    mf.upsert_video(key, "/x/v.avi", 120, 0.2)
    assert len(mf.list_videos()) == 1
    assert mf.get_video(key)["frame_count"] == 120

    mf.upsert(key, "ingest", status=Status.PENDING)
    assert mf.get_status(key, "ingest") == "pending"
    mf.set_status(key, "ingest", Status.DONE)
    assert mf.get_status(key, "ingest") == "done"

    # set_status upserts when the row doesn't exist yet
    mf.set_status(key, "dlc", Status.FAILED, error="boom")
    assert mf.get_status(key, "dlc") == "failed"

    failed = mf.query(status="failed")
    assert len(failed) == 1
    assert failed[0]["error"] == "boom"
    assert len(mf.query(stage="ingest")) == 1


def test_wal_mode_enabled(tmp_path):
    mf = Manifest(tmp_path / "m.db")
    mf.init()
    con = sqlite3.connect(mf.path)
    mode = con.execute("PRAGMA journal_mode").fetchone()[0]
    con.close()
    assert mode.lower() == "wal"
