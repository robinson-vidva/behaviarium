from pathlib import Path

from behaviarium.identity import make_video_id, slugify
from behaviarium.manifest import Manifest
from behaviarium.runner import run_stage


def test_video_id_slug_and_dedupe():
    assert slugify("Mouse 01 (A).avi") == "mouse-01-a-avi"
    existing = set()
    a = make_video_id("clip.avi", existing); existing.add(a)
    b = make_video_id("clip.avi", existing); existing.add(b)
    c = make_video_id("clip.avi", existing)
    assert a == "clip" and b == "clip-2" and c == "clip-3"


def test_discovery_flat_and_nested_with_source_path(new_project):
    # two videos, one flat + one nested under data_path
    cfg, manifest = new_project(specs=[("alpha.avi", 30), ("beta.avi", 40)], nested=True)
    vids = {v["video_id"]: v for v in manifest.list_videos()}
    assert set(vids) == {"alpha", "beta"}
    # source_path is the original absolute path; nested one is under data/nested/
    assert Path(vids["beta"]["source_path"]).name == "beta.avi"
    assert "nested" in vids["beta"]["source_path"]
    # current_path == source_path until reorg
    assert vids["alpha"]["current_path"] == vids["alpha"]["source_path"]


def test_filename_collision_dedupes_video_id(tmp_path, monkeypatch, make_rect_video):
    from behaviarium.config import init_project

    monkeypatch.setenv("BEHAVIARIUM_RECORDING_DURATION_S", "1")
    data = tmp_path / "raw"
    make_rect_video(data / "g1" / "clip.avi", n_frames=20)  # same filename, different folders
    make_rect_video(data / "g2" / "clip.avi", n_frames=25)
    cfg = init_project(tmp_path / "proj", data)
    mf = Manifest(cfg.manifest_path)
    run_stage("ingest", cfg, mf)
    ids = sorted(v["video_id"] for v in mf.list_videos())
    assert ids == ["clip", "clip-2"]  # stable + deduped


def test_reingest_is_idempotent_and_stable(new_project):
    cfg, manifest = new_project(specs=[("m1.avi", 30)])
    first = manifest.list_videos()[0]["video_id"]
    run_stage("ingest", cfg, manifest)  # again
    vids = manifest.list_videos()
    assert len(vids) == 1 and vids[0]["video_id"] == first  # no duplicate, same id
