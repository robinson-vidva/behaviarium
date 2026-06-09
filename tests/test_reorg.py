from pathlib import Path

import pytest

from behaviarium.paths import video_dir
from behaviarium.reorg import reorg_video


def test_reorg_copy_places_file_and_is_idempotent(new_project):
    cfg, manifest = new_project(specs=[("m1.avi", 20)])
    vid = "m1"
    src = Path(manifest.get_video(vid)["source_path"])

    assert reorg_video(cfg, manifest, vid, "copy") == "copy"
    dest = video_dir(cfg, vid) / "m1.avi"
    assert dest.exists()
    assert src.exists()  # copy leaves the original
    assert manifest.get_video(vid)["current_path"] == str(dest)

    # idempotent
    assert reorg_video(cfg, manifest, vid, "copy") == "already-reorganized"


def test_reorg_move_removes_original(new_project):
    cfg, manifest = new_project(specs=[("m1.avi", 20)])
    vid = "m1"
    src = Path(manifest.get_video(vid)["source_path"])
    assert reorg_video(cfg, manifest, vid, "move") == "move"
    dest = video_dir(cfg, vid) / "m1.avi"
    assert dest.exists() and not src.exists()
    assert manifest.get_video(vid)["current_path"] == str(dest)


def test_reorg_symlink(new_project):
    cfg, manifest = new_project(specs=[("m1.avi", 20)])
    vid = "m1"
    assert reorg_video(cfg, manifest, vid, "symlink") == "symlink"
    dest = video_dir(cfg, vid) / "m1.avi"
    assert dest.is_symlink() and dest.exists()


def test_reorg_never_overwrites(new_project):
    cfg, manifest = new_project(specs=[("m1.avi", 20)])
    vid = "m1"
    dest = video_dir(cfg, vid) / "m1.avi"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("pre-existing")  # a different file already at the destination
    with pytest.raises(FileExistsError):
        reorg_video(cfg, manifest, vid, "copy")
    assert dest.read_text() == "pre-existing"  # untouched


def test_reorg_invalid_mode(new_project):
    cfg, manifest = new_project(specs=[("m1.avi", 20)])
    with pytest.raises(ValueError):
        reorg_video(cfg, manifest, "m1", "teleport")
