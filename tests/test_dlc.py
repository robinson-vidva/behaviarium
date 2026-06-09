import shutil

import pytest

from behaviarium import dlc_io, stages  # noqa: F401  registers stages
from behaviarium.config import load_config
from behaviarium.manifest import Manifest, Status, VideoKey
from behaviarium.paths import dlc_filtered_path, dlc_raw_path, video_output
from behaviarium.runner import run_stage


def _prep(monkeypatch, tmp_path, make_rect_video, project, with_mask=True):
    data = tmp_path / "data"
    make_rect_video(data / "T" / "C" / "v.avi", n_frames=12, size=(120, 100), rect=(20, 15, 70, 55))
    monkeypatch.setenv("BEHAVIARIUM_DATA_ROOT", str(data))
    monkeypatch.setenv("BEHAVIARIUM_OUTPUT_ROOT", str(tmp_path / "out"))
    cfg = load_config(project)
    manifest = Manifest(cfg.manifest_path)
    manifest.init()
    run_stage("ingest", cfg, manifest)
    key = VideoKey("T", "C", "v.avi")
    if with_mask:
        # Decouple dlc tests from the boundary-approval chain: a readable mask video is all
        # the stub backend needs (it probes frame count + dims, not pixel content).
        mask = video_output(cfg, key, "mask")
        mask.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest.get_video(key)["path"], mask)
    return cfg, manifest, key


def test_stub_schema_and_by_name_reader(monkeypatch, tmp_path, make_rect_video):
    # oft_demo: engine=stub, filter enabled
    cfg, manifest, key = _prep(monkeypatch, tmp_path, make_rect_video, "oft_demo")
    run_stage("dlc", cfg, manifest, key)

    out = dlc_filtered_path(cfg, key)  # oft has filtering on
    assert out.exists()
    df = dlc_io.read_dlc_csv(out)
    # exact multiindex schema
    assert list(df.columns.names) == ["scorer", "bodyparts", "coords"]
    assert dlc_io.list_bodyparts(df) == cfg.project.dlc.bodyparts  # ["nose","center","tailbase"]
    # bodypart accessed BY NAME (never positional x.1 / x.11)
    nose = dlc_io.get_bodypart(df, "nose")
    assert list(nose.columns) == ["x", "y", "likelihood"]
    assert len(nose) == 12

    params = manifest.get_row(key, "dlc")["params"]
    assert params["engine_requested"] == "stub"
    assert params["backend"] == "stub"
    assert params["filtered"] is True
    assert params["filter_path"] == "pandas-median"  # default path


def test_raw_vs_filtered_naming_and_median_changes_values(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, key = _prep(monkeypatch, tmp_path, make_rect_video, "oft_demo")
    run_stage("dlc", cfg, manifest, key)
    raw, filt = dlc_raw_path(cfg, key), dlc_filtered_path(cfg, key)
    assert raw.exists() and filt.exists()  # raw kept; filtered honestly derived from it

    raw_df = dlc_io.read_dlc_csv(raw)
    filt_df = dlc_io.read_dlc_csv(filt)
    # the median filter must actually change values
    assert not raw_df.equals(filt_df)
    assert (dlc_io.get_bodypart(raw_df, "nose")["x"].values
            != dlc_io.get_bodypart(filt_df, "nose")["x"].values).any()


def test_disabled_filter_emits_raw_only(monkeypatch, tmp_path, make_rect_video):
    # pt_social: engine=tensorflow (falls back to stub on Mac), filter disabled
    cfg, manifest, key = _prep(monkeypatch, tmp_path, make_rect_video, "pt_social")
    run_stage("dlc", cfg, manifest, key)

    assert dlc_raw_path(cfg, key).exists()
    assert not dlc_filtered_path(cfg, key).exists()  # never fake a _filtered file
    params = manifest.get_row(key, "dlc")["params"]
    assert params["engine_requested"] == "tensorflow"
    assert params["backend"].startswith("stub (fallback")  # no TF on Mac
    assert params["filtered"] is False
    assert params["filter_path"] is None  # no filtering ran
    assert dlc_io.list_bodyparts(dlc_io.read_dlc_csv(dlc_raw_path(cfg, key))) == cfg.project.dlc.bodyparts


def test_delegate_to_dlc_with_stub_backend_raises(monkeypatch, tmp_path, make_rect_video):
    # delegate_to_dlc=true is DLC-exact and must NOT silently fall back to pandas on the stub
    cfg, manifest, key = _prep(monkeypatch, tmp_path, make_rect_video, "oft_demo")
    cfg.project.dlc.filter.delegate_to_dlc = True  # engine is stub -> must refuse
    with pytest.raises(RuntimeError, match="requires the tensorflow backend"):
        run_stage("dlc", cfg, manifest, key)
    assert manifest.get_status(key, "dlc") == Status.FAILED.value
    assert not dlc_filtered_path(cfg, key).exists()  # no _filtered emitted


def test_dlc_requires_mask_output(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, key = _prep(monkeypatch, tmp_path, make_rect_video, "oft_demo", with_mask=False)
    with pytest.raises(RuntimeError, match="run mask first"):
        run_stage("dlc", cfg, manifest, key)
    assert manifest.get_status(key, "dlc") == Status.FAILED.value


def test_dlc_idempotent_skip(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, key = _prep(monkeypatch, tmp_path, make_rect_video, "oft_demo")
    assert run_stage("dlc", cfg, manifest, key) == Status.DONE
    assert run_stage("dlc", cfg, manifest, key) == Status.DONE  # short-circuits via skip-if-done
