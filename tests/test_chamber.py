import shutil

import numpy as np
import pandas as pd
import pytest

from behaviarium import dlc_io, stages  # noqa: F401  registers stages
from behaviarium.class_parser import parse_class
from behaviarium.config import load_config
from behaviarium.manifest import Manifest, Status, VideoKey
from behaviarium.paths import chamber_csv, chamber_parquet, dlc_output_path, video_output
from behaviarium.runner import run_stage
from behaviarium.stages.chamber import assign_regions


def _prep(monkeypatch, tmp_path, make_rect_video, project, parked=None, n=100, dims=(120, 100), write_dlc=True):
    klass = "Sal-N_PT" if project == "pt_social" else "PT"
    w, h = dims
    data = tmp_path / "data"
    make_rect_video(data / "cohortA" / klass / "v.avi", n_frames=n, size=(w, h), rect=(0, 0, w, h))
    monkeypatch.setenv("BEHAVIARIUM_DATA_ROOT", str(data))
    monkeypatch.setenv("BEHAVIARIUM_OUTPUT_ROOT", str(tmp_path / "out"))
    cfg = load_config(project)
    manifest = Manifest(cfg.manifest_path)
    manifest.init()
    run_stage("ingest", cfg, manifest)
    key = VideoKey("cohortA", klass, "v.avi")

    # mask video (chamber probes its dims for normalization)
    mask = video_output(cfg, key, "mask")
    mask.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest.get_video(key)["path"], mask)

    if write_dlc:
        tbp = cfg.project.chamber.tracking_bodypart
        if parked is not None:
            coords = np.tile([parked[0], parked[1], 0.99], (n, 1))
        else:
            coords = np.column_stack([np.full(n, w / 2), np.full(n, h / 2), np.full(n, 0.99)])
        df = dlc_io.build_dlc_dataframe("DLC_test_sh1", [tbp], coords)
        dlc_io.write_dlc_csv(df, dlc_output_path(cfg, key, cfg.project.dlc.filter.enabled))
    return cfg, manifest, key


def test_parked_point_occupancy_and_time(monkeypatch, tmp_path, make_rect_video):
    # pt_social (crop=true): ROI bbox == frame, so nx=px/120, ny=py/100.
    # park at (60,20) -> nx=0.5 (center column), ny=0.2 (near) -> center_near
    cfg, manifest, key = _prep(monkeypatch, tmp_path, make_rect_video, "pt_social", parked=(60, 20), n=100)
    run_stage("chamber", cfg, manifest, key)

    occ = pd.read_csv(chamber_csv(cfg, key))
    row = occ.loc[occ["region"] == "center_near"].iloc[0]
    assert row["frame_count"] == 100
    assert row["fraction"] == 1.0
    # time_s uses corrected_fps = frame_count/recording_duration_s, exactly
    fps = manifest.get_video(key)["fps"]
    assert row["time_s"] == 100 / fps
    assert manifest.get_row(key, "chamber")["params"]["corrected_fps"] == fps


def test_tidy_long_has_key_and_parsed_factors(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, key = _prep(monkeypatch, tmp_path, make_rect_video, "pt_social", parked=(60, 20))
    run_stage("chamber", cfg, manifest, key)
    occ = pd.read_csv(chamber_csv(cfg, key))
    # (Type, Class, Filename) join key + parsed factor columns present
    for col in ["Type", "Class", "Filename", "treatment", "housing", "region", "frame_count", "time_s", "fraction"]:
        assert col in occ.columns
    assert (occ["treatment"] == "Sal-N").all()
    assert (occ["housing"] == "PT").all()
    assert (occ["Type"] == "cohortA").all()


def test_parquet_and_csv_round_trip_equal(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, key = _prep(monkeypatch, tmp_path, make_rect_video, "pt_social", parked=(60, 20))
    run_stage("chamber", cfg, manifest, key)
    pq = pd.read_parquet(chamber_parquet(cfg, key))
    csv = pd.read_csv(chamber_csv(cfg, key))
    assert pq.shape == csv.shape
    pd.testing.assert_frame_equal(pq, csv, check_dtype=False)


def test_chamber_requires_dlc_output(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, key = _prep(monkeypatch, tmp_path, make_rect_video, "pt_social", write_dlc=False)
    with pytest.raises(RuntimeError, match="run dlc first"):
        run_stage("chamber", cfg, manifest, key)
    assert manifest.get_status(key, "chamber") == Status.FAILED.value


def test_chamber_idempotent_skip(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, key = _prep(monkeypatch, tmp_path, make_rect_video, "pt_social", parked=(60, 20))
    assert run_stage("chamber", cfg, manifest, key) == Status.DONE
    assert run_stage("chamber", cfg, manifest, key) == Status.DONE  # skip-if-done


def test_oft_center_periphery_classifies():
    cfg = load_config("oft_demo")
    regions = cfg.project.chamber.regions
    nx = np.array([0.5, 0.05])
    ny = np.array([0.5, 0.05])
    labels = assign_regions(nx, ny, regions)
    assert labels[0] == "center"
    assert labels[1] == "periphery"


def test_class_parser_factor_columns():
    pt = load_config("pt_social").project.class_parser
    assert parse_class("LPS-N_EE", pt) == {"treatment": "LPS-N", "housing": "EE"}
    oft = load_config("oft_demo").project.class_parser
    assert parse_class("anything", oft) == {}  # noop proves genericity
