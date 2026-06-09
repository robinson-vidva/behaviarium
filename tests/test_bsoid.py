import numpy as np
import pandas as pd
import pytest

from behaviarium import dlc_io, stages  # noqa: F401  registers stages
from behaviarium.bsoid_reconstruct import n_shift_for_fps, offset_lengths, reconstruct_labels
from behaviarium.config import load_config
from behaviarium.manifest import Manifest, Status, VideoKey
from behaviarium.paths import (
    bsoid_clusters_csv,
    bsoid_clusters_parquet,
    bsoid_labels_csv,
    bsoid_labels_parquet,
    dlc_output_path,
)
from behaviarium.runner import run_stage


# ----------------------------------------------------------------------------------------
# decision #2 — the reconstruction (most important)
# ----------------------------------------------------------------------------------------
def test_frameshift_flatten_interleave_known_sequence_not_block_repeat():
    # n_shift=3, n_frames=7. streams[j] = predictions for frame-offset j.
    s0 = np.array([10, 11, 12])  # frames 0, 3, 6
    s1 = np.array([20, 21])      # frames 1, 4
    s2 = np.array([30, 31])      # frames 2, 5
    out = reconstruct_labels([s0, s1, s2], 7)

    expected = [10, 20, 30, 11, 21, 31, 12]  # interleaved per-frame
    assert out.tolist() == expected
    assert len(out) == 7  # trimmed to the original frame count

    # explicitly NOT the legacy block-repeat-each-label pattern (np.repeat / tile)
    block_repeat = np.repeat(s0, 3)[:7].tolist()  # [10,10,10,11,11,11,12]
    assert out.tolist() != block_repeat


def test_reconstruct_maps_each_frame_to_its_offset_and_bin():
    n_frames, n_shift = 10, 3
    lengths = offset_lengths(n_frames, n_shift)  # [4, 3, 3]
    streams = [np.array([j * 100 + b for b in range(L)]) for j, L in enumerate(lengths)]
    out = reconstruct_labels(streams, n_frames)
    # frame i comes from stream (i % n_shift) at bin (i // n_shift)
    expected = [(i % n_shift) * 100 + (i // n_shift) for i in range(n_frames)]
    assert out.tolist() == expected


def test_n_shift_from_corrected_fps():
    # floor(fps/10) per the authoritative YttriLab B-SOiD source (not round)
    assert n_shift_for_fps(84) == 8   # floor(8.4)
    assert n_shift_for_fps(37) == 3   # floor(3.7) -> 3 (round would give 4)
    assert n_shift_for_fps(30) == 3
    assert n_shift_for_fps(0.05) == 1  # guarded floor of 1, never 0


# ----------------------------------------------------------------------------------------
# stub end-to-end
# ----------------------------------------------------------------------------------------
def _prep(monkeypatch, tmp_path, make_rect_video, project, n=80, duration="1", write_dlc=True):
    klass = "Sal-N_PT" if project == "pt_social" else "PT"
    data = tmp_path / "data"
    make_rect_video(data / "cohortA" / klass / "v.avi", n_frames=n, size=(80, 64), rect=(0, 0, 80, 64))
    monkeypatch.setenv("BEHAVIARIUM_DATA_ROOT", str(data))
    monkeypatch.setenv("BEHAVIARIUM_OUTPUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("BEHAVIARIUM_RECORDING_DURATION_S", duration)  # tiny clips -> realistic fps
    cfg = load_config(project)
    manifest = Manifest(cfg.manifest_path)
    manifest.init()
    run_stage("ingest", cfg, manifest)
    key = VideoKey("cohortA", klass, "v.avi")
    if write_dlc:
        df = dlc_io.build_dlc_dataframe("DLC_test_sh1", ["bp"], np.zeros((n, 3)))
        dlc_io.write_dlc_csv(df, dlc_output_path(cfg, key, cfg.project.dlc.filter.enabled))
    return cfg, manifest, key


def test_stub_labels_over_range_n_clusters_and_time(monkeypatch, tmp_path, make_rect_video):
    # oft_demo: engine=stub, n_clusters=6. n=80, duration=1 -> fps=80 -> n_shift=8
    cfg, manifest, key = _prep(monkeypatch, tmp_path, make_rect_video, "oft_demo", n=80, duration="1")
    run_stage("bsoid", cfg, manifest, key)

    params = manifest.get_row(key, "bsoid")["params"]
    assert params["engine_requested"] == "stub" and params["backend"] == "stub"
    assert params["n_shift"] == 8
    assert params["n_clusters"] == 6

    labels = pd.read_parquet(bsoid_labels_parquet(cfg, key))
    assert len(labels) == 80  # per-frame, trimmed to frame count
    assert labels["label"].between(0, 5).all()  # range(n_clusters)

    clusters = pd.read_csv(bsoid_clusters_csv(cfg, key))
    assert list(clusters["cluster"]) == list(range(6))
    assert int(clusters["frame_count"].sum()) == 80
    fps = manifest.get_video(key)["fps"]
    r0 = clusters.iloc[0]
    assert r0["time_s"] == r0["frame_count"] / fps  # exact, corrected_fps


def test_key_and_factors_present_and_real_falls_back(monkeypatch, tmp_path, make_rect_video):
    # pt_social: engine=real -> falls back to stub on Mac (no B-SOiD); Class parser adds factors
    cfg, manifest, key = _prep(monkeypatch, tmp_path, make_rect_video, "pt_social", n=40, duration="1")
    run_stage("bsoid", cfg, manifest, key)

    params = manifest.get_row(key, "bsoid")["params"]
    assert params["engine_requested"] == "real"
    assert params["backend"].startswith("stub (fallback")  # no heavy deps on Mac
    assert params["n_clusters"] == 14
    assert params["n_shift"] == 4  # round(40/10)

    labels = pd.read_csv(bsoid_labels_csv(cfg, key))
    for col in ["Type", "Class", "Filename", "treatment", "housing", "frame", "label"]:
        assert col in labels.columns
    assert (labels["treatment"] == "Sal-N").all()
    assert labels["label"].between(0, 13).all()


def test_bsoid_parquet_csv_round_trip(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, key = _prep(monkeypatch, tmp_path, make_rect_video, "oft_demo")
    run_stage("bsoid", cfg, manifest, key)
    pq = pd.read_parquet(bsoid_clusters_parquet(cfg, key))
    csv = pd.read_csv(bsoid_clusters_csv(cfg, key))
    pd.testing.assert_frame_equal(pq, csv, check_dtype=False)


def test_bsoid_requires_dlc_output(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, key = _prep(monkeypatch, tmp_path, make_rect_video, "oft_demo", write_dlc=False)
    with pytest.raises(RuntimeError, match="run dlc first"):
        run_stage("bsoid", cfg, manifest, key)
    assert manifest.get_status(key, "bsoid") == Status.FAILED.value


def test_bsoid_idempotent_skip(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, key = _prep(monkeypatch, tmp_path, make_rect_video, "oft_demo")
    assert run_stage("bsoid", cfg, manifest, key) == Status.DONE
    assert run_stage("bsoid", cfg, manifest, key) == Status.DONE
