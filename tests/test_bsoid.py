import numpy as np
import pandas as pd

from behaviarium.bsoid_reconstruct import n_shift_for_fps, offset_lengths, reconstruct_labels
from behaviarium.paths import bsoid_clusters_csv, bsoid_labels_parquet

from .conftest import run_chain


# --- decision #2: the frameshift reconstruction (pure) ----------------------------------
def test_frameshift_flatten_interleave_known_sequence_not_block_repeat():
    s0 = np.array([10, 11, 12])  # frames 0, 3, 6
    s1 = np.array([20, 21])      # frames 1, 4
    s2 = np.array([30, 31])      # frames 2, 5
    out = reconstruct_labels([s0, s1, s2], 7)
    assert out.tolist() == [10, 20, 30, 11, 21, 31, 12]
    assert len(out) == 7
    assert out.tolist() != np.repeat(s0, 3)[:7].tolist()  # NOT block-repeat


def test_reconstruct_maps_each_frame_to_offset_and_bin():
    n_frames, n_shift = 10, 3
    lengths = offset_lengths(n_frames, n_shift)
    streams = [np.array([j * 100 + b for b in range(L)]) for j, L in enumerate(lengths)]
    out = reconstruct_labels(streams, n_frames)
    assert out.tolist() == [(i % n_shift) * 100 + (i // n_shift) for i in range(n_frames)]


def test_n_shift_floor():
    assert n_shift_for_fps(84) == 8
    assert n_shift_for_fps(37) == 3  # floor(3.7), not round
    assert n_shift_for_fps(0.05) == 1


# --- stub end-to-end --------------------------------------------------------------------
def test_bsoid_stub_labels_keyed_by_video_id(new_project):
    cfg, manifest = new_project(template="oft_demo", specs=[("m1.avi", 80)])  # fps 80 -> n_shift 8
    manifest.set_tag("m1", {"group": "A"})
    run_chain(cfg, manifest, "m1")

    params = manifest.get_row("m1", "bsoid")["params"]
    assert params["backend"] == "stub" and params["n_shift"] == 8 and params["n_clusters"] == 6

    labels = pd.read_parquet(bsoid_labels_parquet(cfg, "m1"))
    assert "video_id" in labels.columns and "group" in labels.columns
    assert not ({"Type", "Class", "Filename"} & set(labels.columns))
    assert len(labels) == 80 and labels["label"].between(0, 5).all()

    clusters = pd.read_csv(bsoid_clusters_csv(cfg, "m1"))
    assert list(clusters["cluster"]) == list(range(6))
    fps = manifest.get_video("m1")["fps"]
    assert clusters.iloc[0]["time_s"] == clusters.iloc[0]["frame_count"] / fps
