import pandas as pd
import pytest

from behaviarium import stages  # noqa: F401  registers stages
from behaviarium.class_parser import parse_class
from behaviarium.config import load_config
from behaviarium.manifest import Manifest, Status, VideoKey
from behaviarium.paths import (
    bsoid_clusters_parquet,
    chamber_parquet,
    postprocess_bsoid_long,
    stats_bsoid_table,
    stats_chamber_table,
)
from behaviarium.runner import project_key, run_stage

_SPECS = [("cohortA", "Sal-N_PT"), ("cohortA", "Sal-N_EE"), ("cohortB", "LPS-N_PT"), ("cohortB", "DH_EE")]
_REGIONS = {"left_near": 0.5, "center_near": 0.3, "right_near": 0.2}


def _write_video_outputs(cfg, key, cluster_fracs):
    factors = parse_class(key.klass, cfg.project.class_parser)
    base = {"Type": key.type, "Class": key.klass, "Filename": key.filename, **factors}
    bdf = pd.DataFrame(
        [{**base, "cluster": c, "frame_count": int(f * 100), "time_s": 0.0, "fraction": f}
         for c, f in enumerate(cluster_fracs)]
    )
    rdf = pd.DataFrame(
        [{**base, "region": r, "frame_count": int(f * 100), "time_s": 0.0, "fraction": f}
         for r, f in _REGIONS.items()]
    )
    bp = bsoid_clusters_parquet(cfg, key)
    bp.parent.mkdir(parents=True, exist_ok=True)
    bdf.to_parquet(bp, index=False)
    cp = chamber_parquet(cfg, key)
    cp.parent.mkdir(parents=True, exist_ok=True)
    rdf.to_parquet(cp, index=False)


def _seed(monkeypatch, tmp_path, make_rect_video, project, write_outputs=True):
    data = tmp_path / "data"
    for i, (t, k) in enumerate(_SPECS):
        make_rect_video(data / t / k / f"m{i}.avi", n_frames=10, size=(40, 40), rect=(0, 0, 40, 40))
    monkeypatch.setenv("BEHAVIARIUM_DATA_ROOT", str(data))
    monkeypatch.setenv("BEHAVIARIUM_OUTPUT_ROOT", str(tmp_path / "out"))
    cfg = load_config(project)
    manifest = Manifest(cfg.manifest_path)
    manifest.init()
    run_stage("ingest", cfg, manifest)
    keys = [VideoKey(t, k, f"m{i}.avi") for i, (t, k) in enumerate(_SPECS)]
    if write_outputs:
        for i, key in enumerate(keys):
            _write_video_outputs(cfg, key, [0.2, 0.3, 0.5] if i % 2 == 0 else [0.5, 0.3, 0.2])
    return cfg, manifest, keys


def test_postprocess_and_stats_end_to_end(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, keys = _seed(monkeypatch, tmp_path, make_rect_video, "pt_social")
    assert run_stage("postprocess", cfg, manifest) == Status.DONE
    assert run_stage("stats", cfg, manifest) == Status.DONE

    blong = pd.read_parquet(postprocess_bsoid_long(cfg, ".parquet"))
    assert len(blong) == 4 * 3  # 4 videos x 3 clusters
    assert "housing" in blong.columns  # parsed factor carried through to the aggregate
    # decision #4: NO p-value-like columns anywhere except the real stats output
    assert not ({"p_value", "q_value", "significant"} & set(blong.columns))

    sdf = pd.read_csv(stats_bsoid_table(cfg, ".csv"))
    assert {"cluster", "wasserstein_stat", "p_value", "q_value", "significant"} <= set(sdf.columns)
    # groups come from the config group_factor (pt_social: housing -> PT vs EE)
    assert {sdf.iloc[0]["group_a"], sdf.iloc[0]["group_b"]} == {"EE", "PT"}

    rdf = pd.read_csv(stats_chamber_table(cfg, ".csv"))
    assert {"region", "p_value", "q_value", "significant"} <= set(rdf.columns)

    params = manifest.get_row(project_key(cfg), "stats")["params"]
    assert params["group_factor"] == "housing" and params["n_permutations"] == 1000


def test_oft_groups_by_type_proves_genericity(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, keys = _seed(monkeypatch, tmp_path, make_rect_video, "oft_demo")
    run_stage("postprocess", cfg, manifest)
    run_stage("stats", cfg, manifest)
    sdf = pd.read_csv(stats_bsoid_table(cfg, ".csv"))
    # oft_demo group_factor=Type -> cohortA vs cohortB (different column than pt_social)
    assert {sdf.iloc[0]["group_a"], sdf.iloc[0]["group_b"]} == {"cohortA", "cohortB"}


def test_decision4_no_fake_pvalue_in_per_video_outputs(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, keys = _seed(monkeypatch, tmp_path, make_rect_video, "pt_social")
    for key in keys:
        assert "p_value" not in pd.read_parquet(bsoid_clusters_parquet(cfg, key)).columns
        assert "p_value" not in pd.read_parquet(chamber_parquet(cfg, key)).columns


def test_postprocess_requires_upstream(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, keys = _seed(monkeypatch, tmp_path, make_rect_video, "pt_social", write_outputs=False)
    with pytest.raises(RuntimeError, match="run bsoid first"):
        run_stage("postprocess", cfg, manifest)
    assert manifest.get_status(project_key(cfg), "postprocess") == Status.FAILED.value


def test_stats_parquet_csv_round_trip(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, keys = _seed(monkeypatch, tmp_path, make_rect_video, "pt_social")
    run_stage("postprocess", cfg, manifest)
    run_stage("stats", cfg, manifest)
    pq = pd.read_parquet(stats_bsoid_table(cfg, ".parquet"))
    csv = pd.read_csv(stats_bsoid_table(cfg, ".csv"))
    pd.testing.assert_frame_equal(pq, csv, check_dtype=False)


def test_postprocess_idempotent_skip(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, keys = _seed(monkeypatch, tmp_path, make_rect_video, "pt_social")
    assert run_stage("postprocess", cfg, manifest) == Status.DONE
    assert run_stage("postprocess", cfg, manifest) == Status.DONE
