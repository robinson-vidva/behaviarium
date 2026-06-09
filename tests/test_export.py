import json

import pandas as pd
import pytest

from behaviarium import dlc_io, stages  # noqa: F401  registers stages
from behaviarium.class_parser import parse_class
from behaviarium.config import load_config
from behaviarium.manifest import Manifest, Status, VideoKey
from behaviarium.paths import (
    bsoid_clusters_parquet,
    bsoid_labels_csv,
    bsoid_labels_parquet,
    chamber_parquet,
    export_dir,
)
from behaviarium.runner import project_key, run_stage

_SPECS = [("cohortA", "Sal-N_PT"), ("cohortA", "Sal-N_EE"), ("cohortB", "LPS-N_PT"), ("cohortB", "DH_EE")]
_REGIONS = {"left_near": 0.5, "center_near": 0.3, "right_near": 0.2}


def _write_video_outputs(cfg, key, n=8):
    factors = parse_class(key.klass, cfg.project.class_parser)
    base = {"Type": key.type, "Class": key.klass, "Filename": key.filename, **factors}
    # per-video bsoid clusters (3 clusters)
    bdf = pd.DataFrame([{**base, "cluster": c, "frame_count": 10, "time_s": 1.0, "fraction": 0.33}
                        for c in range(3)])
    bp = bsoid_clusters_parquet(cfg, key); bp.parent.mkdir(parents=True, exist_ok=True)
    bdf.to_parquet(bp, index=False)
    # per-video chamber occupancy
    rdf = pd.DataFrame([{**base, "region": r, "frame_count": int(f * 100), "time_s": 0.0, "fraction": f}
                        for r, f in _REGIONS.items()])
    cp = chamber_parquet(cfg, key); cp.parent.mkdir(parents=True, exist_ok=True)
    rdf.to_parquet(cp, index=False)
    # per-video bsoid per-frame labels
    ldf = pd.DataFrame({**{k: [v] * n for k, v in base.items()},
                        "frame": range(n), "label": [i % 3 for i in range(n)]})
    lp = bsoid_labels_parquet(cfg, key); lp.parent.mkdir(parents=True, exist_ok=True)
    ldf.to_parquet(lp, index=False)
    ldf.to_csv(bsoid_labels_csv(cfg, key), index=False)


def _seed(monkeypatch, tmp_path, make_rect_video, project, upstream=True):
    data = tmp_path / "data"
    for i, (t, k) in enumerate(_SPECS):
        make_rect_video(data / t / k / f"m{i}.avi", n_frames=8, size=(40, 40), rect=(0, 0, 40, 40))
    monkeypatch.setenv("BEHAVIARIUM_DATA_ROOT", str(data))
    monkeypatch.setenv("BEHAVIARIUM_OUTPUT_ROOT", str(tmp_path / "out"))
    cfg = load_config(project)
    manifest = Manifest(cfg.manifest_path)
    manifest.init()
    run_stage("ingest", cfg, manifest)
    keys = [VideoKey(t, k, f"m{i}.avi") for i, (t, k) in enumerate(_SPECS)]
    if upstream:
        for key in keys:
            _write_video_outputs(cfg, key)
        run_stage("postprocess", cfg, manifest)
        run_stage("stats", cfg, manifest)
    return cfg, manifest, keys


def _expected_files(cfg):
    bundle = export_dir(cfg)
    files = ["export_manifest.json", "data_dictionary.md"]
    for name in ["chamber_long", "bsoid_clusters", "cluster_stats", "region_stats"]:
        files += [f"{name}.parquet", f"{name}.csv"]
    return bundle, files


def test_export_assembles_bundle(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, keys = _seed(monkeypatch, tmp_path, make_rect_video, "pt_social")
    assert run_stage("export", cfg, manifest) == Status.DONE
    bundle, files = _expected_files(cfg)
    for f in files:
        assert (bundle / f).exists(), f"missing {f}"
    # per-video bsoid_labels copied (one parquet+csv per included video)
    labels = list((bundle / "bsoid_labels").glob("*.parquet"))
    assert len(labels) == len(keys)


def test_export_manifest_valid_and_matches_files(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, keys = _seed(monkeypatch, tmp_path, make_rect_video, "pt_social")
    run_stage("export", cfg, manifest)
    bundle = export_dir(cfg)
    em = json.loads((bundle / "export_manifest.json").read_text())  # valid JSON

    assert em["project"] == "pt_social" and em["assay"] == "3C_SIT"
    assert em["join_key"] == ["Type", "Class", "Filename"]
    assert em["corrected_fps"]["formula"] == "frame_count / recording_duration_s"
    assert em["n_clusters"] == 14
    assert set(em["factors"]) == {"treatment", "housing"}  # parsed factors

    # declared schema/row-count matches the actual files
    for name, d in em["datasets"].items():
        if d.get("kind") == "per_video":
            total = 0
            for part in d["parts"]:
                df = pd.read_parquet(bundle / part["parquet"])
                assert part["rows"] == len(df)
                assert list(df.columns) == d["columns"]
                total += len(df)
            assert d["rows"] == total
        else:
            df = pd.read_parquet(bundle / d["parquet"])
            assert d["rows"] == len(df)
            assert d["columns"] == list(df.columns)
            assert d["dtypes"] == {c: str(df[c].dtype) for c in df.columns}


def test_data_dictionary_documents_every_column(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, keys = _seed(monkeypatch, tmp_path, make_rect_video, "pt_social")
    run_stage("export", cfg, manifest)
    bundle = export_dir(cfg)
    dd = (bundle / "data_dictionary.md").read_text()
    em = json.loads((bundle / "export_manifest.json").read_text())
    # guard: no undocumented column — every column of every dataset appears in the dictionary
    for d in em["datasets"].values():
        for col in d["columns"]:
            assert f"`{col}`" in dd, f"undocumented column: {col}"
    assert "(Type, Class, Filename)" in dd  # join key documented


def test_export_requires_upstream(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, keys = _seed(monkeypatch, tmp_path, make_rect_video, "pt_social", upstream=False)
    with pytest.raises(RuntimeError, match="run postprocess first"):
        run_stage("export", cfg, manifest)
    assert manifest.get_status(project_key(cfg), "export") == Status.FAILED.value


def test_export_idempotent_skip(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, keys = _seed(monkeypatch, tmp_path, make_rect_video, "pt_social")
    assert run_stage("export", cfg, manifest) == Status.DONE
    assert run_stage("export", cfg, manifest) == Status.DONE


def test_export_runs_for_oft_with_its_own_assay(monkeypatch, tmp_path, make_rect_video):
    cfg, manifest, keys = _seed(monkeypatch, tmp_path, make_rect_video, "oft_demo")
    run_stage("export", cfg, manifest)
    em = json.loads((export_dir(cfg) / "export_manifest.json").read_text())
    assert em["project"] == "oft_demo" and em["assay"] == "OFT"
    assert em["n_clusters"] == 6  # oft's own cluster count
    assert em["factors"] == []  # noop parser -> no factor columns
