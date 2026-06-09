import json

import pandas as pd

from behaviarium.manifest import Status
from behaviarium.paths import export_dir
from behaviarium.runner import run_stage

from .conftest import run_chain, tag_round_robin


def _full(new_project, template="pt_social"):
    cfg, manifest = new_project(template=template, specs=[("m1.avi", 80), ("m2.avi", 80)])
    tag_round_robin(cfg, manifest)
    for v in manifest.list_videos():
        run_chain(cfg, manifest, v["video_id"])
    run_stage("postprocess", cfg, manifest)
    run_stage("stats", cfg, manifest)
    return cfg, manifest


def test_export_bundle_assembled_and_keyed_by_video_id(new_project):
    cfg, manifest = _full(new_project)
    assert run_stage("export", cfg, manifest) == Status.DONE
    bundle = export_dir(cfg)

    for name in ["chamber_long", "bsoid_clusters", "cluster_stats", "region_stats"]:
        assert (bundle / f"{name}.parquet").exists() and (bundle / f"{name}.csv").exists()
    assert (bundle / "export_manifest.json").exists() and (bundle / "data_dictionary.md").exists()
    # per-video labels copied, named by video_id
    parts = sorted(p.name for p in (bundle / "bsoid_labels").glob("*.parquet"))
    assert parts == ["m1__bsoid_labels.parquet", "m2__bsoid_labels.parquet"]

    em = json.loads((bundle / "export_manifest.json").read_text())
    assert em["join_key"] == ["video_id"]  # migrated identity
    assert em["project"] == "pt_social" and em["assay"] == "3C_SIT"
    assert set(em["factors"]) == {"treatment", "housing"}
    assert {f["name"] for f in em["design_factors"]} == {"treatment", "housing"}


def test_export_manifest_schema_matches_files_and_dictionary_complete(new_project):
    cfg, manifest = _full(new_project)
    run_stage("export", cfg, manifest)
    bundle = export_dir(cfg)
    em = json.loads((bundle / "export_manifest.json").read_text())
    dd = (bundle / "data_dictionary.md").read_text()

    for name, d in em["datasets"].items():
        if d.get("kind") == "per_video":
            for part in d["parts"]:
                assert pd.read_parquet(bundle / part["parquet"]).shape[0] == part["rows"]
        else:
            df = pd.read_parquet(bundle / d["parquet"])
            assert d["rows"] == len(df) and d["columns"] == list(df.columns)
        for col in d["columns"]:
            assert f"`{col}`" in dd  # every column documented (no undocumented column)
    assert "video_id" in dd
