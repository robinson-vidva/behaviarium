"""Guard: the old (Type, Class, Filename) join key must NOT appear in any output (Phase 7)."""

import json

import pandas as pd

from behaviarium.paths import (
    bsoid_clusters_parquet,
    chamber_parquet,
    export_dir,
    postprocess_bsoid_long,
)
from behaviarium.runner import run_stage

from .conftest import run_chain, tag_round_robin

_OLD = {"Type", "Class", "Filename"}


def test_no_old_join_key_anywhere(new_project):
    cfg, manifest = new_project(template="pt_social", specs=[("m1.avi", 80), ("m2.avi", 80)])
    tag_round_robin(cfg, manifest)
    for v in manifest.list_videos():
        run_chain(cfg, manifest, v["video_id"])
    for s in ("postprocess", "stats", "export"):
        run_stage(s, cfg, manifest)

    # per-video + aggregate outputs use video_id, never the old triple
    for path in [chamber_parquet(cfg, "m1"), bsoid_clusters_parquet(cfg, "m1"),
                 postprocess_bsoid_long(cfg, ".parquet")]:
        cols = set(pd.read_parquet(path).columns)
        assert not (_OLD & cols)
        assert "video_id" in cols

    em = json.loads((export_dir(cfg) / "export_manifest.json").read_text())
    assert em["join_key"] == ["video_id"]
    for d in em["datasets"].values():
        assert not (_OLD & set(d["columns"]))

    # the codebase no longer ships a Class parser module
    import importlib

    import pytest

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("behaviarium.class_parser")
