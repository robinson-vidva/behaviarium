import pandas as pd
import pytest

from behaviarium.manifest import Status
from behaviarium.paths import postprocess_bsoid_long, stats_bsoid_table, stats_chamber_table
from behaviarium.runner import run_stage

from .conftest import run_chain, tag_round_robin


def _seed_two_groups(new_project, template="pt_social"):
    cfg, manifest = new_project(template=template, specs=[("m1.avi", 80), ("m2.avi", 80)])
    tag_round_robin(cfg, manifest)  # m1 -> first cell, m2 -> second cell (different group level)
    for v in manifest.list_videos():
        run_chain(cfg, manifest, v["video_id"])
    return cfg, manifest


def test_postprocess_and_stats_keyed_by_video_id(new_project):
    cfg, manifest = _seed_two_groups(new_project)
    assert run_stage("postprocess", cfg, manifest) == Status.DONE
    assert run_stage("stats", cfg, manifest) == Status.DONE

    blong = pd.read_parquet(postprocess_bsoid_long(cfg, ".parquet"))
    assert "video_id" in blong.columns and "housing" in blong.columns
    assert not ({"Type", "Class", "Filename"} & set(blong.columns))
    assert set(blong["video_id"]) == {"m1", "m2"}

    sdf = pd.read_csv(stats_bsoid_table(cfg, ".csv"))
    assert {"cluster", "wasserstein_stat", "p_value", "q_value", "significant"} <= set(sdf.columns)
    # groups come from config group_factor=housing -> PT vs EE
    assert {sdf.iloc[0]["group_a"], sdf.iloc[0]["group_b"]} == {"EE", "PT"}
    assert "p_value" not in blong.columns  # decision #4: only the stats table has p_value

    rdf = pd.read_csv(stats_chamber_table(cfg, ".csv"))
    assert "region" in rdf.columns and "p_value" in rdf.columns


def test_postprocess_requires_upstream(new_project):
    cfg, manifest = new_project(specs=[("m1.avi", 30)])
    manifest.set_tag("m1", {"treatment": "Sal-N", "housing": "PT"})  # processable but no outputs yet
    with pytest.raises(RuntimeError, match="run bsoid first"):
        run_stage("postprocess", cfg, manifest)


def test_postprocess_idempotent(new_project):
    cfg, manifest = _seed_two_groups(new_project)
    assert run_stage("postprocess", cfg, manifest) == Status.DONE
    assert run_stage("postprocess", cfg, manifest) == Status.DONE
