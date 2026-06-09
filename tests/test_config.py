from behaviarium.config import (
    DesignConfig,
    DesignFactor,
    init_project,
    load_project,
)


def test_pt_social_template_loads_with_design():
    # load the repo template directly (as a project would after init)
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    cfg = init_project(d / "p", d / "raw", template="pt_social")
    assert cfg.assay == "3C_SIT"
    assert cfg.n_clusters == 14
    names = cfg.project.design.factor_names()
    assert names == ["treatment", "housing"]
    assert cfg.project.design.n_cells() == 4 * 2  # product of level counts
    assert cfg.project.chamber.tracking_bodypart == "spine1"
    assert cfg.project.stats.group_factor == "housing"


def test_oft_template_is_just_config():
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    cfg = init_project(d / "p", d / "raw", template="oft_demo")
    assert cfg.assay == "OFT"
    assert cfg.project.design.factor_names() == ["group"]
    assert cfg.project.design.n_cells() == 2
    assert cfg.project.boundary.shape == "circle"
    assert cfg.project.stats.group_factor == "group"


def test_design_of_arbitrary_shape_cells_is_product():
    d = DesignConfig(factors=[
        DesignFactor(name="a", levels=["x", "y", "z"]),
        DesignFactor(name="b", levels=["1", "2"]),
        DesignFactor(name="c", levels=["p", "q"]),
    ])
    assert d.n_cells() == 3 * 2 * 2
    cells = d.cells()
    assert len(cells) == 12
    assert {"a": "x", "b": "1", "c": "p"} in cells
    assert all(set(c) == {"a", "b", "c"} for c in cells)


def test_recording_duration_override(new_project):
    cfg, manifest = new_project(specs=[("v.avi", 90)])
    # env override (BEHAVIARIUM_RECORDING_DURATION_S=1 set by the fixture) -> fps == frames/1
    assert manifest.get_video("v")["fps"] == 90 / 1
    assert cfg.recording_duration_s == 1.0
