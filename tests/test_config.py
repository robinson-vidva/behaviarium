from behaviarium.config import load_config


def test_loads_and_merges_core_and_project():
    cfg = load_config("pt_social")
    # project (assay-specific) layer
    assert cfg.assay == "3C_SIT"
    assert cfg.n_clusters == 14
    assert cfg.project.design_matrix["treatment"] == ["Sal-N", "Sal-Hx", "LPS-N", "DH"]
    # core (assay-agnostic) layer
    assert cfg.recording_duration_s == 600.0
    assert ".avi" in cfg.video_extensions
    assert cfg.data_root.is_absolute()


def test_preprocessing_params_load_3c():
    cfg = load_config("pt_social")
    assert cfg.project.rotate.degrees == 0
    assert cfg.project.boundary.shape == "rect"
    assert cfg.project.mask.crop is True


def test_dlc_params_load():
    c3 = load_config("pt_social")
    assert c3.project.dlc.engine == "tensorflow"
    assert c3.project.dlc.shuffle == 1 and c3.project.dlc.trainingsetindex == 0
    assert c3.project.dlc.filter.enabled is False  # decision #1 default
    assert c3.project.dlc.filter.delegate_to_dlc is False

    oft = load_config("oft_demo")
    assert oft.project.dlc.engine == "stub"  # stub is first-class, selected purely via config
    assert oft.project.dlc.bodyparts == ["nose", "center", "tailbase"]
    assert oft.project.dlc.filter.enabled is True


def test_dlc_model_config_env_override(monkeypatch):
    from behaviarium.config import resolve_dlc_model_config

    cfg = load_config("pt_social")
    monkeypatch.setenv("BEHAVIARIUM_DLC_MODEL_CONFIG", "/tmp/custom/config.yaml")
    assert str(resolve_dlc_model_config(cfg)) == "/tmp/custom/config.yaml"


def test_second_assay_is_just_config():
    """Adding OFT required only a config file — same core, different assay dimension."""
    cfg = load_config("oft_demo")
    assert cfg.assay == "OFT"
    assert cfg.n_clusters == 10
    # preprocessing differs from 3C_SIT purely via config
    assert cfg.project.boundary.shape == "circle"
    assert cfg.project.rotate.degrees == 90
    assert cfg.project.rotate.flip == "horizontal"
    assert cfg.project.mask.crop is False


def test_chamber_and_class_parser_load():
    c3 = load_config("pt_social")
    assert c3.project.chamber.tracking_bodypart == "spine1"
    assert [r.name for r in c3.project.chamber.regions] == [
        "left_near", "left_far", "center_near", "center_far", "right_near", "right_far",
    ]
    assert c3.project.class_parser.kind == "regex"

    oft = load_config("oft_demo")
    assert [r.name for r in oft.project.chamber.regions] == ["center", "periphery"]
    assert oft.project.class_parser.kind == "noop"


def test_bsoid_params_load():
    c3 = load_config("pt_social")
    assert c3.project.bsoid.engine == "real"
    assert c3.project.bsoid.n_clusters == 14  # the 14 clusters
    oft = load_config("oft_demo")
    assert oft.project.bsoid.engine == "stub"
    assert oft.project.bsoid.n_clusters == 6  # different per project -> genericity


def test_stats_params_load():
    c3 = load_config("pt_social")
    assert c3.project.stats.group_factor == "housing"  # a design-matrix factor
    assert c3.project.stats.n_permutations == 1000 and c3.project.stats.alpha == 0.05
    oft = load_config("oft_demo")
    assert oft.project.stats.group_factor == "Type"  # different comparison -> genericity


def test_env_overrides_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv("BEHAVIARIUM_DATA_ROOT", str(tmp_path))
    cfg = load_config("pt_social")
    assert cfg.data_root == tmp_path
