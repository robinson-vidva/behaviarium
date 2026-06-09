import pytest

from behaviarium.config import init_project, load_project, repo_root


def test_init_project_creates_external_dir_and_writes_nothing_into_repo(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    proj = tmp_path / "myproj"

    repo_before = {p for p in (repo_root() / "configs").rglob("*")}
    cfg = init_project(proj, data, template="pt_social")

    # scaffolded outside the repo
    assert (proj / "project.yml").exists()
    assert (proj / "manifest.db").exists()
    assert (proj / "outputs").is_dir()
    assert (proj / "videos").is_dir()
    assert proj.resolve() not in repo_root().parents and repo_root() not in proj.resolve().parents \
        or not str(proj.resolve()).startswith(str(repo_root()))

    # nothing new written into the repo's configs/
    repo_after = {p for p in (repo_root() / "configs").rglob("*")}
    assert repo_before == repo_after

    # data_path recorded, resolves to the external path
    assert cfg.data_path == data.resolve()
    assert cfg.manifest_path == proj.resolve() / "manifest.db"


def test_load_project_roundtrip_and_rejects_non_project(tmp_path):
    init_project(tmp_path / "p", tmp_path / "raw", template="oft_demo")
    cfg = load_project(tmp_path / "p")
    assert cfg.project.name == "oft_demo" and cfg.assay == "OFT"

    with pytest.raises(FileNotFoundError):
        load_project(tmp_path / "not_a_project")


def test_init_project_refuses_to_reinitialize(tmp_path):
    init_project(tmp_path / "p", tmp_path / "raw")
    with pytest.raises(FileExistsError):
        init_project(tmp_path / "p", tmp_path / "raw")


def test_data_path_env_override(tmp_path, monkeypatch):
    cfg = init_project(tmp_path / "p", tmp_path / "raw")
    other = tmp_path / "elsewhere"
    monkeypatch.setenv("BEHAVIARIUM_DATA_PATH", str(other))
    cfg2 = load_project(tmp_path / "p")
    assert cfg2.data_path == other
