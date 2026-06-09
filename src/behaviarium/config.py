"""Configuration (Phase 7): the repo is the installed tool; a PROJECT is an external folder.

``configs/core.yml`` (in the repo) holds global defaults. ``configs/projects/*.yml`` are now
TEMPLATES — ``init_project`` copies one into ``<project_dir>/project.yml``. All project paths
(manifest, outputs, per-video folders) resolve under ``<project_dir>``; the raw video
``data_path`` lives anywhere and is recorded in the project config (env-overridable).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

PROJECT_CONFIG_NAME = "project.yml"


def repo_root() -> Path:
    """Repo root, derived from this file's location (src/behaviarium/config.py)."""
    return Path(__file__).resolve().parents[2]


def default_config_dir() -> Path:
    env = os.environ.get("BEHAVIARIUM_CONFIG_DIR")
    return Path(env) if env else repo_root() / "configs"


def template_path(name: str) -> Path:
    return default_config_dir() / "projects" / f"{name}.yml"


def _abs(p: Path, base: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (base / p)


class CoreConfig(BaseSettings):
    """Global, project-agnostic defaults. ``BEHAVIARIUM_*`` env vars take precedence."""

    model_config = SettingsConfigDict(env_prefix="BEHAVIARIUM_", extra="ignore")

    recording_duration_s: float = 600.0
    video_extensions: list[str] = Field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"])

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls), file_secret_settings)


def _load_core(yaml_path: Path) -> CoreConfig:
    class _Core(CoreConfig):
        model_config = SettingsConfigDict(env_prefix="BEHAVIARIUM_", extra="ignore", yaml_file=str(yaml_path))

    return _Core()


# --- stage params (unchanged across Phase 7 except chamber/stats key on factors) ---
class RotateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    degrees: float = 0.0
    flip: str | None = None
    auto: bool = False


class BoundaryParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shape: str = "rect"
    threshold: int = 127
    blur_ksize: int = 5
    min_area_frac: float = 0.05
    max_area_frac: float = 0.95
    pixel_min: int = 0
    pixel_max: int = 255
    sample_frame: int = 0


class MaskParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    crop: bool = False
    fill_value: int = 0


class DlcFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    type: str = "median"
    windowlength: int = 5
    delegate_to_dlc: bool = False


class DlcParams(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    engine: str = "tensorflow"
    model_config_path: str | None = None
    shuffle: int = 1
    trainingsetindex: int = 0
    bodyparts: list[str] = Field(default_factory=list)
    filter: DlcFilter = Field(default_factory=DlcFilter)


class ChamberRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    shape: str = "rect"
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0
    cx: float | None = None
    cy: float | None = None
    r: float | None = None


class ChamberParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tracking_bodypart: str
    regions: list[ChamberRegion] = Field(default_factory=list)


class BsoidParams(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    engine: str = "real"
    n_clusters: int
    module: str = "bsoid_py.classify"
    model_path: str | None = None


class StatsParams(BaseModel):
    """``group_factor`` is a design-factor name; the two groups are its first two levels."""

    model_config = ConfigDict(extra="forbid")
    group_factor: str
    metric: str = "fraction"
    n_permutations: int = 1000
    alpha: float = 0.05
    seed: int = 0


# --- design matrix (Phase 7) ---
class DesignFactor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    levels: list[str]


class DesignConfig(BaseModel):
    """Ordered factors; the cells are the Cartesian product of levels (fully generic)."""

    model_config = ConfigDict(extra="forbid")
    factors: list[DesignFactor] = Field(default_factory=list)

    def factor_names(self) -> list[str]:
        return [f.name for f in self.factors]

    def n_cells(self) -> int:
        n = 1
        for f in self.factors:
            n *= len(f.levels)
        return n

    def cells(self) -> list[dict[str, str]]:
        cells: list[dict[str, str]] = [{}]
        for f in self.factors:
            cells = [{**c, f.name: lv} for c in cells for lv in f.levels]
        return cells


class ProjectConfig(BaseModel):
    """Per-project config, loaded from an external ``<project_dir>/project.yml``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    assay: str
    data_path: str  # raw video root — any layout; env-overridable via BEHAVIARIUM_DATA_PATH
    n_clusters: int
    recording_duration_s: float | None = None
    design: DesignConfig = Field(default_factory=DesignConfig)
    rotate: RotateParams = Field(default_factory=RotateParams)
    boundary: BoundaryParams = Field(default_factory=BoundaryParams)
    mask: MaskParams = Field(default_factory=MaskParams)
    dlc: DlcParams = Field(default_factory=DlcParams)
    chamber: ChamberParams | None = None
    bsoid: BsoidParams | None = None
    stats: StatsParams | None = None


def _load_project_config(yaml_path: Path) -> ProjectConfig:
    data = yaml.safe_load(yaml_path.read_text()) or {}
    return ProjectConfig.model_validate(data)


class Config(BaseModel):
    """Active project: its config + the external project_dir + global core defaults."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    core: CoreConfig
    project: ProjectConfig
    project_dir: Path

    @property
    def data_path(self) -> Path:
        env = os.environ.get("BEHAVIARIUM_DATA_PATH")
        return _abs(Path(env or self.project.data_path), self.project_dir)

    @property
    def manifest_path(self) -> Path:
        return self.project_dir / "manifest.db"

    @property
    def outputs_dir(self) -> Path:
        return self.project_dir / "outputs"

    @property
    def videos_dir(self) -> Path:
        return self.project_dir / "videos"

    @property
    def recording_duration_s(self) -> float:
        if self.project.recording_duration_s is not None:
            return self.project.recording_duration_s
        return self.core.recording_duration_s

    @property
    def video_extensions(self) -> list[str]:
        return list(self.core.video_extensions)

    @property
    def assay(self) -> str:
        return self.project.assay

    @property
    def n_clusters(self) -> int:
        return self.project.n_clusters


def load_project(project_dir: Path | str) -> Config:
    """Open an existing project folder."""
    project_dir = Path(project_dir).resolve()
    cfg_path = project_dir / PROJECT_CONFIG_NAME
    if not cfg_path.exists():
        raise FileNotFoundError(f"Not a Behaviarium project (no {PROJECT_CONFIG_NAME}): {project_dir}")
    core = _load_core(default_config_dir() / "core.yml")
    project = _load_project_config(cfg_path)
    return Config(core=core, project=project, project_dir=project_dir)


def init_project(project_dir: Path | str, data_path: Path | str, template: str = "pt_social") -> Config:
    """Scaffold a NEW external project folder from a repo template. Writes nothing into the repo."""
    project_dir = Path(project_dir).resolve()
    tmpl = template_path(template)
    if not tmpl.exists():
        raise FileNotFoundError(f"No project template named {template!r} ({tmpl})")
    if (project_dir / PROJECT_CONFIG_NAME).exists():
        raise FileExistsError(f"Project already initialized: {project_dir}")

    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "outputs").mkdir(exist_ok=True)
    (project_dir / "videos").mkdir(exist_ok=True)

    # copy the template, then record the chosen data_path inside the project config
    data = yaml.safe_load(tmpl.read_text()) or {}
    data["data_path"] = str(Path(data_path))
    (project_dir / PROJECT_CONFIG_NAME).write_text(yaml.safe_dump(data, sort_keys=False))

    cfg = load_project(project_dir)
    from .manifest import Manifest  # local import to avoid a cycle

    Manifest(cfg.manifest_path).init()
    return cfg


def resolve_dlc_model_config(cfg: Config) -> Path | None:
    env = os.environ.get("BEHAVIARIUM_DLC_MODEL_CONFIG")
    raw = env or cfg.project.dlc.model_config_path
    return Path(raw) if raw else None


def resolve_bsoid_model(cfg: Config) -> Path | None:
    env = os.environ.get("BEHAVIARIUM_BSOID_MODEL")
    raw = env or (cfg.project.bsoid.model_path if cfg.project.bsoid else None)
    return Path(raw) if raw else None
