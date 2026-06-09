"""Configuration: one core (assay-agnostic) layer + one per-project (assay-specific) layer.

The loader merges ``configs/core.yml`` with ``configs/projects/<project>.yml``. No paths or
constants are hardcoded elsewhere in the codebase; everything machine-specific (notably
``data_root``) is overridable via ``BEHAVIARIUM_*`` environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


def repo_root() -> Path:
    """Repo root, derived from this file's location (src/behaviarium/config.py)."""
    return Path(__file__).resolve().parents[2]


def default_config_dir() -> Path:
    env = os.environ.get("BEHAVIARIUM_CONFIG_DIR")
    return Path(env) if env else repo_root() / "configs"


def _abs(p: Path, base: Path) -> Path:
    """Resolve ``p`` against ``base`` when relative; absolute paths pass through."""
    p = Path(p)
    return p if p.is_absolute() else (base / p)


class CoreConfig(BaseSettings):
    """Assay-agnostic settings. Loaded from YAML; ``BEHAVIARIUM_*`` env vars take precedence."""

    model_config = SettingsConfigDict(env_prefix="BEHAVIARIUM_", extra="ignore")

    data_root: Path = Path("data")
    output_root: Path = Path("outputs")
    manifest_path: Path = Path("manifest.db")  # relative -> resolved under output_root
    # Single source of truth for fps: corrected fps = frame_count / recording_duration_s.
    recording_duration_s: float = 600.0
    video_extensions: list[str] = Field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"])

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Precedence: explicit init > env > YAML file. (env overrides the YAML on disk.)
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


def _load_core(yaml_path: Path) -> CoreConfig:
    # Subclass to inject the resolved YAML path into model_config for the YAML source.
    class _Core(CoreConfig):
        model_config = SettingsConfigDict(
            env_prefix="BEHAVIARIUM_", extra="ignore", yaml_file=str(yaml_path)
        )

    return _Core()


class RotateParams(BaseModel):
    """Orientation correction. Generic across assays; values come from per-project config."""

    model_config = ConfigDict(extra="forbid")

    degrees: float = 0.0  # clockwise; 90/180/270 are exact, other angles expand the canvas
    flip: str | None = None  # "horizontal" | "vertical" | "both" | None
    auto: bool = False  # hook for future auto-detection — NOT implemented this phase


class BoundaryParams(BaseModel):
    """Arena-ROI auto-detection params. Generic geometry (rect or circle) fits every assay."""

    model_config = ConfigDict(extra="forbid")

    shape: str = "rect"  # shape hint: "rect" | "circle"
    threshold: int = 127  # binary threshold separating bright arena from background
    blur_ksize: int = 5  # Gaussian blur kernel (odd; <=1 disables)
    min_area_frac: float = 0.05  # contour area bounds as a fraction of the frame
    max_area_frac: float = 0.95
    pixel_min: int = 0  # optional intensity gate applied on top of the threshold
    pixel_max: int = 255
    sample_frame: int = 0  # frame index sampled for detection + preview overlay


class MaskParams(BaseModel):
    """How the approved ROI is applied to produce the DLC-ready video."""

    model_config = ConfigDict(extra="forbid")

    crop: bool = False  # also crop to the ROI bounding box
    fill_value: int = 0  # value written outside the ROI


class DlcFilter(BaseModel):
    """Decision #1: filtering is honest and config-driven. Disabled => _raw, no filter step.
    Enabled => actually filter and name _filtered. Never emit _filtered that wasn't filtered."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    type: str = "median"  # "median" (backend-independent) | "arima" (optional)
    windowlength: int = 5
    # false: our pandas rolling median (backend-independent, testable on Mac).
    # true: delegate to deeplabcut.filterpredictions — DLC-exact, tensorflow backend ONLY.
    delegate_to_dlc: bool = False


class DlcParams(BaseModel):
    """Pose-estimation params. Engine-aware, one interface; v1 real engine is tensorflow.
    The stub is first-class so Mac dev never needs TF/DLC."""

    # ``protected_namespaces=()`` so the ``model_config_path`` field is allowed (pydantic v2
    # otherwise reserves the ``model_`` prefix).
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    engine: str = "tensorflow"  # "tensorflow" (real, lazy import) | "stub" (synthetic)
    model_config_path: str | None = None  # DLC project config.yaml; env-overridable per machine
    shuffle: int = 1
    trainingsetindex: int = 0
    bodyparts: list[str] = Field(default_factory=list)  # used by the stub backend
    filter: DlcFilter = Field(default_factory=DlcFilter)


class ChamberRegion(BaseModel):
    """One named spatial region, in fractions [0,1] of the boundary-ROI bbox. Generic geometry:
    rect (x, y, w, h) or circle (cx, cy, r). The assay decides the regions; core just maps
    points to whichever regions the config declares."""

    model_config = ConfigDict(extra="forbid")

    name: str
    shape: str = "rect"  # "rect" | "circle"
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0
    cx: float | None = None
    cy: float | None = None
    r: float | None = None


class ChamberParams(BaseModel):
    """Region scheme for occupancy. ``regions`` are tried in order; first match wins."""

    model_config = ConfigDict(extra="forbid")

    tracking_bodypart: str
    regions: list[ChamberRegion] = Field(default_factory=list)


class ClassParser(BaseModel):
    """Study-specific parser turning the ``Class`` string into factor columns. Core is generic:
    ``noop`` yields no factors; ``regex`` extracts named groups. No design-matrix literals here —
    the pattern (and thus the allowed values) live in per-project config."""

    model_config = ConfigDict(extra="forbid")

    kind: str = "noop"  # "noop" | "regex"
    pattern: str | None = None  # for kind=regex: named groups become factor columns


class BsoidParams(BaseModel):
    """B-SOiD behavioral clustering. Engine-aware like dlc; the stub exercises the full
    frameshift reconstruction without heavy deps."""

    # ``protected_namespaces=()`` so the ``model_path`` field name is allowed.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    engine: str = "real"  # "real" (lazy import) | "stub" (synthetic per-10Hz predictions)
    n_clusters: int  # B-SOiD model cluster count (range(n_clusters) for the summary)
    module: str = "bsoid_py.classify"  # real backend import path — CONFIRM per B-SOiD fork
    model_path: str | None = None  # trained B-SOiD model bundle; env-overridable per machine


class StatsParams(BaseModel):
    """Real significance path. ``group_factor`` is a column in the tidy long output (a parsed
    Class factor, or Type/Class) defining the two groups to compare — config-driven, so the
    comparison differs per project with no core change."""

    model_config = ConfigDict(extra="forbid")

    group_factor: str  # column that defines the groups (e.g. a design-matrix factor)
    metric: str = "fraction"  # per-video value compared per cluster/region
    n_permutations: int = 1000
    alpha: float = 0.05
    seed: int = 0  # reproducible permutations


class ProjectConfig(BaseModel):
    """Per-project, assay-specific config. ``assay`` is a free-form string (not a closed enum),
    so new assays are added via config + registered stage variants, with no core changes."""

    model_config = ConfigDict(extra="forbid")

    name: str
    assay: str  # e.g. "3C_SIT", "OFT" — documented in configs, not enforced in core
    n_clusters: int
    design_matrix: dict[str, list[str]] = Field(default_factory=dict)
    # Recording length override (s). None => use core's value. corrected_fps = frames / this.
    recording_duration_s: float | None = None
    # OpenCV preprocessing params (Phase 1). Defaults keep these optional per project.
    rotate: RotateParams = Field(default_factory=RotateParams)
    boundary: BoundaryParams = Field(default_factory=BoundaryParams)
    mask: MaskParams = Field(default_factory=MaskParams)
    # DLC pose estimation (Phase 2).
    dlc: DlcParams = Field(default_factory=DlcParams)
    # Chamber occupancy + Class parsing (Phase 3).
    chamber: ChamberParams | None = None
    class_parser: ClassParser = Field(default_factory=ClassParser)
    # B-SOiD behavioral clustering (Phase 4).
    bsoid: BsoidParams | None = None
    # Aggregation + real significance (Phase 5).
    stats: StatsParams | None = None


def _load_project(yaml_path: Path) -> ProjectConfig:
    data = yaml.safe_load(yaml_path.read_text()) or {}
    return ProjectConfig.model_validate(data)


class Config(BaseModel):
    """Merged view of core + one project, with resolved (absolute) paths."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    core: CoreConfig
    project: ProjectConfig
    config_dir: Path
    root: Path

    @property
    def data_root(self) -> Path:
        return _abs(self.core.data_root, self.root)

    @property
    def output_root(self) -> Path:
        return _abs(self.core.output_root, self.root)

    @property
    def manifest_path(self) -> Path:
        return _abs(self.core.manifest_path, self.output_root)

    @property
    def recording_duration_s(self) -> float:
        # Per-project override wins over core (which itself is env-overridable per machine).
        if self.project.recording_duration_s is not None:
            return self.project.recording_duration_s
        return self.core.recording_duration_s

    @property
    def video_extensions(self) -> list[str]:
        return list(self.core.video_extensions)

    # Convenience passthroughs to the active project.
    @property
    def assay(self) -> str:
        return self.project.assay

    @property
    def n_clusters(self) -> int:
        return self.project.n_clusters


def load_config(project: str, config_dir: Path | None = None) -> Config:
    """Select and merge core + one project config."""
    config_dir = Path(config_dir) if config_dir else default_config_dir()
    core = _load_core(config_dir / "core.yml")
    proj = _load_project(config_dir / "projects" / f"{project}.yml")
    return Config(core=core, project=proj, config_dir=config_dir, root=repo_root())


def resolve_dlc_model_config(cfg: Config) -> Path | None:
    """DLC project config.yaml path for the real backend. ``BEHAVIARIUM_DLC_MODEL_CONFIG`` env
    overrides the per-project value (the model lives at a machine-specific path on Windows)."""
    env = os.environ.get("BEHAVIARIUM_DLC_MODEL_CONFIG")
    raw = env or cfg.project.dlc.model_config_path
    return Path(raw) if raw else None


def resolve_bsoid_model(cfg: Config) -> Path | None:
    """Trained B-SOiD model path for the real backend. ``BEHAVIARIUM_BSOID_MODEL`` env overrides
    the per-project value (machine-specific path on Windows)."""
    env = os.environ.get("BEHAVIARIUM_BSOID_MODEL")
    raw = env or (cfg.project.bsoid.model_path if cfg.project.bsoid else None)
    return Path(raw) if raw else None
