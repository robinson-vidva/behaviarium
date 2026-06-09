"""Stage registry — stages are pluggable by name, optionally specialised per assay.

Adding a new assay means: add a per-project config and (if a stage needs assay-specific
behaviour) register a stage variant for that assay. No core changes required.
"""

from __future__ import annotations

from .stage import Stage

# (stage_name, assay | None) -> Stage subclass. assay=None is the generic/default variant.
_REGISTRY: dict[tuple[str, str | None], type[Stage]] = {}


def register(name: str, assay: str | None = None):
    """Class decorator that registers a stage under ``name`` (optionally for one ``assay``)."""

    def deco(cls: type[Stage]) -> type[Stage]:
        cls.name = name
        cls.assay = assay
        _REGISTRY[(name, assay)] = cls
        return cls

    return deco


def get_stage(name: str, assay: str | None = None) -> type[Stage]:
    """Resolve a stage: prefer the assay-specific variant, fall back to the generic one."""
    if (name, assay) in _REGISTRY:
        return _REGISTRY[(name, assay)]
    if (name, None) in _REGISTRY:
        return _REGISTRY[(name, None)]
    raise KeyError(f"No stage registered for name={name!r} assay={assay!r}")


def registered_stages() -> list[tuple[str, str | None]]:
    return sorted(_REGISTRY.keys(), key=lambda k: (k[0], k[1] or ""))
