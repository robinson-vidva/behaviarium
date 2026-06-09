"""Importing this package registers all built-in stages with the registry."""

from . import (  # noqa: F401  (register on import)
    boundary,
    bsoid,
    chamber,
    dlc,
    export,
    ingest,
    mask,
    postprocess,
    rotate,
    stats,
)

__all__ = [
    "ingest", "rotate", "boundary", "mask", "dlc", "chamber", "bsoid",
    "postprocess", "stats", "export",
]
