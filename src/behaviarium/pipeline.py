"""Canonical stage order for display and batch runs. Not a DAG engine — just an ordering."""

STAGE_ORDER = [
    "ingest", "rotate", "boundary", "mask", "dlc", "chamber", "bsoid",
    "postprocess", "stats", "export",
]

# Project-level (aggregate) stages — run once across videos, not per-video.
PROJECT_STAGES = ["postprocess", "stats", "export"]
