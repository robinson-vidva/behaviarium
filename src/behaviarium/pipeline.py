"""Canonical stage order for display and batch runs. Not a DAG engine — just an ordering."""

STAGE_ORDER = [
    "ingest", "rotate", "boundary", "mask", "dlc", "chamber", "bsoid",
    "postprocess", "stats", "export",
]

# Project-level (aggregate) stages — run once across videos, not per-video.
PROJECT_STAGES = ["postprocess", "stats", "export"]

# THE one place that defines which stages need each video's design-cell assignment.
# Untagged videos are excluded from THESE (the grouping stages) only; they still run every
# per-video prep/analysis stage (rotate..bsoid). Adjust here to change the policy.
TAG_REQUIRED_STAGES = ["postprocess", "stats", "export"]

