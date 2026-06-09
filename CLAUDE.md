# CLAUDE.md

Durable context for the **Behaviarium** project. Read this first in every session.

## What this is

Behaviarium is an **orchestration / control-plane layer over EXISTING tools** (OpenCV,
DeepLabCut, B-SOiD). We are **NOT** authoring new pose-estimation or clustering methods —
we wrap the tools that already do that work.

Goal: replace a drifted collection of ~30 fragmented scripts with **idempotent,
individually re-runnable stages** coordinated by **ONE sqlite manifest** and **ONE config**.

Target assays: mouse three-chamber social interaction test (`3C_SIT`) and open-field test
(`OFT`).

## Architecture (decided)

- **Streamlit "control plane": thin UI + job manifest.** The UI never runs long compute
  inline — Streamlit reruns the script top-to-bottom on every interaction, so anything
  blocking would re-fire. DLC / B-SOiD run as **background subprocesses** tracked by the
  manifest.
- **Single sqlite manifest (WAL mode)** replaces all scattered status CSVs. **One row per
  `(video, stage)`** with status / timestamps / params.
- **Single `pydantic-settings` + YAML config** holds **EVERY** path, constant, and the
  experimental design matrix. The config is the declarative "setup file." **No hardcoded
  paths or constants anywhere in code.**
- **Stage contract:** each stage (`ingest`, `rotate`, `boundary`, `mask`, `dlc`, `chamber`,
  `bsoid`, `postprocess`, `stats`, `export`) is one **idempotent** unit with a uniform
  interface — inputs from manifest + config, outputs to known paths, status update.
  **Single-source each stage** — no forked variants.

## Project model (Phase 7 — supersedes the old data layout)

- **Data lives OUTSIDE the repo. The repo is the installed tool.** A **project is a separate
  folder**, scaffolded by `behaviarium init-project <project_dir> --data <data_path>`. The
  repo's `configs/projects/*.yml` are **TEMPLATES**, not live config; `init-project` copies one
  into `<project_dir>/project.yml`. The manifest (`<project_dir>/manifest.db`), `outputs/`, and
  per-video folders all resolve under `<project_dir>`. **Nothing project-specific is written
  inside the repo.**
- **Raw video sits in ANY folder layout** under `data_path` and stays put by default. `ingest`
  scans recursively (flat or nested).
- **Primary identity = stable `video_id`** (filename slug, deduped on collision), with
  `source_path` (original) + `current_path` recorded. **`video_id` is the join key everywhere**
  — in the manifest and ALL tidy outputs. (Replaces the old `(Type, Class, Filename)` triple
  and the Class-string parser.)
- **Per-video folder** `<project_dir>/videos/<video_id>/` always exists — home for sidecars and
  that video's stage outputs/metadata that don't belong in the DB.
- **Design factors + tagging** replace folder-derived Type/Class. `project.yml` declares
  `design.factors` (ordered `{name, levels}`; cells = product of level counts). The user tags
  each video into a cell in the UI; the tag becomes the factor columns carried alongside
  `video_id` downstream. **Untagged or excluded videos are skipped by the runner.**
- **Reorg is an explicit user action** (never automatic): `copy` (default) | `move` | `symlink`
  the source into its per-video folder; idempotent, never overwrites, verifies before updating
  `current_path`.

## Platform strategy (v1)

- **PRODUCTION = Windows + TensorFlow DLC stack.** DLC 2.3.10 / TF 2.10, real TF model
  `double_hit_social-Javid-2022-03-21`, `shuffle 1`, `trainingsetindex 0`. **Do NOT
  introduce new DLC troubleshooting** — Windows is what gets used for real runs.
- **Bonsai is REPLACED entirely by portable OpenCV** (rotate, mask, ROI) so non-DLC stages
  run cross-platform (Mac + Windows).
- **Mac (Apple Silicon) = dev + proof-of-concept only.** TF DLC on Mac is ~0.5 fps and
  acceptable only to exercise the flow. If the install is too brittle, use a **synthetic
  DLC-output stub**. Real batch DLC stays on Windows.
- **DLC stage is engine-aware but v1 hardcodes `engine="tensorflow"`.** A v2 PyTorch path is
  designed-for but **NOT built now**.
- **Dev loop:** code on Mac → push to GitHub → pull/run on Windows.
- **Downstream R / RMarkdown is in scope:** the pipeline emits tidy long-format
  **Parquet + CSV** that R consumes as a **separate manual step**. **Python never calls R.**

## Conventions to follow

- **src layout.** `pathlib` everywhere; never string-concatenate paths. **No absolute paths
  in code** — `data_root` comes from config, env-overridable per machine.
- Reference DLC bodyparts **BY NAME** via the multiindex header (`header=[0,1,2]`), **NOT**
  by positional columns like `x.1` / `x.11` (the old code's brittleness we are fixing).
- Cluster count, fps, camera pixel ranges, and the experimental design matrix come from
  **config**, never hardcoded. Express clusters as `range(cfg.n_clusters)`.
- Keep code comments minimal; do not write large README blocks inside scripts. **Do not
  invent facts or APIs** — if unsure about a library's interface, check it rather than
  guessing.

## Open scientific decision points

**DO NOT silently resolve these — flag to me when a phase hits them.**

1. **`_dlc_filtered.csv` is currently NOT filtered** (no `filterpredictions` call). Decide
   per config: actually filter (median / arima) and name honestly, **or** rename to `_raw`.
2. **B-SOiD label reconstruction:** the correct method is the **frameshift + `flatten('F')`
   interleave** — predict `floor(fps/10)` frame-offset streams, then interleave them. (Phase 5
   correction: `floor`, not `round`, to match the authoritative YttriLab B-SOiD source.) The
   "block-repeat each label 8x" version is **WRONG** and has been failing — **do not reuse
   it.** *Resolved & implemented in `bsoid_reconstruct.py`.*
3. **Framerate:** three conflicting values exist (B-SOiD `84`, chamber analysis `frames/600`,
   kinematics `30`). **Single source of truth = corrected fps = `actual_frame_count / 600`**,
   stored per-video in the manifest, propagated to **ALL** fps-dependent math. Flag the
   reconciliation.
4. **Network `P_Value = 1 - normalized_weight` is NOT a real p-value.** Rename to a
   **display-weight** column. Reserve "p-value" / "significance" for the real **Wasserstein
   permutation + Benjamini-Hochberg** results only.
5. **Study-specific items live in per-project config** (the project's `project.yml`, copied
   from a repo template), **not** core code: the design matrix as `design.factors`
   (`treatment: Sal-N/Sal-Hx/LPS-N/DH` × `housing: PT/EE`), the 14 clusters, and the
   6-quadrant chamber rule. *(Phase 7: the `Class`-string parser is GONE — factor columns now
   come from UI **tagging** of each `video_id` into a design cell.)*

## Build process

We build **PHASE BY PHASE**. Each phase: build, then **stop** so I can verify the milestone
on the correct machine (Mac vs Windows) before the next phase. **Do not run ahead into later
phases.**
