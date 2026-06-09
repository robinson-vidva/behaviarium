"""Behaviarium control plane (thin) — Phase 7 project model.

Open an EXTERNAL project folder by path. Tag videos into design cells, review boundaries,
run stages. The UI never runs long compute inline — everything is a background subprocess
(``python -m behaviarium ...``) tracked by the manifest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from behaviarium import stages  # noqa: F401  registers stages
from behaviarium.config import load_project
from behaviarium.manifest import PROJECT_ID, Approval, Manifest, Status
from behaviarium.paths import (
    boundary_preview,
    bsoid_clusters_csv,
    chamber_csv,
    export_dir,
    stats_bsoid_table,
    stats_chamber_table,
)
from behaviarium.pipeline import PROJECT_STAGES, STAGE_ORDER
from behaviarium.runner import is_processable

st.set_page_config(page_title="Behaviarium", layout="wide")
st.title("Behaviarium — control plane")

PER_VIDEO_STAGES = ["rotate", "boundary", "mask", "dlc", "chamber", "bsoid"]

# --- open a project by path -------------------------------------------------------------
default_dir = os.environ.get("BEHAVIARIUM_PROJECT", "")
project_path = st.sidebar.text_input("Project folder", value=default_dir,
                                     help="Path to a project created with `behaviarium init-project`")
if not project_path:
    st.info("Enter a project folder path in the sidebar (create one with `behaviarium init-project`).")
    st.stop()
try:
    cfg = load_project(project_path)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not open project: {exc}")
    st.stop()

manifest = Manifest(cfg.manifest_path)
factor_names = cfg.project.design.factor_names()


def spawn(*cli_args: str) -> None:
    subprocess.Popen([sys.executable, "-m", "behaviarium", *cli_args], env=os.environ.copy())


st.sidebar.write(f"**Project:** `{cfg.project.name}`  ·  **assay:** `{cfg.assay}`")
st.sidebar.write(f"**data_path:** `{cfg.data_path}`")
st.sidebar.caption(f"project_dir: {cfg.project_dir}")

if st.sidebar.button("Run ingest", help="Discover videos under data_path (background)"):
    spawn("ingest", "--project", str(cfg.project_dir))
    st.sidebar.info("Ingest launched. Click Refresh.")

batch_stage = st.sidebar.selectbox("Run stage (all videos)", PER_VIDEO_STAGES, key="batch")
if st.sidebar.button(f"Run {batch_stage} (all)"):
    spawn("run-all", "--project", str(cfg.project_dir), "--stage", batch_stage)
    st.sidebar.info(f"{batch_stage} launched. Click Refresh.")

proj_stage = st.sidebar.selectbox("Run project stage", PROJECT_STAGES, key="proj")
if st.sidebar.button(f"Run {proj_stage} (project)"):
    spawn("run-project", "--project", str(cfg.project_dir), "--stage", proj_stage)
    st.sidebar.info(f"{proj_stage} launched. Click Refresh.")

if st.sidebar.button("Refresh"):
    st.rerun()

if not Path(cfg.manifest_path).exists():
    st.info("No manifest yet. Click **Run ingest**.")
    st.stop()
videos = manifest.list_videos()
if not videos:
    st.info("Manifest is empty. Click **Run ingest** to discover videos under data_path.")
    st.stop()

# --- design coverage --------------------------------------------------------------------
st.subheader(f"Design coverage — {len(cfg.project.design.factors)} factor(s), {cfg.project.design.n_cells()} cells")
if factor_names:
    cell_counts = []
    for cell in cfg.project.design.cells():
        n = sum(1 for v in videos if v.get("tag") and all(v["tag"].get(k) == cell[k] for k in cell))
        cell_counts.append({**cell, "videos": n})
    untagged = sum(1 for v in videos if not is_processable(cfg, manifest, v["video_id"]) and v.get("include", 1))
    st.dataframe(pd.DataFrame(cell_counts), use_container_width=True, hide_index=True)
    st.caption(
        f"Untagged: **{untagged}** of {len(videos)} videos — these still run the per-video "
        "stages, but are excluded from postprocess/stats/export until tagged."
    )
else:
    st.caption("This project declares no design factors.")

# --- videos x stage status --------------------------------------------------------------
stage_names = [s for s in STAGE_ORDER if s not in PROJECT_STAGES and s != "ingest"]
rows = []
for v in videos:
    vid = v["video_id"]
    tag = v.get("tag") or {}
    row = {
        "video_id": vid,
        "filename": v["filename"],
        "tag": ", ".join(f"{k}={tag[k]}" for k in factor_names if k in tag) or "—",
        "include": bool(v["include"]),
        "fps": v["fps"],
    }
    for name in stage_names:
        status = manifest.get_status(vid, name) or "—"
        if name == "boundary":
            ap = manifest.get_approval(vid, name)
            status = f"{status} ({ap})" if ap else status
        row[name] = status
    rows.append(row)
st.subheader(f"Videos × stage status ({len(rows)})")
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- per-video control ------------------------------------------------------------------
st.divider()
st.subheader("Per-video control")
labels = [f"{v['video_id']}  ({v['filename']})" for v in videos]
choice = st.selectbox("Video", range(len(videos)), format_func=lambda i: labels[i])
v = videos[choice]
vid = v["video_id"]
tag = v.get("tag") or {}

col_tag, col_act = st.columns(2)
with col_tag:
    st.markdown("**Tag into a design cell**")
    new_tag = {}
    for f in cfg.project.design.factors:
        opts = ["(unset)"] + f.levels
        cur = tag.get(f.name, "(unset)")
        sel = st.selectbox(f.name, opts, index=opts.index(cur) if cur in opts else 0, key=f"tag_{f.name}")
        if sel != "(unset)":
            new_tag[f.name] = sel
    if st.button("Save tag"):
        manifest.set_tag(vid, new_tag or None)
        st.rerun()

    include = st.checkbox("Include this video", value=bool(v["include"]))
    if include != bool(v["include"]):
        manifest.set_include(vid, include)
        st.rerun()

with col_act:
    st.markdown("**Reorg** (move source into the per-video folder)")
    st.caption(f"current: `{v['current_path']}`")
    mode = st.radio("Mode", ["copy", "move", "symlink"], horizontal=True)
    if st.button(f"Reorg ({mode})"):
        spawn("reorg", "--project", str(cfg.project_dir), "--video-id", vid, "--mode", mode)
        st.info("Reorg launched. Click Refresh.")

    st.markdown("**Re-run a stage**")
    rerun_stage = st.selectbox("Stage", PER_VIDEO_STAGES, key="rerun")
    if st.button(f"Re-run {rerun_stage}"):
        manifest.set_status(vid, rerun_stage, Status.PENDING)
        spawn("run", "--project", str(cfg.project_dir), "--stage", rerun_stage, "--video-id", vid)
        st.info(f"Re-running {rerun_stage}. Click Refresh.")

# boundary review
st.markdown("**Boundary review**")
brow = manifest.get_row(vid, "boundary")
if not brow:
    st.caption("No boundary result yet.")
else:
    st.caption(f"approval: `{brow.get('approval')}`  ·  status: `{brow.get('status')}`")
    preview = boundary_preview(cfg, vid)
    if preview.exists():
        st.image(str(preview), use_container_width=True)
    c1, c2 = st.columns(2)
    if c1.button("Approve"):
        manifest.set_approval(vid, "boundary", Approval.APPROVED)
        st.rerun()
    if c2.button("Reject"):
        manifest.set_approval(vid, "boundary", Approval.REJECTED)
        st.rerun()

# per-video summaries
for label, path, idx in [("Chamber occupancy", chamber_csv(cfg, vid), "region"),
                         ("B-SOiD cluster usage", bsoid_clusters_csv(cfg, vid), "cluster")]:
    if path.exists():
        st.markdown(f"**{label}**")
        df = pd.read_csv(path)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index(idx)["fraction"])

# --- project-level: stats + export ------------------------------------------------------
st.divider()
st.header("Project-level: aggregate + significance + export")
pcols = st.columns(len(PROJECT_STAGES))
for col, pstage in zip(pcols, PROJECT_STAGES):
    col.metric(pstage, manifest.get_status(PROJECT_ID, pstage) or "—")
    if col.button(f"Re-run {pstage}", key=f"rp_{pstage}"):
        manifest.set_status(PROJECT_ID, pstage, Status.PENDING)
        spawn("run-project", "--project", str(cfg.project_dir), "--stage", pstage)
        col.info(f"Re-running {pstage}. Click Refresh.")

for label, path in [("B-SOiD cluster stats", stats_bsoid_table(cfg, ".csv")),
                    ("Chamber region stats", stats_chamber_table(cfg, ".csv"))]:
    if path.exists():
        st.subheader(label)
        st.dataframe(pd.read_csv(path), use_container_width=True, hide_index=True)

manifest_json = export_dir(cfg) / "export_manifest.json"
if manifest_json.exists():
    em = json.loads(manifest_json.read_text())
    st.subheader("Export bundle")
    st.caption(f"`{export_dir(cfg)}`  ·  join key: `{em['join_key']}`  ·  generated: {em['generated_at']}")
    st.dataframe(
        pd.DataFrame([{"dataset": k, "producer": d["producer"],
                       "file": d.get("parquet", "bsoid_labels/ (per-video)"), "rows": d.get("rows")}
                      for k, d in em["datasets"].items()]),
        use_container_width=True, hide_index=True,
    )
    with st.expander("data_dictionary.md"):
        dd = export_dir(cfg) / "data_dictionary.md"
        if dd.exists():
            st.markdown(dd.read_text())
