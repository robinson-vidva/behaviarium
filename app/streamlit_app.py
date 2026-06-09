"""Behaviarium control plane (thin).

Streamlit reruns top-to-bottom on every interaction, so this script does NO long compute
inline — it only reads the manifest, renders status, and edits small bits of state. Every
video/image operation is launched as a background subprocess (``python -m behaviarium ...``)
tracked by the manifest.
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
from behaviarium.config import default_config_dir, load_config
from behaviarium.manifest import Approval, Manifest, Status, VideoKey
from behaviarium.paths import (
    boundary_preview,
    bsoid_clusters_csv,
    chamber_csv,
    export_dir,
    postprocess_bsoid_long,
    stats_bsoid_table,
    stats_chamber_table,
)
from behaviarium.pipeline import PROJECT_STAGES, STAGE_ORDER
from behaviarium.registry import registered_stages
from behaviarium.runner import project_key

st.set_page_config(page_title="Behaviarium", layout="wide")
st.title("Behaviarium — control plane")

PER_VIDEO_STAGES = ["rotate", "boundary", "mask", "dlc", "chamber", "bsoid"]


def available_projects() -> list[str]:
    proj_dir = default_config_dir() / "projects"
    return sorted(p.stem for p in proj_dir.glob("*.yml"))


projects = available_projects()
if not projects:
    st.error(f"No project configs found under {default_config_dir() / 'projects'}")
    st.stop()

project = st.sidebar.selectbox("Project", projects)
cfg = load_config(project)
manifest = Manifest(cfg.manifest_path)


def spawn(*cli_args: str) -> None:
    """Launch a behaviarium CLI subcommand as a background subprocess (never inline compute)."""
    subprocess.Popen(
        [sys.executable, "-m", "behaviarium", *cli_args],
        cwd=str(cfg.root),
        env=os.environ.copy(),
    )


st.sidebar.write(f"**Assay:** `{cfg.assay}`")
st.sidebar.write(f"**data_root:** `{cfg.data_root}`")
st.sidebar.write(f"**manifest:** `{cfg.manifest_path}`")

if st.sidebar.button("Run ingest", help="Discover videos (background subprocess)"):
    spawn("ingest", "--project", project)
    st.sidebar.info("Ingest launched. Click Refresh to update.")

st.sidebar.markdown("**Run a stage over all videos**")
batch_stage = st.sidebar.selectbox("Stage", PER_VIDEO_STAGES, key="batch_stage")
if st.sidebar.button(f"Run {batch_stage} (all)"):
    spawn("run-all", "--project", project, "--stage", batch_stage)
    st.sidebar.info(f"{batch_stage} launched for all videos. Click Refresh.")

st.sidebar.markdown("**Run a project-level stage**")
proj_stage = st.sidebar.selectbox("Project stage", PROJECT_STAGES, key="proj_stage")
if st.sidebar.button(f"Run {proj_stage} (project)"):
    spawn("run-project", "--project", project, "--stage", proj_stage)
    st.sidebar.info(f"{proj_stage} launched. Click Refresh.")

if st.sidebar.button("Refresh"):
    st.rerun()

# Per-video status columns (exclude project-level aggregate stages — shown separately).
all_stage_names = {name for name, _assay in registered_stages()}
stage_names = [s for s in STAGE_ORDER if s in all_stage_names and s not in PROJECT_STAGES]

if not Path(cfg.manifest_path).exists():
    st.info("No manifest yet. Click **Run ingest** to discover videos.")
    st.stop()

videos = manifest.list_videos()
if not videos:
    st.info("Manifest is empty. Click **Run ingest** to discover videos.")
    st.stop()

# --- read-only status table -------------------------------------------------------------
rows = []
for v in videos:
    key = VideoKey(v["type"], v["class"], v["filename"])
    row = {
        "Include": bool(v["include"]),
        "Type": v["type"],
        "Class": v["class"],
        "Filename": v["filename"],
        "frames": v["frame_count"],
        "corrected_fps": v["fps"],
    }
    for name in stage_names:
        status = manifest.get_status(key, name) or "—"
        if name == "boundary":
            approval = manifest.get_approval(key, name)
            if approval:
                status = f"{status} ({approval})"
        elif name == "dlc":
            p = (manifest.get_row(key, name) or {}).get("params") or {}
            if p:
                engine = str(p.get("backend", "?")).split(" ")[0]  # stub / tensorflow
                status = f"{status} [{engine}, {'filtered' if p.get('filtered') else 'raw'}]"
        elif name == "bsoid":
            p = (manifest.get_row(key, name) or {}).get("params") or {}
            if p:
                engine = str(p.get("backend", "?")).split(" ")[0]  # stub / real
                status = f"{status} [{engine}, n_shift={p.get('n_shift')}]"
        row[name] = status
    rows.append(row)

st.subheader(f"Videos × stage status ({len(rows)})")
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- per-video control panel ------------------------------------------------------------
st.divider()
st.subheader("Per-video control")

labels = [f"{v['type']} / {v['class']} / {v['filename']}" for v in videos]
choice = st.selectbox("Video", range(len(videos)), format_func=lambda i: labels[i])
v = videos[choice]
key = VideoKey(v["type"], v["class"], v["filename"])

col_a, col_b = st.columns(2)

with col_a:
    include = st.checkbox("Include this video (runner skips when off)", value=bool(v["include"]))
    if include != bool(v["include"]):
        manifest.set_include(key, include)
        st.rerun()

    st.markdown("**Re-run a stage** (resets the row to pending, then runs in the background)")
    rerun_stage = st.selectbox("Stage to re-run", PER_VIDEO_STAGES, key="rerun_stage")
    if st.button(f"Re-run {rerun_stage}"):
        manifest.set_status(key, rerun_stage, Status.PENDING)
        spawn("run", "--project", project, "--stage", rerun_stage,
              "--type", key.type, "--class", key.klass, "--filename", key.filename)
        st.info(f"Re-running {rerun_stage} for this video. Click Refresh.")

with col_b:
    st.markdown("**Boundary review**")
    brow = manifest.get_row(key, "boundary")
    if not brow:
        st.caption("No boundary result yet. Run rotate then boundary.")
    else:
        st.caption(f"approval: `{brow.get('approval')}`  ·  status: `{brow.get('status')}`")
        preview = boundary_preview(cfg, key)
        if preview.exists():
            st.image(str(preview), caption=str(preview), use_container_width=True)
        else:
            st.caption("No preview PNG yet.")

        c1, c2 = st.columns(2)
        if c1.button("Approve"):
            manifest.set_approval(key, "boundary", Approval.APPROVED)
            st.rerun()
        if c2.button("Reject"):
            manifest.set_approval(key, "boundary", Approval.REJECTED)
            st.rerun()

        geom = (brow.get("params") or {}).get("roi") or {}
        shape = geom.get("shape", cfg.project.boundary.shape)
        st.markdown(f"**Manual ROI override** (shape: `{shape}`)")
        if shape == "circle":
            cx = st.number_input("cx", value=int(geom.get("cx", 0)), step=1)
            cy = st.number_input("cy", value=int(geom.get("cy", 0)), step=1)
            r = st.number_input("r", value=int(geom.get("r", 1)), step=1)
            new_geom = {"shape": "circle", "cx": int(cx), "cy": int(cy), "r": int(r)}
        else:
            x = st.number_input("x", value=int(geom.get("x", 0)), step=1)
            y = st.number_input("y", value=int(geom.get("y", 0)), step=1)
            w = st.number_input("w", value=int(geom.get("w", 1)), step=1)
            h = st.number_input("h", value=int(geom.get("h", 1)), step=1)
            new_geom = {"shape": "rect", "x": int(x), "y": int(y), "w": int(w), "h": int(h)}

        if st.button("Apply override"):
            params = dict(brow.get("params") or {})
            params["roi"] = new_geom
            manifest.set_params(key, "boundary", params)
            manifest.set_approval(key, "boundary", Approval.PENDING_REVIEW)
            spawn("render-preview", "--project", project,
                  "--type", key.type, "--class", key.klass, "--filename", key.filename)
            st.info("Override saved; preview re-rendering in the background. Click Refresh.")

# --- chamber occupancy summary (read-only, from the chamber output) ---------------------
st.divider()
st.subheader("Chamber occupancy")
chamber_out = chamber_csv(cfg, key)
if not chamber_out.exists():
    st.caption("No chamber output yet. Run the chamber stage for this video.")
else:
    occ = pd.read_csv(chamber_out)
    st.dataframe(occ, use_container_width=True, hide_index=True)
    st.caption("Fraction of frames per region")
    st.bar_chart(occ.set_index("region")["fraction"])

# --- B-SOiD cluster usage summary (read-only, from the bsoid output) ---------------------
st.divider()
brow = manifest.get_row(key, "bsoid")
bp = (brow or {}).get("params") or {}
st.subheader(f"B-SOiD cluster usage  ·  backend: `{bp.get('backend', '—')}`  ·  n_shift: `{bp.get('n_shift', '—')}`")
bsoid_out = bsoid_clusters_csv(cfg, key)
if not bsoid_out.exists():
    st.caption("No bsoid output yet. Run the bsoid stage for this video.")
else:
    clusters = pd.read_csv(bsoid_out)
    st.dataframe(clusters, use_container_width=True, hide_index=True)
    st.caption("Fraction of frames per cluster")
    st.bar_chart(clusters.set_index("cluster")["fraction"])

# --- project-level stages: postprocess + real stats -------------------------------------
st.divider()
st.header("Project-level: aggregate + significance")
pkey = project_key(cfg)
pcols = st.columns(len(PROJECT_STAGES))
for col, pstage in zip(pcols, PROJECT_STAGES):
    status = manifest.get_status(pkey, pstage) or "—"
    col.metric(pstage, status)
    if col.button(f"Re-run {pstage}", key=f"rerun_{pstage}"):
        manifest.set_status(pkey, pstage, Status.PENDING)
        spawn("run-project", "--project", project, "--stage", pstage)
        col.info(f"Re-running {pstage}. Click Refresh.")

agg = postprocess_bsoid_long(cfg, ".csv")
if agg.exists():
    st.caption("Group aggregate — B-SOiD clusters long (head)")
    st.dataframe(pd.read_csv(agg).head(20), use_container_width=True, hide_index=True)

srow = manifest.get_row(pkey, "stats")
sp = (srow or {}).get("params") or {}
if sp:
    st.caption(
        f"Stats — group_factor: `{sp.get('group_factor')}`  ·  "
        f"N permutations: `{sp.get('n_permutations')}`  ·  alpha: `{sp.get('alpha')}`"
    )
for label, path in [("B-SOiD cluster stats", stats_bsoid_table(cfg, ".csv")),
                    ("Chamber region stats", stats_chamber_table(cfg, ".csv"))]:
    if path.exists():
        st.subheader(label)
        st.dataframe(pd.read_csv(path), use_container_width=True, hide_index=True)

# --- export bundle listing (read-only) --------------------------------------------------
st.divider()
st.subheader("Export bundle")
bundle = export_dir(cfg)
manifest_json = bundle / "export_manifest.json"
if not manifest_json.exists():
    st.caption("No export bundle yet. Run the export stage.")
else:
    em = json.loads(manifest_json.read_text())
    st.caption(f"Bundle: `{bundle}`  ·  generated: `{em.get('generated_at')}`")
    listing = []
    for name, d in em["datasets"].items():
        listing.append({"dataset": name, "producer": d["producer"],
                        "file": d.get("parquet", "bsoid_labels/ (per-video)"), "rows": d.get("rows")})
    st.dataframe(pd.DataFrame(listing), use_container_width=True, hide_index=True)
    with st.expander("export_manifest.json"):
        st.json(em)
    dd = bundle / "data_dictionary.md"
    if dd.exists():
        with st.expander("data_dictionary.md"):
            st.markdown(dd.read_text())
