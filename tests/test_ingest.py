from behaviarium import stages  # noqa: F401  registers the ingest stage
from behaviarium.config import load_config
from behaviarium.manifest import Manifest, VideoKey
from behaviarium.runner import run_stage


def test_ingest_registers_rows_and_corrected_fps(monkeypatch, tmp_path, make_video):
    data = tmp_path / "data"
    # layout: data_root/<Type>/<Class>/<filename>
    make_video(data / "Saline" / "PT" / "m1.avi", n_frames=30)
    make_video(data / "LPS" / "EE" / "m2.avi", n_frames=60)

    monkeypatch.setenv("BEHAVIARIUM_DATA_ROOT", str(data))
    monkeypatch.setenv("BEHAVIARIUM_OUTPUT_ROOT", str(tmp_path / "out"))
    cfg = load_config("pt_social")

    manifest = Manifest(cfg.manifest_path)
    manifest.init()
    run_stage("ingest", cfg, manifest)

    videos = manifest.list_videos()
    assert len(videos) == 2

    # corrected fps = actual_frame_count / recording_duration_s (config knob, no buried 600)
    duration = cfg.recording_duration_s
    k1 = VideoKey("Saline", "PT", "m1.avi")
    rec = manifest.get_video(k1)
    assert rec is not None
    assert rec["frame_count"] == 30
    assert rec["fps"] == 30 / duration
    assert manifest.get_status(k1, "ingest") == "done"

    k2 = VideoKey("LPS", "EE", "m2.avi")
    assert manifest.get_video(k2)["fps"] == 60 / duration


def test_recording_duration_env_override(monkeypatch, tmp_path, make_video):
    make_video(tmp_path / "data" / "T" / "C" / "v.avi", n_frames=90)
    monkeypatch.setenv("BEHAVIARIUM_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("BEHAVIARIUM_OUTPUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("BEHAVIARIUM_RECORDING_DURATION_S", "300")
    cfg = load_config("pt_social")
    assert cfg.recording_duration_s == 300.0
    manifest = Manifest(cfg.manifest_path)
    manifest.init()
    run_stage("ingest", cfg, manifest)
    assert manifest.get_video(VideoKey("T", "C", "v.avi"))["fps"] == 90 / 300


def test_recording_duration_project_override():
    cfg = load_config("pt_social")
    cfg.project.recording_duration_s = 120.0  # per-project override wins over core
    assert cfg.recording_duration_s == 120.0
