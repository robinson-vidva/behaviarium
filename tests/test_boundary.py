from behaviarium.config import BoundaryParams
from behaviarium.stages.boundary import detect_roi
from behaviarium.video import read_frame


def test_detect_roi_bbox_within_tolerance(tmp_path, make_rect_video):
    path = make_rect_video(tmp_path / "v.avi", n_frames=3, size=(200, 160), rect=(40, 30, 100, 80))
    frame = read_frame(path, 0)
    geom = detect_roi(frame, BoundaryParams(shape="rect", threshold=127, min_area_frac=0.01))
    assert geom is not None and geom["shape"] == "rect"
    # detected bbox should be within a few px of the known rectangle (MJPG is mildly lossy)
    assert abs(geom["x"] - 40) <= 6
    assert abs(geom["y"] - 30) <= 6
    assert abs(geom["w"] - 100) <= 8
    assert abs(geom["h"] - 80) <= 8


def test_detect_roi_circle_shape_hint(tmp_path, make_rect_video):
    path = make_rect_video(tmp_path / "v.avi", n_frames=3, size=(200, 160), rect=(50, 40, 90, 90))
    frame = read_frame(path, 0)
    geom = detect_roi(frame, BoundaryParams(shape="circle", threshold=127, min_area_frac=0.01))
    assert geom is not None and geom["shape"] == "circle"
    assert geom["r"] > 0
