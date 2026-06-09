import numpy as np

from behaviarium.roi import apply_mask, bbox_of, roi_to_mask
from behaviarium.video import apply_rotation


def test_rotate_yields_expected_dimensions():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)  # (H=10, W=20)
    assert apply_rotation(frame, 90).shape == (20, 10, 3)  # 90/270 swap dims
    assert apply_rotation(frame, 270).shape == (20, 10, 3)
    assert apply_rotation(frame, 180).shape == (10, 20, 3)  # 180 preserves dims
    assert apply_rotation(frame, 0).shape == (10, 20, 3)
    # flip changes orientation, not dimensions
    assert apply_rotation(frame, 0, flip="horizontal").shape == (10, 20, 3)


def test_mask_zeroes_pixels_outside_roi():
    frame = np.full((50, 60, 3), 255, dtype=np.uint8)
    geom = {"shape": "rect", "x": 10, "y": 5, "w": 30, "h": 20}
    out = apply_mask(frame, geom)
    assert out.shape == frame.shape
    assert (out[0, 0] == 0).all()  # outside the ROI -> zeroed
    assert (out[6, 11] == 255).all()  # inside the ROI -> preserved
    assert (out[40, 50] == 0).all()  # outside the ROI -> zeroed


def test_mask_crop_returns_bbox_sized_frame():
    frame = np.full((50, 60, 3), 255, dtype=np.uint8)
    geom = {"shape": "rect", "x": 10, "y": 5, "w": 30, "h": 20}
    out = apply_mask(frame, geom, crop=True)
    assert out.shape == (20, 30, 3)


def test_roi_to_mask_circle_and_bbox():
    mask = roi_to_mask({"shape": "circle", "cx": 30, "cy": 30, "r": 10}, (60, 60))
    assert mask[30, 30] == 255  # center inside
    assert mask[0, 0] == 0  # corner outside
    assert bbox_of({"shape": "circle", "cx": 30, "cy": 30, "r": 10}) == (20, 20, 20, 20)
