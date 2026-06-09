"""B-SOiD per-frame label reconstruction (decision #2). Pure, backend-independent.

B-SOiD predicts on frameshifted feature streams at ~10 Hz: one prediction stream per starting
frame offset, ``n_shift = floor(fps / 10)`` of them (matching the authoritative YttriLab
B-SOiD source). The per-frame labels are reconstructed by
the FRAMESHIFT + ``flatten('F')`` INTERLEAVE — NOT by block-repeating each label n_shift times
(the legacy ``np.repeat``/``tile`` approach, which is wrong and has been failing).

Stream j (j = 0..n_shift-1) holds the predictions for frames j, j+n_shift, j+2*n_shift, ...
Stack the streams as rows of an (n_shift, n_bins) matrix (shorter streams right-padded), then
column-major ``flatten('F')`` reads it as frame 0 = stream0[bin0], frame 1 = stream1[bin0], ...
frame n_shift = stream0[bin1], ... — exactly per-frame order. Trim to the original frame count
(the pad values fall in the trimmed tail).
"""

from __future__ import annotations

import math

import numpy as np

_PAD = -1


def n_shift_for_fps(fps: float) -> int:
    """n_shift = floor(fps/10), at least 1 (per-video, from corrected_fps — never a literal).

    floor (not round) matches the authoritative YttriLab B-SOiD source."""
    return max(1, int(math.floor(fps / 10.0)))


def reconstruct_labels(streams: list[np.ndarray], n_frames: int) -> np.ndarray:
    """Interleave per-offset prediction ``streams`` back to ``n_frames`` per-frame labels.

    ``streams[j]`` = predictions for frame-offset j. This is the frameshift + flatten('F')
    reconstruction, deliberately NOT block-repeat."""
    if not streams:
        raise ValueError("reconstruct_labels needs at least one stream")
    n_shift = len(streams)
    n_bins = max(len(s) for s in streams)
    mat = np.full((n_shift, n_bins), _PAD, dtype=int)
    for j, s in enumerate(streams):
        mat[j, : len(s)] = np.asarray(s, dtype=int)
    flat = mat.flatten("F")  # column-major interleave back to per-frame resolution
    return flat[:n_frames]


def offset_lengths(n_frames: int, n_shift: int) -> list[int]:
    """Number of ~10Hz bins each frame-offset stream contains (offset j: frames j, j+n_shift...)."""
    return [len(range(j, n_frames, n_shift)) for j in range(n_shift)]
