"""DeepLabCut tracking-CSV I/O — backend-independent (no TF/DLC import).

DLC single-animal CSV schema (3 header rows + frame index), verified to round-trip via pandas:

    scorer,<scorer>,<scorer>,...
    bodyparts,<bp>,<bp>,<bp>,...        # each bodypart repeated for x,y,likelihood
    coords,x,y,likelihood,...
    0,<x>,<y>,<lik>,...

Bodyparts are referenced BY NAME through the multiindex header (``header=[0,1,2]``), never by
positional columns like ``x.1`` / ``x.11`` (the legacy brittleness this project fixes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COORDS = ["x", "y", "likelihood"]
_LEVELS = ["scorer", "bodyparts", "coords"]


def build_dlc_dataframe(
    scorer: str, bodyparts: list[str], coords: np.ndarray
) -> pd.DataFrame:
    """Build a DLC-schema DataFrame. ``coords`` has shape (n_frames, n_bodyparts*3)."""
    columns = pd.MultiIndex.from_product([[scorer], bodyparts, COORDS], names=_LEVELS)
    if coords.shape[1] != len(columns):
        raise ValueError(f"coords has {coords.shape[1]} cols, expected {len(columns)}")
    index = pd.RangeIndex(coords.shape[0])  # frame numbers
    return pd.DataFrame(coords, index=index, columns=columns)


def write_dlc_csv(df: pd.DataFrame, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def read_dlc_csv(path: Path) -> pd.DataFrame:
    """Read a DLC tracking CSV with its 3-row multiindex header and frame index."""
    return pd.read_csv(path, header=[0, 1, 2], index_col=0)


def scorer_name(df: pd.DataFrame) -> str:
    return str(df.columns.get_level_values("scorer")[0])


def list_bodyparts(df: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(df.columns.get_level_values("bodyparts")))


def get_bodypart(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Columns (x, y, likelihood) for one bodypart, selected BY NAME via the multiindex."""
    sub = df.xs(name, axis=1, level="bodyparts")
    if isinstance(sub.columns, pd.MultiIndex):
        sub = sub.droplevel("scorer", axis=1)  # leave only the coords level
    return sub


def median_filter(df: pd.DataFrame, windowlength: int) -> pd.DataFrame:
    """Rolling-median smoothing over every coordinate column (length-preserving).

    Backend-independent so it works for both the stub and real DLC output."""
    out = df.copy()
    out.iloc[:, :] = df.rolling(window=windowlength, center=True, min_periods=1).median()
    return out
