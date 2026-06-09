"""Real significance testing: Wasserstein permutation + Benjamini-Hochberg FDR.

These produce the ONLY columns in the whole pipeline allowed to be named ``p_value`` /
``q_value`` / ``significant`` (decision #4). Generic over the unit of analysis (cluster or
region) and the grouping column — all of which come from per-project config.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values (monotone, clipped to [0,1])."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    scaled = p[order] * n / ranks
    # enforce monotonicity from the largest p downward
    q_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q


def permutation_pvalue(a: np.ndarray, b: np.ndarray, n_permutations: int, rng) -> tuple[float, float]:
    """Wasserstein distance between samples ``a`` and ``b`` + a label-shuffle permutation p-value.

    p = (#perm stats >= observed + 1) / (n_permutations + 1)  (add-one empirical p; never 0)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = float(wasserstein_distance(a, b))
    combined = np.concatenate([a, b])
    na = a.size
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        if wasserstein_distance(perm[:na], perm[na:]) >= observed:
            count += 1
    pvalue = (count + 1) / (n_permutations + 1)
    return observed, pvalue


def group_comparison(
    long_df: pd.DataFrame,
    unit_col: str,
    group_col: str,
    value_col: str,
    n_permutations: int,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    """For each unit (cluster/region), compare the first two groups' per-video ``value_col``
    distributions via Wasserstein + permutation p, then BH across units -> q + significant."""
    groups = sorted(pd.Series(long_df[group_col]).dropna().unique().tolist())
    if len(groups) < 2:
        raise ValueError(f"group column {group_col!r} has <2 groups: {groups}")
    ga, gb = groups[0], groups[1]

    rng = np.random.default_rng(seed)
    rows = []
    for unit in sorted(long_df[unit_col].unique().tolist()):
        sub = long_df[long_df[unit_col] == unit]
        a = sub[sub[group_col] == ga][value_col].to_numpy(dtype=float)
        b = sub[sub[group_col] == gb][value_col].to_numpy(dtype=float)
        if a.size == 0 or b.size == 0:
            stat, pvalue = float("nan"), 1.0
        else:
            stat, pvalue = permutation_pvalue(a, b, n_permutations, rng)
        rows.append(
            {
                unit_col: unit,
                "group_a": ga,
                "group_b": gb,
                "n_a": int(a.size),
                "n_b": int(b.size),
                "wasserstein_stat": stat,
                "p_value": pvalue,
            }
        )
    out = pd.DataFrame(rows)
    out["q_value"] = benjamini_hochberg(out["p_value"].to_numpy())
    out["significant"] = out["q_value"] < alpha
    return out
