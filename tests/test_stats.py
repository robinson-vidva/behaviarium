import numpy as np
import pandas as pd

from behaviarium.stats import benjamini_hochberg, group_comparison, permutation_pvalue


def test_benjamini_hochberg_monotone_and_known_values():
    q = benjamini_hochberg([0.01, 0.02, 0.5])
    assert list(np.round(q, 4)) == [0.03, 0.03, 0.5]

    p = np.array([0.001, 0.2, 0.04, 0.3, 0.5])
    q = benjamini_hochberg(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)  # monotone non-decreasing in p order
    assert np.all(q >= p - 1e-12)  # q >= raw p
    assert np.all((q >= 0) & (q <= 1))


def test_permutation_pvalue_reproducible_under_seed():
    a = np.array([0.10, 0.20, 0.15, 0.18, 0.12, 0.20])
    b = np.array([0.80, 0.90, 0.85, 0.82, 0.88, 0.90])
    r1 = permutation_pvalue(a, b, 500, np.random.default_rng(0))
    r2 = permutation_pvalue(a, b, 500, np.random.default_rng(0))
    assert r1 == r2  # same seed -> identical stat and p


def test_planted_difference_significant_null_not():
    rows = []
    for i in range(6):  # 6 videos per group
        rows.append({"group": "A", "cluster": 0, "fraction": 0.10 + 0.01 * i})  # low
        rows.append({"group": "B", "cluster": 0, "fraction": 0.90 + 0.01 * i})  # high (planted)
        rows.append({"group": "A", "cluster": 1, "fraction": 0.50 + 0.01 * i})  # identical
        rows.append({"group": "B", "cluster": 1, "fraction": 0.50 + 0.01 * i})  # identical (null)
    df = pd.DataFrame(rows)
    res = group_comparison(df, "cluster", "group", "fraction", n_permutations=2000, alpha=0.05, seed=0)

    r0 = res[res["cluster"] == 0].iloc[0]
    r1 = res[res["cluster"] == 1].iloc[0]
    assert r0["p_value"] < 0.05 and bool(r0["significant"]) is True
    assert r1["p_value"] > 0.05 and bool(r1["significant"]) is False
    assert r0["wasserstein_stat"] > r1["wasserstein_stat"]


def test_group_comparison_columns_and_config_factor():
    rows = []
    for i in range(4):
        rows.append({"housing": "PT", "cluster": 0, "fraction": 0.3 + 0.01 * i})
        rows.append({"housing": "EE", "cluster": 0, "fraction": 0.4 + 0.01 * i})
    res = group_comparison(pd.DataFrame(rows), "cluster", "housing", "fraction", 200, 0.05, 0)
    assert {"cluster", "group_a", "group_b", "n_a", "n_b", "wasserstein_stat",
            "p_value", "q_value", "significant"} <= set(res.columns)
    assert {res.iloc[0]["group_a"], res.iloc[0]["group_b"]} == {"EE", "PT"}
