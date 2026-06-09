"""Statistics package — the real significance path (Wasserstein permutation + BH FDR)."""

from .significance import benjamini_hochberg, group_comparison, permutation_pvalue

__all__ = ["benjamini_hochberg", "permutation_pvalue", "group_comparison"]
