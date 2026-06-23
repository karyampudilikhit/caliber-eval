"""Multiple-comparisons correction — Benjamini-Hochberg false discovery rate.

When you call ``compare()`` on N metrics at once, the chance that at least one
comes back BETTER/WORSE *by accident* under H₀ grows with N. Without correction,
testing 20 metrics at α=0.05 gives you about a 64% chance of at least one
spurious "significant" verdict.

The Benjamini-Hochberg procedure (Benjamini & Hochberg 1995) controls the
**false discovery rate** (expected fraction of false rejections among all
rejections) rather than the family-wise error rate. It's strictly less
conservative than Bonferroni and the right default for ranking many metrics
where you can tolerate some false discoveries.

Procedure
---------
1. Sort p-values ascending: p₍₁₎ ≤ p₍₂₎ ≤ … ≤ p₍ₙ₎
2. Find the largest i such that p₍ᵢ₎ ≤ (i/n)·α
3. Reject H₀ for tests 1..i (in sorted order); fail to reject the rest

References
----------
.. [1] Benjamini, Y. & Hochberg, Y. "Controlling the false discovery rate:
       a practical and powerful approach to multiple testing."
       J. Royal Statistical Society, Series B, 57(1):289-300 (1995).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def benjamini_hochberg(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> list[bool]:
    """Apply the Benjamini-Hochberg step-up FDR correction.

    Parameters
    ----------
    p_values : sequence of float
        The unadjusted p-values from your N tests, in any order. The returned
        list preserves the input order: ``result[i]`` corresponds to
        ``p_values[i]``.
    alpha : float in (0, 1), default 0.05
        Target false discovery rate.

    Returns
    -------
    list[bool]
        ``True`` for each test whose null hypothesis is rejected after FDR
        correction; ``False`` otherwise.

    Raises
    ------
    ValueError
        If any p-value is outside [0, 1], if ``p_values`` is empty, or if
        ``alpha`` is outside (0, 1).

    Examples
    --------
    >>> from caliber import benjamini_hochberg
    >>> benjamini_hochberg([0.001, 0.02, 0.025, 0.5, 0.9], alpha=0.05)
    [True, True, True, False, False]
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    p_arr = np.asarray(p_values, dtype=np.float64)
    if p_arr.size == 0:
        raise ValueError("p_values must be non-empty")
    if p_arr.ndim != 1:
        raise ValueError(f"p_values must be 1-dimensional; got shape {p_arr.shape}")
    if not np.all(np.isfinite(p_arr)):
        raise ValueError("p_values contains NaN or Inf")
    if np.any(p_arr < 0.0) or np.any(p_arr > 1.0):
        raise ValueError("p_values must lie in [0, 1]")

    n = p_arr.size
    # Sort and track original indices so we can map decisions back.
    order = np.argsort(p_arr, kind="stable")
    sorted_p = p_arr[order]

    # BH thresholds: i/n * alpha for i = 1..n.
    ranks = np.arange(1, n + 1, dtype=np.float64)
    thresholds = ranks / n * alpha

    # Largest i where sorted_p[i-1] ≤ threshold[i-1]; reject all up to and
    # including that i. If no i satisfies, reject nothing.
    below = sorted_p <= thresholds
    if not np.any(below):
        return [False] * n
    cutoff_sorted_idx = int(np.max(np.where(below)[0]))

    rejected_sorted = np.zeros(n, dtype=bool)
    rejected_sorted[: cutoff_sorted_idx + 1] = True

    # Map back to input order.
    rejected = np.empty(n, dtype=bool)
    rejected[order] = rejected_sorted
    # Explicit list comprehension keeps the return type as list[bool] across
    # mypy versions; .tolist() returns Any on some numpy-stub versions.
    return [bool(x) for x in rejected]
