"""Paired bootstrap for `caliber.core.compare`.

The paired bootstrap treats the array of paired differences as the population
of interest and resamples from it with replacement. For each of B resamples we
compute the mean; the empirical distribution of those B means approximates the
sampling distribution of the mean.

CI : percentile method — α/2 and 1−α/2 quantiles of the bootstrap distribution.
p  : bootstrap test of H₀: μ = 0 — centre the bootstrap distribution at zero
     (subtract the observed mean) and report the fraction of centred means at
     least as extreme as the observed mean in absolute value, with the
     (k+1)/(B+1) adjustment so a finite resample can never yield p=0.

Why these choices
-----------------
The percentile method is the simplest defensible bootstrap CI: no studentisation,
no acceleration, no bias correction. It under-covers slightly for small n
(empirically ~88–93% at n=20 instead of the nominal 95%) but its semantics are
transparent and it's the canonical choice for a v1 release.

The (k+1)/(B+1) p-value adjustment is standard practice (Davison & Hinkley
1997, §4.2.3); it prevents the absurd "p = 0" report from a finite Monte-Carlo
test where the true tail probability is merely unobserved.

References
----------
.. [1] Efron, B. & Tibshirani, R.J. *An Introduction to the Bootstrap* (1993).
.. [2] Davison, A.C. & Hinkley, D.V. *Bootstrap Methods and their Application* (1997).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def paired_bootstrap_ci(
    diff: np.ndarray[Any, Any],
    *,
    confidence_level: float,
    n_bootstrap: int,
    seed: int | None,
) -> tuple[float, float, float, float]:
    """Return ``(delta, ci_lower, ci_upper, p_value)`` for the paired bootstrap.

    Parameters
    ----------
    diff : np.ndarray, shape (n,)
        Paired differences ``new - old``. Must have ``n >= 2`` and not be all-zero
        (the caller validates this upstream).
    confidence_level : float in (0, 1)
        Two-sided CI level — e.g. 0.95.
    n_bootstrap : int
        Number of bootstrap resamples. 10_000 is the standard default.
    seed : int | None
        Seed for the bootstrap RNG. ``None`` → fresh non-deterministic state.

    Returns
    -------
    (delta, ci_lower, ci_upper, p_value) : tuple of float
        ``delta`` is the observed mean of ``diff``; the CI is the percentile-method
        CI on the bootstrap distribution; ``p_value`` is the two-sided bootstrap
        p-value for ``H₀: μ_diff = 0``.

    Notes
    -----
    Memory usage is O(B · n) for the indices and resampled-mean arrays. At the
    defaults (B=10_000) this is ~80 MB at n=1000, fine for typical eval sizes.
    """
    n = len(diff)
    delta = float(np.mean(diff))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = diff[indices].mean(axis=1)

    # Percentile-method CI.
    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    # Bootstrap test of H₀: μ = 0.
    # `boot_means - delta` mimics the sampling distribution centred at zero;
    # the proportion ≥ |delta| in absolute value is the two-sided p-value.
    centred_means = boot_means - delta
    n_more_extreme = int(np.sum(np.abs(centred_means) >= abs(delta)))
    p_value = (n_more_extreme + 1) / (n_bootstrap + 1)

    return delta, ci_lower, ci_upper, p_value
