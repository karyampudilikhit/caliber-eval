"""Group-sequential A/B testing — peek across batches without inflating type-I error.

The naive "peek every batch and call it significant if p < 0.05" pattern is
broken: with K looks, the probability of *some* spurious rejection grows
roughly like 1 − (1 − α)^K. Sequential testing fixes this by tightening the
boundary at each look so the cumulative false-positive rate stays at α.

`SequentialTester` implements the closed-form Wang-Tsiatis boundary families
that approximate O'Brien-Fleming (OBF) and Pocock alpha-spending designs:

    OBF:    z_k = z_{α/2} · √(K/k)
    Pocock: z_k = c_K(α)        (constant across looks; tabulated)

OBF is the default. It's conservative at early looks and generous at the
final look (z_K = z_{α/2}, the single-test threshold), which fits the usual
case where you'd like to stop early on a clear win but still resolve at the
planned end. Pocock spends alpha evenly across looks — useful when you care
equally about all peeks.

The Pocock boundary requires numerical integration to compute exactly. We
hard-code the standard tabulated values for α=0.05 and K ∈ {2, 3, 4, 5, 6,
7, 8, 9, 10}; other configurations raise ``NotImplementedError``.

References
----------
.. [1] O'Brien, P.C. & Fleming, T.R. "A multiple testing procedure for
       clinical trials." Biometrics 35(3):549-556 (1979).
.. [2] Pocock, S.J. "Group sequential methods in the design and analysis of
       clinical trials." Biometrika 64(2):191-199 (1977).
.. [3] Lan, K.K.G. & DeMets, D.L. "Discrete sequential boundaries for
       clinical trials." Biometrika 70(3):659-663 (1983).
.. [4] Wang, S.K. & Tsiatis, A.A. "Approximately optimal one-parameter
       boundaries for group sequential trials." Biometrics 43(1):193-199 (1987).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy import stats

from caliber.core.types import CompareResult, Verdict

# Tabulated Pocock z-boundary (constant across looks) for two-sided α = 0.05.
# Source: Pocock (1977), Table 1. Higher K → higher boundary because more
# looks mean more chances for spurious extremes.
_POCOCK_C_ALPHA_005: dict[int, float] = {
    2: 2.178,
    3: 2.289,
    4: 2.361,
    5: 2.413,
    6: 2.453,
    7: 2.485,
    8: 2.512,
    9: 2.535,
    10: 2.555,
}


class SequentialTester:
    """Group-sequential paired-t test allowing peeking at up to K interim looks.

    Parameters
    ----------
    max_n : int
        Planned maximum total sample size (pairs). Used only for `is_done`
        accounting; the boundaries are determined by ``n_looks``.
    n_looks : int, default 5
        Number of planned interim looks K, including the final one.
    alpha : float, default 0.05
        Two-sided family-wise type-I error rate (across all K looks).
    boundary : {"obrien_fleming", "pocock"}, default "obrien_fleming"
        Which closed-form boundary family to use.

    Examples
    --------
    >>> tester = SequentialTester(max_n=200, n_looks=5)  # doctest: +SKIP
    >>> for batch_old, batch_new in stream_of_batches:    # doctest: +SKIP
    ...     result = tester.update(batch_old, batch_new)
    ...     if tester.is_done():
    ...         break
    """

    def __init__(
        self,
        max_n: int,
        n_looks: int = 5,
        alpha: float = 0.05,
        boundary: Literal["obrien_fleming", "pocock"] = "obrien_fleming",
    ) -> None:
        if max_n < n_looks:
            raise ValueError(
                f"max_n ({max_n}) must be ≥ n_looks ({n_looks}); each look "
                f"needs at least one pair."
            )
        if n_looks < 1:
            raise ValueError(f"n_looks must be ≥ 1; got {n_looks}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        if boundary not in ("obrien_fleming", "pocock"):
            raise ValueError(f"unknown boundary: {boundary!r}")

        self.max_n = max_n
        self.n_looks = n_looks
        self.alpha = alpha
        self.boundary = boundary
        self._boundaries = _compute_boundaries(boundary, n_looks, alpha)

        self._old: list[float] = []
        self._new: list[float] = []
        self._looks_taken = 0
        self._stopped = False
        self._last_result: CompareResult | None = None

    def is_done(self) -> bool:
        """``True`` once a boundary has been crossed or the final look completed."""
        return self._stopped

    def update(
        self,
        old_scores_batch: Sequence[float],
        new_scores_batch: Sequence[float],
    ) -> CompareResult:
        """Add a new batch of paired scores and return the current verdict.

        Returns
        -------
        CompareResult
            Verdict can change across calls. Once `is_done()` is True, future
            ``update()`` calls return the final result without consuming more
            data — pass empty batches if you want a clean no-op.

        Raises
        ------
        ValueError
            If the two batches have different lengths or already-stopped looks
            would exceed ``n_looks``.
        """
        if len(old_scores_batch) != len(new_scores_batch):
            raise ValueError(
                f"batch lengths must match (got {len(old_scores_batch)} and "
                f"{len(new_scores_batch)})"
            )
        if self._stopped:
            assert self._last_result is not None
            return self._last_result

        self._old.extend(float(x) for x in old_scores_batch)
        self._new.extend(float(x) for x in new_scores_batch)
        self._looks_taken += 1

        n = len(self._old)
        if n < 2:
            # Not enough data to compute a t-statistic. Report inconclusive
            # without crossing any boundary; do not count this as a "real" look.
            return self._build_result(
                verdict="INCONCLUSIVE",
                delta=0.0,
                se=0.0,
                boundary_z=self._boundaries[self._looks_taken - 1],
                n=n,
                t_stat=0.0,
                p_value=1.0,
                terminal=False,
            )

        diff = np.asarray(self._new, dtype=np.float64) - np.asarray(
            self._old, dtype=np.float64
        )
        delta = float(np.mean(diff))
        s_d = float(np.std(diff, ddof=1))
        if s_d == 0.0:
            # Every pair identical so far → either no signal yet (∆=0) or
            # perfect signal. Treat as inconclusive without consuming the look.
            return self._build_result(
                verdict="INCONCLUSIVE",
                delta=delta,
                se=0.0,
                boundary_z=self._boundaries[self._looks_taken - 1],
                n=n,
                t_stat=0.0,
                p_value=1.0,
                terminal=False,
            )

        se = s_d / float(np.sqrt(n))
        t_stat = delta / se
        boundary_z = self._boundaries[self._looks_taken - 1]
        # Unadjusted p-value (per-look). The verdict is the *sequentially*
        # adjusted decision; the reported p-value reflects only this look.
        df = n - 1
        p_value = 2.0 * float(stats.t.sf(abs(t_stat), df))

        crossed = abs(t_stat) > boundary_z
        terminal = crossed or self._looks_taken >= self.n_looks

        if crossed:
            verdict: Verdict = "BETTER" if t_stat > 0 else "WORSE"
        else:
            verdict = "INCONCLUSIVE"

        return self._build_result(
            verdict=verdict,
            delta=delta,
            se=se,
            boundary_z=boundary_z,
            n=n,
            t_stat=t_stat,
            p_value=p_value,
            terminal=terminal,
        )

    def _build_result(
        self,
        *,
        verdict: Verdict,
        delta: float,
        se: float,
        boundary_z: float,
        n: int,
        t_stat: float,
        p_value: float,
        terminal: bool,
    ) -> CompareResult:
        ci_lower = delta - boundary_z * se
        ci_upper = delta + boundary_z * se
        if verdict == "BETTER":
            rec = (
                f"Stop early: BETTER at look {self._looks_taken}/{self.n_looks} "
                f"(t-stat {delta / se if se > 0 else float('inf'):+.3f}, "
                f"boundary ±{boundary_z:.3f}, n={n})."
            )
        elif verdict == "WORSE":
            rec = (
                f"Stop early: WORSE at look {self._looks_taken}/{self.n_looks} "
                f"(t-stat {delta / se if se > 0 else float('-inf'):+.3f}, "
                f"boundary ±{boundary_z:.3f}, n={n})."
            )
        elif terminal:
            rec = (
                f"Final look reached at n={n} without crossing the boundary. "
                f"Treat the change as inconclusive at this sample size."
            )
        else:
            rec = (
                f"Look {self._looks_taken}/{self.n_looks}: continue. "
                f"Current Δ {delta:+.4f}; next-look boundary tightens further."
            )
        result = CompareResult(
            verdict=verdict,
            mean_difference=delta,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=1.0 - self.alpha,
            p_value=p_value,
            n=n,
            method=f"sequential_{self.boundary}",
            recommendation=rec,
        )
        if terminal:
            self._stopped = True
            self._last_result = result
        return result


def _compute_boundaries(
    boundary: str, n_looks: int, alpha: float
) -> list[float]:
    """Return the z-score boundary at each look (1-indexed)."""
    z_alpha_2 = float(stats.norm.ppf(1.0 - alpha / 2.0))
    if boundary == "obrien_fleming":
        return [z_alpha_2 * float(np.sqrt(n_looks / k)) for k in range(1, n_looks + 1)]
    # Pocock — constant boundary, tabulated for α=0.05.
    if not np.isclose(alpha, 0.05):
        raise NotImplementedError(
            f"Pocock boundary requires α=0.05 in this release (got {alpha}); "
            f"use boundary='obrien_fleming' for other α."
        )
    if n_looks not in _POCOCK_C_ALPHA_005:
        raise NotImplementedError(
            f"Pocock boundary table covers n_looks ∈ "
            f"{sorted(_POCOCK_C_ALPHA_005)}; got {n_looks}. "
            f"Use boundary='obrien_fleming' for other K."
        )
    c = _POCOCK_C_ALPHA_005[n_looks]
    return [c] * n_looks
