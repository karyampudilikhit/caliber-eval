"""Paired-sample comparison for eval scores — entry point: `compare()`.

Two underlying methods are available. The paired t-test treats `new[i] - old[i]`
as a single sample and asks whether its mean is distinguishable from zero. The
paired bootstrap resamples those differences with replacement and reads the CI
off the empirical distribution of resampled means.

Both methods use the paired form because the same eval set was scored under
two versions (the standard A/B setup for LLM evals). The paired form is
strictly more powerful than the unpaired version: it removes per-example
variance from the noise floor.

Auto-selection (`method="auto"`):
    n < 30                              → paired_bootstrap   (CLT too weak)
    30 ≤ n ≤ 5000, Shapiro p < 0.001    → paired_bootstrap   (clearly non-normal)
    30 ≤ n ≤ 5000, Shapiro p ≥ 0.001    → paired_t
    n > 5000                            → paired_t           (CLT dominates;
                                                              Shapiro unreliable)

The Shapiro threshold is strict (0.001 not 0.05) on purpose — we only flip to
the slower bootstrap when there is *strong* evidence that paired-t's normality
assumption is wrong. For moderately non-normal data with n ≥ 30, the CLT makes
paired-t essentially correct.

References
----------
.. [1] Student. "The probable error of a mean." Biometrika 6 (1908): 1-25.
.. [2] Efron, B. & Tibshirani, R.J. *An Introduction to the Bootstrap* (1993).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy import stats

from caliber.core._bootstrap import paired_bootstrap_ci
from caliber.core.types import CompareResult, Verdict

ArrayLike = Sequence[float] | np.ndarray

# Auto-selection thresholds. Documented here so users can audit the heuristic.
_AUTO_MIN_N_FOR_T = 30          # below this, always bootstrap (CLT too weak)
_AUTO_MAX_N_FOR_SHAPIRO = 5000  # above this, Shapiro is unreliable, CLT dominates
_AUTO_SHAPIRO_ALPHA = 0.001     # only switch to bootstrap on STRONG non-normality


def compare(
    old_scores: ArrayLike,
    new_scores: ArrayLike,
    *,
    confidence_level: float = 0.95,
    method: Literal["paired_bootstrap", "paired_t", "auto"] = "auto",
    n_bootstrap: int = 10_000,
    seed: int | None = None,
    metric_name: str = "score",
    practical_threshold: float = 0.0,
    target_effect: float | None = None,
) -> CompareResult:
    """Decide whether `new_scores` is statistically distinguishable from `old_scores`.

    The two arrays must be **paired** — `old_scores[i]` and `new_scores[i]` must
    correspond to the same eval example. This is the standard setup for comparing
    two versions of a prompt/model/agent against a fixed eval set.

    Parameters
    ----------
    old_scores, new_scores : array-like of float, shape (n,)
        Paired score arrays. Same length, both non-empty, not all-identical.
    confidence_level : float, default 0.95
        Two-sided CI level. The CI excludes 0 ⇒ verdict BETTER or WORSE.
    method : {"paired_t", "paired_bootstrap", "auto"}, default "auto"
        Estimator. "auto" selects paired-t when n ≥ 30 and the paired differences
        are not clearly non-normal; otherwise it falls back to the bootstrap.
        See the module docstring for the full decision rule.
    n_bootstrap : int, default 10_000
        Number of bootstrap resamples (used only by the bootstrap path).
    seed : int | None, default None
        Seed for the bootstrap RNG. None → fresh non-deterministic state.
    metric_name : str, default "score"
        Used in the human-readable recommendation.
    practical_threshold : float, default 0.0
        If `|mean_difference| < practical_threshold` the verdict is `NO_CHANGE`
        regardless of statistical significance. Use this to encode "effects
        below X aren't worth shipping for" — e.g. a 0.3% accuracy gain that
        wouldn't justify the rollout cost.
    target_effect : float | None, default None
        If the verdict is INCONCLUSIVE, this is the raw-units effect size you
        wanted to detect. Caliber will populate `sample_size_needed` with the
        required n-per-arm to detect it at 80% power.

    Returns
    -------
    CompareResult
        Verdict + CI + p-value + recommendation. See `caliber.core.types`.

    Raises
    ------
    ValueError
        If the inputs are different lengths, empty, contain NaN/Inf, have fewer
        than 2 observations, or are element-wise identical (no signal).

    Examples
    --------
    >>> import numpy as np, caliber
    >>> rng = np.random.default_rng(0)
    >>> old = rng.normal(0.7, 0.05, 100)
    >>> new = rng.normal(0.75, 0.05, 100)
    >>> r = caliber.compare(old, new)
    >>> r.verdict
    'BETTER'

    Notes
    -----
    Paired t-test formula:
        Δ = mean(new - old)
        s = std(new - old, ddof=1)
        SE = s / sqrt(n)
        CI = Δ ± t_{α/2, n-1} · SE
        t-stat = Δ / SE,  df = n - 1
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be in (0, 1); got {confidence_level}"
        )

    old = _validate_array(old_scores, "old_scores")
    new = _validate_array(new_scores, "new_scores")

    if len(old) != len(new):
        raise ValueError(
            f"old_scores and new_scores must have the same length "
            f"(got {len(old)} and {len(new)})"
        )
    if len(old) < 2:
        raise ValueError(
            f"Need at least 2 paired observations to compute a t-statistic; got {len(old)}"
        )

    diff = new - old
    if np.all(diff == 0.0):
        raise ValueError(
            "old_scores and new_scores are identical at every index — no signal to compare."
        )

    chosen_method = _select_method(diff) if method == "auto" else method

    if chosen_method == "paired_t":
        delta, ci_lower, ci_upper, p_value = _paired_t_ci(diff, confidence_level)
    else:
        delta, ci_lower, ci_upper, p_value = paired_bootstrap_ci(
            diff,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

    verdict, sample_size_needed = _decide_verdict(
        delta=delta,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        diff=diff,
        confidence_level=confidence_level,
        practical_threshold=practical_threshold,
        target_effect=target_effect,
    )

    recommendation = _build_recommendation(
        verdict=verdict,
        delta=delta,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=len(old),
        metric_name=metric_name,
        practical_threshold=practical_threshold,
        target_effect=target_effect,
        sample_size_needed=sample_size_needed,
    )

    return CompareResult(
        verdict=verdict,
        mean_difference=float(delta),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        confidence_level=confidence_level,
        p_value=float(p_value),
        n=len(old),
        method=chosen_method,
        sample_size_needed=sample_size_needed,
        recommendation=recommendation,
    )


def _validate_array(x: ArrayLike, name: str) -> np.ndarray:
    """Coerce input to a 1-D float64 ndarray; reject empty / non-finite / multi-dim."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional; got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def _select_method(diff: np.ndarray) -> Literal["paired_t", "paired_bootstrap"]:
    """Choose between paired-t and paired-bootstrap for ``method='auto'``.

    Decision rule (see module docstring for rationale):
        n < 30                              → bootstrap (CLT too weak)
        n > 5000                            → paired-t (CLT dominates;
                                              Shapiro-Wilk unreliable)
        30 ≤ n ≤ 5000, Shapiro p < 0.001    → bootstrap (clearly non-normal)
        otherwise                           → paired-t
    """
    n = len(diff)
    if n < _AUTO_MIN_N_FOR_T:
        return "paired_bootstrap"
    if n > _AUTO_MAX_N_FOR_SHAPIRO:
        return "paired_t"
    p_normal = float(stats.shapiro(diff).pvalue)
    if p_normal < _AUTO_SHAPIRO_ALPHA:
        return "paired_bootstrap"
    return "paired_t"


def _paired_t_ci(
    diff: np.ndarray, confidence_level: float
) -> tuple[float, float, float, float]:
    """Return (delta, ci_lower, ci_upper, p_value) for the paired t-test.

    Handles the degenerate-zero-variance case: if every paired difference equals
    the same nonzero constant, the t-stat is +∞ — we return p=0 and a zero-width
    CI at delta. (`diff` of all zeros is rejected earlier in `compare`.)
    """
    n = len(diff)
    delta = float(np.mean(diff))
    s_d = float(np.std(diff, ddof=1))

    if s_d == 0.0:
        # Every diff identical and nonzero → perfect signal, degenerate CI.
        return delta, delta, delta, 0.0

    se = s_d / np.sqrt(n)
    df = n - 1
    alpha = 1.0 - confidence_level
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    ci_lower = delta - t_crit * se
    ci_upper = delta + t_crit * se
    t_stat = delta / se
    # Two-sided p-value. Use sf(|t|) instead of 1 - cdf(|t|) for numerical
    # accuracy in the tail.
    p_value = 2.0 * float(stats.t.sf(abs(t_stat), df))
    return delta, ci_lower, ci_upper, p_value


def _decide_verdict(
    *,
    delta: float,
    ci_lower: float,
    ci_upper: float,
    diff: np.ndarray,
    confidence_level: float,
    practical_threshold: float,
    target_effect: float | None,
) -> tuple[Verdict, int | None]:
    """Apply the PRD verdict ladder: NO_CHANGE > {BETTER,WORSE} > INCONCLUSIVE."""
    if abs(delta) < practical_threshold:
        return "NO_CHANGE", None
    if ci_lower > 0.0:
        return "BETTER", None
    if ci_upper < 0.0:
        return "WORSE", None

    # INCONCLUSIVE. Compute sample_size_needed if the user asked.
    sample_size_needed: int | None = None
    if target_effect is not None and target_effect != 0.0:
        observed_std = float(np.std(diff, ddof=1))
        if observed_std > 0.0:
            from caliber.core.sample_size import sample_size

            ss = sample_size(
                effect_size=abs(target_effect),
                effect_type="absolute",
                baseline_std=observed_std,
                power=0.8,
                confidence_level=confidence_level,
                test="paired_t",
            )
            sample_size_needed = ss.n_per_arm
    return "INCONCLUSIVE", sample_size_needed


def _build_recommendation(
    *,
    verdict: Verdict,
    delta: float,
    ci_lower: float,
    ci_upper: float,
    n: int,
    metric_name: str,
    practical_threshold: float,
    target_effect: float | None,
    sample_size_needed: int | None,
) -> str:
    """Plain-English next step for the engineer reading the result."""
    ci_str = f"95% CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]"
    if verdict == "BETTER":
        return (
            f"Ship the new version. {metric_name} improved by {delta:+.4f} "
            f"({ci_str}, n={n}). The interval excludes zero."
        )
    if verdict == "WORSE":
        return (
            f"Do not ship. {metric_name} regressed by {delta:+.4f} "
            f"({ci_str}, n={n}). The interval excludes zero."
        )
    if verdict == "NO_CHANGE":
        return (
            f"No practical change. {metric_name} moved by {delta:+.4f}, within "
            f"the practical threshold of ±{practical_threshold} — below the "
            f"smallest effect you said is worth shipping for."
        )
    # INCONCLUSIVE
    base = (
        f"Inconclusive. {metric_name} changed by {delta:+.4f} but {ci_str} "
        f"includes zero — too noisy at n={n} to call."
    )
    if sample_size_needed is not None and target_effect is not None:
        return base + (
            f" Collect ~{sample_size_needed} paired samples to detect a "
            f"difference of {abs(target_effect):.4f} with 80% power."
        )
    return base + " Collect more samples, or set `target_effect` to estimate how many."
