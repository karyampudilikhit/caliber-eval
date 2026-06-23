"""Property-based and coverage tests for `compare()`.

Two flavors:
    1. Hypothesis invariants — properties that must hold for ALL valid inputs.
    2. Statistical correctness — empirical CI coverage, FPR, and power, run
       across simulations for each method.

The session-2 critical tests (`test_compare.py::TestCriticalCICoverage` and
`::TestCriticalFalsePositiveRate`) cover the paired-t path. This file covers:
    - paired_bootstrap, with slightly lenient coverage thresholds because the
      percentile-method bootstrap is known to under-cover at small n.
    - the auto-selector's behavioral contract.
    - statistical power: BETTER fires at large n with a meaningful effect.
    - non-normal data: bootstrap CI still covers the truth.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from caliber import compare

# ----------------------------------------------------------------------------
# Hypothesis strategies
# ----------------------------------------------------------------------------


@st.composite
def paired_score_arrays(
    draw: st.DrawFn,
    *,
    min_n: int = 2,
    max_n: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a pair of float arrays plausibly resembling LLM eval scores.

    The two arrays may be identical; we nudge one element to keep `compare()`
    from rejecting the all-identical input.
    """
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    base_mean = draw(st.floats(min_value=0.0, max_value=1.0))
    noise = draw(st.floats(min_value=0.01, max_value=0.3))
    true_effect = draw(st.floats(min_value=-0.3, max_value=0.3))
    seed = draw(st.integers(min_value=0, max_value=10_000))

    rng = np.random.default_rng(seed)
    old = rng.normal(base_mean, noise, n)
    new = rng.normal(base_mean + true_effect, noise, n)
    if np.array_equal(new, old):
        new = new.copy()
        new[0] += 1e-6
    return old, new


# ----------------------------------------------------------------------------
# Invariants — must hold for any valid input
# ----------------------------------------------------------------------------


class TestInvariants:
    """Properties of CompareResult that hold for every well-formed input."""

    @given(arrays=paired_score_arrays())
    @settings(
        max_examples=40,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_result_is_well_formed(
        self, arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        old, new = arrays
        result = compare(old, new, n_bootstrap=500, seed=0)
        # Sample size matches input.
        assert result.n == len(old)
        # CI brackets the point estimate.
        assert result.ci_lower <= result.mean_difference <= result.ci_upper
        # Mean difference matches the obvious computation.
        expected = float(np.mean(new - old))
        assert abs(result.mean_difference - expected) < 1e-9
        # Verdict is one of the documented labels.
        assert result.verdict in {"BETTER", "WORSE", "INCONCLUSIVE", "NO_CHANGE"}
        # p-value is a probability.
        assert 0.0 <= result.p_value <= 1.0
        # The chosen method label is one of the documented values.
        assert result.method in {"paired_t", "paired_bootstrap"}

    @given(arrays=paired_score_arrays(min_n=10, max_n=50))
    @settings(
        max_examples=15,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_bootstrap_deterministic_with_seed(
        self, arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Same seed → byte-identical bootstrap result."""
        old, new = arrays
        r1 = compare(old, new, method="paired_bootstrap", n_bootstrap=500, seed=42)
        r2 = compare(old, new, method="paired_bootstrap", n_bootstrap=500, seed=42)
        assert r1.mean_difference == r2.mean_difference
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper
        assert r1.p_value == r2.p_value
        assert r1.verdict == r2.verdict

    @given(arrays=paired_score_arrays(min_n=10, max_n=50))
    @settings(
        max_examples=15,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_bootstrap_different_seeds_differ(
        self, arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Different seeds usually give different CIs (sanity check on RNG plumbing)."""
        old, new = arrays
        r1 = compare(old, new, method="paired_bootstrap", n_bootstrap=500, seed=1)
        r2 = compare(old, new, method="paired_bootstrap", n_bootstrap=500, seed=2)
        # Point estimate is deterministic; CI bounds differ across seeds.
        assert r1.mean_difference == r2.mean_difference
        # Allow occasional ties; require at least one bound to differ.
        differ = (r1.ci_lower != r2.ci_lower) or (r1.ci_upper != r2.ci_upper)
        assert differ


# ----------------------------------------------------------------------------
# Auto-selection behavioral contract
# ----------------------------------------------------------------------------


class TestAutoSelection:
    def test_auto_picks_bootstrap_under_30(self) -> None:
        rng = np.random.default_rng(0)
        old = rng.normal(0.5, 0.1, 20)
        new = rng.normal(0.55, 0.1, 20)
        result = compare(old, new, method="auto", seed=0)
        assert result.method == "paired_bootstrap"

    def test_auto_picks_paired_t_at_threshold(self) -> None:
        # n=30 is the boundary; normal data → paired-t.
        rng = np.random.default_rng(0)
        old = rng.normal(0.5, 0.1, 30)
        new = rng.normal(0.55, 0.1, 30)
        result = compare(old, new, method="auto")
        assert result.method == "paired_t"

    def test_auto_picks_paired_t_above_shapiro_threshold(self) -> None:
        # n > 5000 → skip Shapiro (unreliable at very large N), use paired-t.
        rng = np.random.default_rng(0)
        old = rng.normal(0.5, 0.1, 5001)
        new = rng.normal(0.51, 0.1, 5001)
        result = compare(old, new, method="auto")
        assert result.method == "paired_t"

    def test_auto_falls_back_to_bootstrap_for_clearly_non_normal(self) -> None:
        # Heavy outliers in a moderate-sized sample → Shapiro rejects → bootstrap.
        rng = np.random.default_rng(0)
        old = rng.normal(0.5, 0.01, 50)
        new = old.copy()
        new[:5] += 5.0  # extreme outliers in 10% of pairs
        result = compare(old, new, method="auto", seed=0)
        assert result.method == "paired_bootstrap"


# ----------------------------------------------------------------------------
# Bootstrap statistical correctness — coverage and FPR
# ----------------------------------------------------------------------------


class TestBootstrapCoverage:
    """Bootstrap percentile CI undercovers slightly at small n; ≥90% is the bar."""

    def test_bootstrap_ci_under_h0_small_n(self) -> None:
        """True Δ=0, n=20 — bootstrap percentile CI covers 0 in ≥90% of sims."""
        rng = np.random.default_rng(42)
        n_sims = 500
        n_per_sim = 20
        true_delta = 0.0
        covers = 0
        for i in range(n_sims):
            old = rng.normal(0.7, 0.1, n_per_sim)
            new = rng.normal(0.7 + true_delta, 0.1, n_per_sim)
            r = compare(
                old,
                new,
                method="paired_bootstrap",
                n_bootstrap=2000,
                seed=i,
            )
            if r.ci_lower <= true_delta <= r.ci_upper:
                covers += 1
        coverage = covers / n_sims
        assert coverage >= 0.90, (
            f"Bootstrap CI coverage (n=20, Δ=0) was {coverage:.3f}; "
            f"percentile method ≥0.90 expected"
        )

    def test_bootstrap_ci_with_real_effect(self) -> None:
        """True Δ=0.05, n=50 — bootstrap covers 0.05 in ≥92% of sims."""
        rng = np.random.default_rng(123)
        n_sims = 500
        n_per_sim = 50
        true_delta = 0.05
        covers = 0
        for i in range(n_sims):
            old = rng.normal(0.7, 0.1, n_per_sim)
            new = rng.normal(0.7 + true_delta, 0.1, n_per_sim)
            r = compare(
                old,
                new,
                method="paired_bootstrap",
                n_bootstrap=2000,
                seed=i,
            )
            if r.ci_lower <= true_delta <= r.ci_upper:
                covers += 1
        coverage = covers / n_sims
        assert coverage >= 0.92, (
            f"Bootstrap CI coverage (n=50, Δ=0.05) was {coverage:.3f}; "
            f"want ≥0.92"
        )


class TestBootstrapFalsePositiveRate:
    """Under H₀, bootstrap should fire BETTER/WORSE at no more than ~10% (n=20)."""

    def test_fpr_small_n(self) -> None:
        rng = np.random.default_rng(7)
        n_sims = 500
        n_per_sim = 20
        false_positives = 0
        for i in range(n_sims):
            old = rng.normal(0.7, 0.1, n_per_sim)
            new = rng.normal(0.7, 0.1, n_per_sim)
            r = compare(
                old,
                new,
                method="paired_bootstrap",
                n_bootstrap=2000,
                seed=i,
            )
            if r.verdict in ("BETTER", "WORSE"):
                false_positives += 1
        fpr = false_positives / n_sims
        # Percentile bootstrap at n=20 has FPR roughly 7-10% (the inverse of
        # the ~90% coverage). 0.10 is the empirical ceiling we allow.
        assert fpr <= 0.10, (
            f"Bootstrap FPR (n=20) was {fpr:.3f}; percentile method bound ≤0.10"
        )


# ----------------------------------------------------------------------------
# Statistical power — BETTER fires when there is a real effect
# ----------------------------------------------------------------------------


class TestPower:
    def test_better_with_05_sigma_effect_large_n(self) -> None:
        """True Δ = 0.5σ at n=200 — BETTER fires in ≥95% of sims (paired-t path)."""
        rng = np.random.default_rng(0)
        n_sims = 200
        n_per_sim = 200
        sigma = 0.1
        true_delta = 0.5 * sigma
        better = 0
        for _ in range(n_sims):
            old = rng.normal(0.7, sigma, n_per_sim)
            new = rng.normal(0.7 + true_delta, sigma, n_per_sim)
            r = compare(old, new)
            if r.verdict == "BETTER":
                better += 1
        power = better / n_sims
        assert power >= 0.95, (
            f"Power at Δ=0.5σ, n=200 was {power:.3f}; expected ≥0.95"
        )

    def test_worse_with_negative_effect_large_n(self) -> None:
        rng = np.random.default_rng(0)
        n_sims = 200
        n_per_sim = 200
        sigma = 0.1
        true_delta = -0.5 * sigma
        worse = 0
        for _ in range(n_sims):
            old = rng.normal(0.7, sigma, n_per_sim)
            new = rng.normal(0.7 + true_delta, sigma, n_per_sim)
            r = compare(old, new)
            if r.verdict == "WORSE":
                worse += 1
        power = worse / n_sims
        assert power >= 0.95, (
            f"Power at Δ=-0.5σ, n=200 was {power:.3f}; expected ≥0.95"
        )


# ----------------------------------------------------------------------------
# Non-normal data — bootstrap is the right tool here
# ----------------------------------------------------------------------------


class TestNonNormalData:
    """Bootstrap CI should cover the truth even when paired-t's normality
    assumption is badly violated."""

    def test_bootstrap_coverage_with_heavy_tailed_diffs(self) -> None:
        """Differences drawn from a t(df=3) — heavy-tailed, paired-t would
        under-cover. Bootstrap CI should still cover Δ=0 ≥85% of the time at n=30."""
        rng = np.random.default_rng(11)
        n_sims = 400
        n_per_sim = 30
        covers = 0
        for i in range(n_sims):
            # Build diffs that are heavy-tailed but centred at zero.
            old = rng.normal(0.7, 0.05, n_per_sim)
            heavy_noise = rng.standard_t(df=3, size=n_per_sim) * 0.1
            new = old + heavy_noise
            r = compare(
                old,
                new,
                method="paired_bootstrap",
                n_bootstrap=2000,
                seed=i,
            )
            if r.ci_lower <= 0.0 <= r.ci_upper:
                covers += 1
        coverage = covers / n_sims
        # Heavy tails knock coverage down; 0.85 is the empirical floor.
        assert coverage >= 0.85, (
            f"Bootstrap coverage on t(3) noise was {coverage:.3f}; want ≥0.85"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_bootstrap_finds_real_effect_in_heavy_tailed_data(
        self, seed: int
    ) -> None:
        """With heavy-tailed noise and a strong real effect, bootstrap finds it."""
        rng = np.random.default_rng(seed)
        n = 100
        old = rng.normal(0.5, 0.05, n)
        new = old + 0.2 + rng.standard_t(df=3, size=n) * 0.05
        r = compare(old, new, method="paired_bootstrap", n_bootstrap=2000, seed=seed)
        assert r.verdict == "BETTER"
