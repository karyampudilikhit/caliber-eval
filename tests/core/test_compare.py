"""Unit, behavioral, and critical statistical tests for caliber.core.compare.

The two critical tests at the bottom (CI coverage, false-positive rate under H₀)
are non-negotiable per the build PRD: they prove the math is correct in
aggregate. Without them passing the library has no claim to "statistical rigor."
"""

from __future__ import annotations

import numpy as np
import pytest

from caliber import CompareResult, compare


class TestValidation:
    def test_different_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compare([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_empty_old_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            compare([], [1.0, 2.0])

    def test_empty_both_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            compare([], [])

    def test_single_observation_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            compare([0.5], [0.6])

    def test_all_identical_raises(self) -> None:
        with pytest.raises(ValueError, match="identical"):
            compare([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            compare([0.1, 0.2, np.nan], [0.1, 0.2, 0.3])

    def test_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN or Inf"):
            compare([0.1, 0.2, 0.3], [0.1, 0.2, np.inf])

    def test_two_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="1-dimensional"):
            compare(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_invalid_confidence_low_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            compare([1.0, 2.0, 3.0], [1.0, 2.0, 3.1], confidence_level=0.0)

    def test_invalid_confidence_high_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            compare([1.0, 2.0, 3.0], [1.0, 2.0, 3.1], confidence_level=1.5)


class TestReturnShape:
    def test_returns_compare_result(self) -> None:
        result = compare([0.5, 0.6, 0.7], [0.6, 0.7, 0.8])
        assert isinstance(result, CompareResult)

    def test_n_is_sample_count(self) -> None:
        result = compare([0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5])
        assert result.n == 4

    def test_explicit_paired_t_is_paired_t(self) -> None:
        result = compare([0.1, 0.2, 0.3], [0.2, 0.3, 0.4], method="paired_t")
        assert result.method == "paired_t"

    def test_explicit_bootstrap_is_bootstrap(self) -> None:
        result = compare(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            method="paired_bootstrap",
            n_bootstrap=500,
            seed=0,
        )
        assert result.method == "paired_bootstrap"

    def test_auto_picks_bootstrap_at_small_n(self) -> None:
        # n=5 < 30 → bootstrap.
        result = compare(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            method="auto",
            seed=0,
        )
        assert result.method == "paired_bootstrap"

    def test_auto_picks_paired_t_at_large_normal_n(self) -> None:
        rng = np.random.default_rng(0)
        old = rng.normal(0.5, 0.1, 50)
        new = rng.normal(0.55, 0.1, 50)
        result = compare(old, new, method="auto")
        assert result.method == "paired_t"

    def test_ci_property_returns_tuple(self) -> None:
        result = compare([0.1, 0.2, 0.3], [0.2, 0.3, 0.4])
        assert result.ci == (result.ci_lower, result.ci_upper)

    def test_ci_lower_below_upper(self) -> None:
        rng = np.random.default_rng(0)
        old = rng.normal(0.5, 0.1, 30)
        new = rng.normal(0.55, 0.1, 30)
        result = compare(old, new)
        assert result.ci_lower <= result.ci_upper

    def test_confidence_level_stored(self) -> None:
        result = compare([0.1, 0.2, 0.3], [0.2, 0.3, 0.4], confidence_level=0.99)
        assert result.confidence_level == 0.99

    def test_recommendation_nonempty(self) -> None:
        result = compare([0.1, 0.2, 0.3], [0.2, 0.3, 0.4])
        assert len(result.recommendation) > 20


class TestVerdict:
    def test_clear_better(self) -> None:
        rng = np.random.default_rng(0)
        old = rng.normal(0.6, 0.05, 100)
        new = rng.normal(0.8, 0.05, 100)
        result = compare(old, new)
        assert result.verdict == "BETTER"
        assert result.ci_lower > 0
        assert result.p_value < 0.001

    def test_clear_worse(self) -> None:
        rng = np.random.default_rng(0)
        old = rng.normal(0.8, 0.05, 100)
        new = rng.normal(0.6, 0.05, 100)
        result = compare(old, new)
        assert result.verdict == "WORSE"
        assert result.ci_upper < 0
        assert result.p_value < 0.001

    def test_inconclusive_when_noisy_small_n(self) -> None:
        rng = np.random.default_rng(0)
        old = rng.normal(0.7, 0.3, 20)
        new = rng.normal(0.7, 0.3, 20)
        result = compare(old, new)
        assert result.verdict == "INCONCLUSIVE"
        assert result.ci_lower < 0 < result.ci_upper

    def test_practical_threshold_yields_no_change(self) -> None:
        # A statistically significant but practically trivial effect should
        # be NO_CHANGE if the threshold dominates.
        rng = np.random.default_rng(0)
        old = rng.normal(0.700, 0.005, 200)
        new = rng.normal(0.703, 0.005, 200)
        result = compare(old, new, practical_threshold=0.05)
        assert result.verdict == "NO_CHANGE"

    def test_practical_threshold_does_not_hide_real_effect(self) -> None:
        # When the effect *exceeds* the threshold, BETTER/WORSE wins.
        rng = np.random.default_rng(0)
        old = rng.normal(0.6, 0.02, 100)
        new = rng.normal(0.8, 0.02, 100)
        result = compare(old, new, practical_threshold=0.05)
        assert result.verdict == "BETTER"

    def test_target_effect_populates_sample_size_needed(self) -> None:
        rng = np.random.default_rng(0)
        old = rng.normal(0.7, 0.1, 10)
        new = rng.normal(0.71, 0.1, 10)
        result = compare(old, new, target_effect=0.02)
        if result.verdict == "INCONCLUSIVE":
            assert result.sample_size_needed is not None
            assert result.sample_size_needed > 10

    def test_target_effect_ignored_when_decisive(self) -> None:
        rng = np.random.default_rng(0)
        old = rng.normal(0.6, 0.05, 200)
        new = rng.normal(0.8, 0.05, 200)
        result = compare(old, new, target_effect=0.02)
        assert result.verdict == "BETTER"
        assert result.sample_size_needed is None


class TestRecommendation:
    def test_better_recommendation_mentions_ship(self) -> None:
        rng = np.random.default_rng(0)
        old = rng.normal(0.6, 0.05, 100)
        new = rng.normal(0.8, 0.05, 100)
        result = compare(old, new, metric_name="accuracy")
        assert "Ship" in result.recommendation
        assert "accuracy" in result.recommendation

    def test_worse_recommendation_mentions_do_not_ship(self) -> None:
        rng = np.random.default_rng(0)
        old = rng.normal(0.8, 0.05, 100)
        new = rng.normal(0.6, 0.05, 100)
        result = compare(old, new)
        assert "Do not ship" in result.recommendation

    def test_inconclusive_recommendation_suggests_more_samples(self) -> None:
        rng = np.random.default_rng(0)
        old = rng.normal(0.7, 0.3, 20)
        new = rng.normal(0.7, 0.3, 20)
        result = compare(old, new)
        rec = result.recommendation.lower()
        assert "more" in rec or "collect" in rec


class TestDegenerate:
    def test_near_constant_diff_is_near_perfect_signal(self) -> None:
        # Every paired diff is +0.1 in real numbers; float64 rounding gives
        # a tiny but nonzero variance, so we don't hit the exact s_d==0 branch.
        # Pinned to paired_t — bootstrap with finite B can never give p<1/(B+1).
        old = [0.5, 0.6, 0.7, 0.8]
        new = [0.6, 0.7, 0.8, 0.9]
        result = compare(old, new, method="paired_t")
        assert result.verdict == "BETTER"
        assert result.p_value < 1e-30
        assert result.ci_lower == pytest.approx(0.1, abs=1e-6)
        assert result.ci_upper == pytest.approx(0.1, abs=1e-6)

    def test_exact_constant_diff_is_perfect_signal(self) -> None:
        # Differences EXACTLY constant in float64 → hits the s_d==0 branch in
        # paired-t. Pinned to paired_t because bootstrap can't represent p=0.
        old = np.array([0.0, 0.0, 0.0, 0.0])
        new = np.array([0.1, 0.1, 0.1, 0.1])
        result = compare(old, new, method="paired_t")
        assert result.verdict == "BETTER"
        assert result.p_value == 0.0
        assert result.ci_lower == result.ci_upper == result.mean_difference


# ----------------------------------------------------------------------------
# CRITICAL STATISTICAL TESTS — these prove the math is correct.
#
# Per the PRD: "If any of these tests fail, the bug is critical — do not ship."
#
# We run 1000 paired simulations and check:
#   1. 95% CIs cover the true Δ at least 94% of the time.
#   2. Under H₀ (true Δ = 0), BETTER/WORSE verdicts fire at most 6% of the time
#      (the nominal 5% bound + a small slack for sampling variation).
# ----------------------------------------------------------------------------


class TestCriticalCICoverage:
    """Frequentist coverage: 95% CIs must cover the true mean ≥94% of the time."""

    def test_ci_coverage_under_h0(self) -> None:
        """True Δ = 0 → 95% CI covers 0 in ≥94% of simulations."""
        rng = np.random.default_rng(42)
        n_sims = 1000
        n_per_sim = 50
        true_delta = 0.0
        covers = 0
        for _ in range(n_sims):
            old = rng.normal(0.7, 0.1, n_per_sim)
            new = rng.normal(0.7 + true_delta, 0.1, n_per_sim)
            r = compare(old, new)
            if r.ci_lower <= true_delta <= r.ci_upper:
                covers += 1
        coverage = covers / n_sims
        assert coverage >= 0.94, (
            f"CI coverage under H₀ was {coverage:.3f}; nominal 95% requires ≥0.94"
        )

    def test_ci_coverage_with_real_effect(self) -> None:
        """True Δ = 0.05 → 95% CI covers 0.05 in ≥94% of simulations."""
        rng = np.random.default_rng(123)
        n_sims = 1000
        n_per_sim = 50
        true_delta = 0.05
        covers = 0
        for _ in range(n_sims):
            old = rng.normal(0.7, 0.1, n_per_sim)
            new = rng.normal(0.7 + true_delta, 0.1, n_per_sim)
            r = compare(old, new)
            if r.ci_lower <= true_delta <= r.ci_upper:
                covers += 1
        coverage = covers / n_sims
        assert coverage >= 0.94, (
            f"CI coverage with Δ=0.05 was {coverage:.3f}; nominal 95% requires ≥0.94"
        )

    def test_ci_coverage_with_negative_effect(self) -> None:
        """True Δ = -0.05 → 95% CI covers -0.05 in ≥94% of simulations."""
        rng = np.random.default_rng(456)
        n_sims = 1000
        n_per_sim = 50
        true_delta = -0.05
        covers = 0
        for _ in range(n_sims):
            old = rng.normal(0.7, 0.1, n_per_sim)
            new = rng.normal(0.7 + true_delta, 0.1, n_per_sim)
            r = compare(old, new)
            if r.ci_lower <= true_delta <= r.ci_upper:
                covers += 1
        coverage = covers / n_sims
        assert coverage >= 0.94, (
            f"CI coverage with Δ=-0.05 was {coverage:.3f}; nominal 95% requires ≥0.94"
        )


class TestCriticalFalsePositiveRate:
    """Under H₀, BETTER/WORSE verdicts must fire ≤6% of the time."""

    def test_fpr_under_h0(self) -> None:
        rng = np.random.default_rng(7)
        n_sims = 1000
        n_per_sim = 50
        false_positives = 0
        for _ in range(n_sims):
            old = rng.normal(0.7, 0.1, n_per_sim)
            new = rng.normal(0.7, 0.1, n_per_sim)
            r = compare(old, new)
            if r.verdict in ("BETTER", "WORSE"):
                false_positives += 1
        fpr = false_positives / n_sims
        assert fpr <= 0.06, (
            f"False-positive rate under H₀ was {fpr:.3f}; "
            f"5% nominal bound + sampling slack requires ≤0.06"
        )
