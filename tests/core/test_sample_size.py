"""Tests for caliber.core.sample_size.

The PRD's critical test is at the bottom: simulate at the returned n and verify
the empirical rejection rate matches the targeted power within ±5%.
"""

from __future__ import annotations

import numpy as np
import pytest

from caliber import SampleSizeResult, compare, sample_size


class TestValidation:
    def test_zero_effect_raises(self) -> None:
        with pytest.raises(ValueError, match="nonzero"):
            sample_size(0.0)

    def test_negative_effect_ok(self) -> None:
        # Direction doesn't matter — we standardise on |effect|.
        r = sample_size(-0.5)
        assert r.effect_size == 0.5

    def test_invalid_power_raises(self) -> None:
        with pytest.raises(ValueError, match="power"):
            sample_size(0.5, power=1.5)

    def test_invalid_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            sample_size(0.5, confidence_level=0.0)

    def test_absolute_without_baseline_raises(self) -> None:
        with pytest.raises(ValueError, match="baseline_std"):
            sample_size(0.05, effect_type="absolute")

    def test_absolute_with_zero_baseline_raises(self) -> None:
        with pytest.raises(ValueError, match="baseline_std"):
            sample_size(0.05, effect_type="absolute", baseline_std=0.0)

    def test_absolute_for_proportion_raises(self) -> None:
        with pytest.raises(ValueError, match="proportion"):
            sample_size(
                0.05,
                effect_type="absolute",
                baseline_std=0.5,
                test="proportion",
            )


class TestReturnShape:
    def test_returns_sample_size_result(self) -> None:
        r = sample_size(0.5)
        assert isinstance(r, SampleSizeResult)

    def test_n_is_positive_integer(self) -> None:
        r = sample_size(0.5)
        assert isinstance(r.n_per_arm, int)
        assert r.n_per_arm >= 2

    def test_effect_size_stored_as_cohens_d(self) -> None:
        r = sample_size(
            0.05, effect_type="absolute", baseline_std=0.1
        )
        # Cohen's d = 0.05 / 0.1 = 0.5
        assert r.effect_size == pytest.approx(0.5)

    def test_power_and_confidence_stored(self) -> None:
        r = sample_size(0.5, power=0.9, confidence_level=0.99)
        assert r.power == 0.9
        assert r.confidence_level == 0.99


class TestEquivalence:
    """Cohen's d input and equivalent absolute input give the same n."""

    def test_d_and_absolute_match(self) -> None:
        r_d = sample_size(0.5, effect_type="cohens_d")
        r_abs = sample_size(
            0.05, effect_type="absolute", baseline_std=0.1
        )
        assert r_d.n_per_arm == r_abs.n_per_arm

    def test_doubling_effect_quarters_n_approximately(self) -> None:
        # n scales as 1/d², so doubling d should give roughly n/4.
        r_small = sample_size(0.25)
        r_big = sample_size(0.5)
        # Allow ±10% slack for the t-distribution correction at small n.
        ratio = r_small.n_per_arm / r_big.n_per_arm
        assert 3.5 <= ratio <= 4.5


class TestTestFamilies:
    def test_paired_t_default(self) -> None:
        r = sample_size(0.5)
        assert r.n_per_arm > 0

    def test_unpaired_t_needs_more(self) -> None:
        # Unpaired (independent two-sample) t-test needs more n per arm than
        # paired for the same Cohen's d, because the paired form removes
        # per-example variance.
        r_paired = sample_size(0.5, test="paired_t")
        r_unpaired = sample_size(0.5, test="unpaired_t")
        assert r_unpaired.n_per_arm > r_paired.n_per_arm

    def test_proportion_runs(self) -> None:
        # Cohen's h = 0.3 — a moderate proportion effect.
        r = sample_size(0.3, test="proportion")
        assert r.n_per_arm > 0


class TestHigherPowerRequiresMoreN:
    def test_higher_power_means_more_samples(self) -> None:
        r_80 = sample_size(0.5, power=0.8)
        r_95 = sample_size(0.5, power=0.95)
        assert r_95.n_per_arm > r_80.n_per_arm


# ----------------------------------------------------------------------------
# CRITICAL: power-recovery test.
#
# Per the PRD: "Sample-size sanity: returns required n that, when used,
# produces the targeted power within ±5%."
#
# We compute n for d=0.5 / power=0.8, then simulate 1000 paired comparisons
# at that n and verify the empirical fraction of BETTER/WORSE verdicts is
# 0.80 ± 0.05.
# ----------------------------------------------------------------------------


class TestCriticalPowerRecovery:
    def test_paired_t_power_recovery(self) -> None:
        """Compute n for d=0.5/power=0.8; simulate and verify ~80% rejection."""
        r = sample_size(0.5, power=0.8, test="paired_t")
        n = r.n_per_arm

        rng = np.random.default_rng(0)
        n_sims = 1000
        rejections = 0
        for _ in range(n_sims):
            # Differences directly drawn with mean 0.5, std 1.0 → Cohen's d = 0.5.
            # We feed (zeros, diffs) to compare so the paired-t test sees
            # exactly these differences.
            diff = rng.normal(0.5, 1.0, n)
            old = np.zeros(n)
            new = old + diff
            result = compare(old, new, method="paired_t")
            if result.verdict in ("BETTER", "WORSE"):
                rejections += 1
        empirical_power = rejections / n_sims

        assert abs(empirical_power - 0.8) <= 0.05, (
            f"Empirical power {empirical_power:.3f} differs from target 0.80 "
            f"by more than 0.05 at n={n}"
        )

    def test_paired_t_power_recovery_high_power(self) -> None:
        """Same check at power=0.9 — verify the higher target is achieved."""
        r = sample_size(0.5, power=0.9, test="paired_t")
        n = r.n_per_arm

        rng = np.random.default_rng(7)
        n_sims = 1000
        rejections = 0
        for _ in range(n_sims):
            diff = rng.normal(0.5, 1.0, n)
            old = np.zeros(n)
            new = old + diff
            result = compare(old, new, method="paired_t")
            if result.verdict in ("BETTER", "WORSE"):
                rejections += 1
        empirical_power = rejections / n_sims

        assert abs(empirical_power - 0.9) <= 0.05, (
            f"Empirical power {empirical_power:.3f} differs from target 0.90 "
            f"by more than 0.05 at n={n}"
        )


class TestCompareIntegration:
    """compare() uses sample_size() internally when an INCONCLUSIVE result
    is paired with a target_effect."""

    def test_target_effect_routes_through_sample_size(self) -> None:
        rng = np.random.default_rng(0)
        old = rng.normal(0.7, 0.1, 10)
        new = rng.normal(0.71, 0.1, 10)
        r = compare(old, new, target_effect=0.02)
        if r.verdict == "INCONCLUSIVE":
            assert r.sample_size_needed is not None
            # Sanity: matches what sample_size() would return directly with
            # the observed std as baseline.
            observed_std = float(np.std(new - old, ddof=1))
            expected = sample_size(
                0.02,
                effect_type="absolute",
                baseline_std=observed_std,
                power=0.8,
            ).n_per_arm
            assert r.sample_size_needed == expected
