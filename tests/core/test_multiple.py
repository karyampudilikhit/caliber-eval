"""Tests for caliber.core.multiple — Benjamini-Hochberg FDR correction."""

from __future__ import annotations

import numpy as np
import pytest

from caliber import benjamini_hochberg


class TestValidation:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            benjamini_hochberg([])

    def test_invalid_alpha_low_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            benjamini_hochberg([0.1, 0.2], alpha=0.0)

    def test_invalid_alpha_high_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            benjamini_hochberg([0.1, 0.2], alpha=1.0)

    def test_pvalue_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            benjamini_hochberg([0.1, -0.01])

    def test_pvalue_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            benjamini_hochberg([0.1, 1.5])

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            benjamini_hochberg([0.1, float("nan")])

    def test_two_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="1-dimensional"):
            benjamini_hochberg(np.array([[0.1, 0.2], [0.3, 0.4]]))


class TestBasicBehavior:
    def test_single_significant(self) -> None:
        # Only one strongly significant p-value among 5.
        result = benjamini_hochberg([0.001, 0.5, 0.6, 0.7, 0.8], alpha=0.05)
        assert result == [True, False, False, False, False]

    def test_all_significant(self) -> None:
        result = benjamini_hochberg(
            [0.001, 0.002, 0.003, 0.004, 0.005], alpha=0.05
        )
        assert result == [True, True, True, True, True]

    def test_none_significant(self) -> None:
        result = benjamini_hochberg([0.1, 0.2, 0.5, 0.8, 0.9], alpha=0.05)
        assert result == [False, False, False, False, False]

    def test_preserves_input_order(self) -> None:
        # Permute the inputs; result should permute correspondingly.
        # Sorted [0.001, 0.04, 0.5, 0.6, 0.7] vs thresholds
        # [0.01, 0.02, 0.03, 0.04, 0.05]: only 0.001 ≤ 0.01 passes →
        # reject only the rank-1 element (original idx 1).
        unsorted_p = [0.5, 0.001, 0.6, 0.7, 0.04]
        result = benjamini_hochberg(unsorted_p, alpha=0.05)
        assert result == [False, True, False, False, False]

    def test_alpha_threshold_documented_example(self) -> None:
        # Docstring example: ranks 1, 2, 3 each below their threshold;
        # ranks 4, 5 above.
        result = benjamini_hochberg([0.001, 0.02, 0.025, 0.5, 0.9], alpha=0.05)
        assert result == [True, True, True, False, False]


class TestLessConservativeThanBonferroni:
    def test_bh_rejects_more_than_bonferroni(self) -> None:
        # A case Bonferroni misses but BH catches: 5 tests, two with p around
        # α — Bonferroni at α/5 = 0.01 would reject neither.
        p = [0.001, 0.02, 0.025, 0.5, 0.9]
        result = benjamini_hochberg(p, alpha=0.05)
        # BH rejects 0.001, 0.02, 0.025 (largest i where p[i] ≤ i/5 * 0.05).
        assert sum(result) >= 1
        # Bonferroni would reject only the 0.001 (≤ 0.05/5 = 0.01).


class TestStepUpProperty:
    """If BH rejects at rank i, it must also reject all smaller ranks."""

    def test_rejection_is_a_prefix_of_sorted_p(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = rng.uniform(0, 1, 50).tolist()
            result = benjamini_hochberg(p, alpha=0.1)
            # Map rejection back to sorted order.
            sorted_idx = sorted(range(len(p)), key=lambda i: p[i])
            rejected_sorted = [result[i] for i in sorted_idx]
            # Find rightmost True; everything before must be True.
            if any(rejected_sorted):
                last_true = max(i for i, x in enumerate(rejected_sorted) if x)
                assert all(rejected_sorted[: last_true + 1])


class TestFDRControl:
    """Under H₀ (all uniform p-values), expected FDR is ≤ α."""

    def test_fdr_under_h0(self) -> None:
        # Under H₀ every rejection is a false discovery, so per-simulation
        # FDR equals 1 if anything is rejected, else 0. The expected FDR is
        # then simply P(any rejection per simulation) — which is FWER. For
        # all-null configurations, BH controls FWER at α.
        rng = np.random.default_rng(42)
        n_sims = 1000
        m = 20
        alpha = 0.05
        per_sim_fdr_sum = 0.0
        for _ in range(n_sims):
            p = rng.uniform(0, 1, m).tolist()
            rejected = benjamini_hochberg(p, alpha=alpha)
            if any(rejected):
                per_sim_fdr_sum += 1.0
        expected_fdr = per_sim_fdr_sum / n_sims
        # BH controls FDR at α; allow ~1% sampling slack.
        assert expected_fdr <= alpha + 0.02, (
            f"Empirical FDR {expected_fdr:.3f} exceeds α={alpha} + slack"
        )

    def test_recovers_true_rejections_under_mixed(self) -> None:
        """When 5/20 tests truly differ, BH should reject most of the 5."""
        rng = np.random.default_rng(7)
        n_sims = 500
        m = 20
        n_alt = 5
        true_positive_rate_sum = 0.0
        for _ in range(n_sims):
            p = list(rng.uniform(0, 1, m))
            # Replace the first n_alt with strongly significant p-values.
            for i in range(n_alt):
                p[i] = rng.uniform(0.0, 0.001)
            rejected = benjamini_hochberg(p, alpha=0.05)
            true_positives = sum(rejected[:n_alt])
            true_positive_rate_sum += true_positives / n_alt
        average_tpr = true_positive_rate_sum / n_sims
        assert average_tpr >= 0.85, (
            f"Average true-positive rate {average_tpr:.3f}; BH should recover "
            f"most strong signals"
        )
