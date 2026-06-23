"""Tests for caliber.core.explain — stratification + driving examples."""

from __future__ import annotations

import numpy as np
import pytest

from caliber import (
    ExampleFlip,
    ExplanationResult,
    Stratum,
    explain,
)


# Reusable fixture: 10 paired binary scores split across 4 categories.
@pytest.fixture
def binary_data() -> dict[str, list]:
    return {
        "old_scores": [0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        "new_scores": [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        "inputs":     ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10"],
        "categories": ["math", "math", "word", "word", "word",
                       "code", "code", "code", "logic", "logic"],
    }


# ============================================================================
# Return shape and types
# ============================================================================


class TestReturnShape:
    def test_returns_explanation_result(self, binary_data: dict) -> None:
        r = explain(binary_data["old_scores"], binary_data["new_scores"], seed=0)
        assert isinstance(r, ExplanationResult)

    def test_includes_compare_verdict(self, binary_data: dict) -> None:
        r = explain(binary_data["old_scores"], binary_data["new_scores"], seed=0)
        # verdict must be one of the documented labels
        assert r.verdict in {"BETTER", "WORSE", "INCONCLUSIVE", "NO_CHANGE"}
        assert 0.0 <= r.p_value <= 1.0
        assert r.n == len(binary_data["old_scores"])

    def test_ci_property_returns_tuple(self, binary_data: dict) -> None:
        r = explain(binary_data["old_scores"], binary_data["new_scores"], seed=0)
        assert r.ci == (r.ci_lower, r.ci_upper)

    def test_summary_non_empty(self, binary_data: dict) -> None:
        r = explain(binary_data["old_scores"], binary_data["new_scores"], seed=0)
        assert len(r.summary) > 20
        assert "verdict" in r.summary.lower()


# ============================================================================
# Optional arguments — works without any of them
# ============================================================================


class TestWorksWithoutOptionalArgs:
    def test_only_scores(self) -> None:
        r = explain([0, 0, 1, 1], [1, 1, 1, 1], seed=0)
        assert isinstance(r, ExplanationResult)
        assert r.strata == []
        assert r.biggest_gain_category is None
        assert r.smallest_gain_category is None
        # All improvements have no category/input populated
        for ex in r.top_improvements:
            assert ex.input is None
            assert ex.category is None
            assert ex.old_output is None
            assert ex.new_output is None


# ============================================================================
# Stratification
# ============================================================================


class TestStratification:
    def test_no_categories_returns_empty_strata(self, binary_data: dict) -> None:
        r = explain(binary_data["old_scores"], binary_data["new_scores"], seed=0)
        assert r.strata == []

    def test_categories_create_one_stratum_per_unique(self, binary_data: dict) -> None:
        r = explain(
            binary_data["old_scores"],
            binary_data["new_scores"],
            categories=binary_data["categories"],
            seed=0,
        )
        names = {s.name for s in r.strata}
        assert names == {"math", "word", "code", "logic"}

    def test_stratum_math_matches_manual(self, binary_data: dict) -> None:
        r = explain(
            binary_data["old_scores"],
            binary_data["new_scores"],
            categories=binary_data["categories"],
            seed=0,
        )
        by_name = {s.name: s for s in r.strata}
        # math: indices 0, 1 — old [0, 0] → new [1, 1]
        assert by_name["math"].n == 2
        assert by_name["math"].old_accuracy == 0.0
        assert by_name["math"].new_accuracy == 1.0
        assert by_name["math"].delta == 1.0
        # word: indices 2, 3, 4 — old [1, 1, 0] → new [1, 1, 0]
        assert by_name["word"].n == 3
        assert by_name["word"].old_accuracy == pytest.approx(2 / 3)
        assert by_name["word"].new_accuracy == pytest.approx(2 / 3)
        assert by_name["word"].delta == pytest.approx(0.0)

    def test_biggest_and_smallest_identified(self, binary_data: dict) -> None:
        r = explain(
            binary_data["old_scores"],
            binary_data["new_scores"],
            categories=binary_data["categories"],
            seed=0,
        )
        # math went from 0% to 100% — biggest gain
        # word stayed flat — smallest gain
        assert r.biggest_gain_category == "math"
        assert r.smallest_gain_category == "word"

    def test_strata_returns_stratum_instances(self, binary_data: dict) -> None:
        r = explain(
            binary_data["old_scores"],
            binary_data["new_scores"],
            categories=binary_data["categories"],
            seed=0,
        )
        assert all(isinstance(s, Stratum) for s in r.strata)

    def test_single_category_works(self) -> None:
        r = explain(
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            categories=["x"] * 5,
            seed=0,
        )
        assert len(r.strata) == 1
        assert r.strata[0].name == "x"
        # Biggest == smallest when there's only one
        assert r.biggest_gain_category == "x"
        assert r.smallest_gain_category == "x"


# ============================================================================
# Driving examples
# ============================================================================


class TestDrivingExamples:
    def test_top_improvements_have_positive_delta(self, binary_data: dict) -> None:
        r = explain(
            binary_data["old_scores"],
            binary_data["new_scores"],
            seed=0,
        )
        for ex in r.top_improvements:
            assert ex.delta > 0

    def test_top_regressions_have_negative_delta(self) -> None:
        # Construct data with regressions: new[0]=0 worse than old[0]=1.
        r = explain(
            [1, 1, 0, 0, 1],
            [0, 1, 1, 1, 0],  # indices 0 and 4 regress; 2, 3 improve
            seed=0,
        )
        for ex in r.top_regressions:
            assert ex.delta < 0
        # 2 improvements, 2 regressions in this synthetic data
        assert len(r.top_improvements) == 2
        assert len(r.top_regressions) == 2

    def test_top_n_examples_respected(self) -> None:
        # 5 improvements possible — request only 2.
        r = explain(
            [0] * 5,
            [1] * 5,
            top_n_examples=2,
            seed=0,
        )
        assert len(r.top_improvements) == 2
        assert len(r.top_regressions) == 0

    def test_no_improvements_returns_empty(self) -> None:
        # Identical scores → no improvements, no regressions.
        # Avoid all-identical (compare() would reject) by varying one pair.
        r = explain([0, 1, 1, 1], [0, 1, 1, 0], seed=0)
        assert len(r.top_improvements) == 0
        assert len(r.top_regressions) == 1

    def test_input_text_populated_when_provided(self, binary_data: dict) -> None:
        r = explain(
            binary_data["old_scores"],
            binary_data["new_scores"],
            inputs=binary_data["inputs"],
            seed=0,
        )
        for ex in r.top_improvements:
            assert ex.input is not None
            assert ex.input.startswith("Q")

    def test_outputs_populated_when_provided(self) -> None:
        old_o = [f"old{i}" for i in range(5)]
        new_o = [f"new{i}" for i in range(5)]
        r = explain(
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            old_outputs=old_o,
            new_outputs=new_o,
            seed=0,
        )
        for ex in r.top_improvements:
            assert ex.old_output is not None
            assert ex.old_output.startswith("old")
            assert ex.new_output is not None
            assert ex.new_output.startswith("new")

    def test_categories_populated_when_provided(self, binary_data: dict) -> None:
        r = explain(
            binary_data["old_scores"],
            binary_data["new_scores"],
            categories=binary_data["categories"],
            seed=0,
        )
        for ex in r.top_improvements:
            assert ex.category in {"math", "word", "code", "logic"}

    def test_improvements_sorted_by_delta_descending(self) -> None:
        r = explain([0.2, 0.5, 0.1, 0.8], [0.9, 0.6, 0.7, 0.85], seed=0)
        deltas = [ex.delta for ex in r.top_improvements]
        assert deltas == sorted(deltas, reverse=True)

    def test_returns_example_flip_instances(self, binary_data: dict) -> None:
        r = explain(
            binary_data["old_scores"],
            binary_data["new_scores"],
            seed=0,
        )
        assert all(isinstance(ex, ExampleFlip) for ex in r.top_improvements)


# ============================================================================
# Continuous (non-binary) scores
# ============================================================================


class TestContinuousScores:
    def test_continuous_scores_work(self) -> None:
        rng = np.random.default_rng(0)
        old = rng.normal(0.5, 0.1, 30)
        new = rng.normal(0.6, 0.1, 30)
        r = explain(old, new, seed=0)
        assert isinstance(r, ExplanationResult)
        assert r.verdict in {"BETTER", "INCONCLUSIVE"}

    def test_continuous_strata_use_mean(self) -> None:
        # 4 examples in category 'a' with scores ~0.5 → 0.7 (mean shift)
        r = explain(
            [0.4, 0.5, 0.6, 0.5, 0.5, 0.5],
            [0.6, 0.7, 0.8, 0.7, 0.5, 0.5],
            categories=["a", "a", "a", "a", "b", "b"],
            seed=0,
        )
        by_name = {s.name: s for s in r.strata}
        assert by_name["a"].delta == pytest.approx(0.2, abs=0.01)
        assert by_name["b"].delta == pytest.approx(0.0)


# ============================================================================
# Validation
# ============================================================================


class TestValidation:
    def test_mismatched_inputs_length_raises(self, binary_data: dict) -> None:
        with pytest.raises(ValueError, match="inputs"):
            explain(
                binary_data["old_scores"],
                binary_data["new_scores"],
                inputs=["only", "three", "items"],
                seed=0,
            )

    def test_mismatched_categories_length_raises(self, binary_data: dict) -> None:
        with pytest.raises(ValueError, match="categories"):
            explain(
                binary_data["old_scores"],
                binary_data["new_scores"],
                categories=["math"],
                seed=0,
            )

    def test_mismatched_old_outputs_length_raises(self, binary_data: dict) -> None:
        with pytest.raises(ValueError, match="old_outputs"):
            explain(
                binary_data["old_scores"],
                binary_data["new_scores"],
                old_outputs=["a", "b"],
                seed=0,
            )

    def test_mismatched_new_outputs_length_raises(self, binary_data: dict) -> None:
        with pytest.raises(ValueError, match="new_outputs"):
            explain(
                binary_data["old_scores"],
                binary_data["new_scores"],
                new_outputs=["a", "b"],
                seed=0,
            )

    def test_negative_top_n_raises(self, binary_data: dict) -> None:
        with pytest.raises(ValueError, match="top_n_examples"):
            explain(
                binary_data["old_scores"],
                binary_data["new_scores"],
                top_n_examples=-1,
                seed=0,
            )

    def test_top_n_zero_works(self, binary_data: dict) -> None:
        # zero is allowed — surfaces no examples
        r = explain(
            binary_data["old_scores"],
            binary_data["new_scores"],
            top_n_examples=0,
            seed=0,
        )
        assert r.top_improvements == []
        assert r.top_regressions == []

    def test_invalid_input_propagates_compare_error(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            explain([0.1, 0.2, 0.3], [0.1, 0.2], seed=0)


# ============================================================================
# Summary content
# ============================================================================


class TestSummary:
    def test_summary_mentions_verdict(self, binary_data: dict) -> None:
        r = explain(
            binary_data["old_scores"],
            binary_data["new_scores"],
            seed=0,
        )
        assert r.verdict in r.summary

    def test_summary_mentions_categories_when_provided(self, binary_data: dict) -> None:
        r = explain(
            binary_data["old_scores"],
            binary_data["new_scores"],
            categories=binary_data["categories"],
            seed=0,
        )
        for cat in {"math", "word", "code", "logic"}:
            assert cat in r.summary

    def test_summary_omits_categories_section_when_absent(self, binary_data: dict) -> None:
        r = explain(
            binary_data["old_scores"],
            binary_data["new_scores"],
            seed=0,
        )
        assert "By category" not in r.summary
