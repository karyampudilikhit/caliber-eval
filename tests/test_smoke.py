"""Smoke tests — verify the package imports and Pydantic models construct."""

from __future__ import annotations

import caliber
from caliber import CompareResult, DriftEvent, SampleSizeResult


def test_version_is_set() -> None:
    assert isinstance(caliber.__version__, str)
    assert caliber.__version__.count(".") >= 1


def test_compare_result_constructs() -> None:
    r = CompareResult(
        verdict="BETTER",
        mean_difference=0.05,
        ci_lower=0.01,
        ci_upper=0.09,
        confidence_level=0.95,
        p_value=0.012,
        n=100,
        method="paired_t",
        recommendation="Ship it.",
    )
    assert r.verdict == "BETTER"
    assert r.ci == (0.01, 0.09)
    assert r.sample_size_needed is None


def test_sample_size_result_constructs() -> None:
    r = SampleSizeResult(
        n_per_arm=128,
        effect_size=0.5,
        power=0.8,
        confidence_level=0.95,
    )
    assert r.n_per_arm == 128


def test_drift_event_constructs() -> None:
    e = DriftEvent(
        detected_at_index=42,
        mean_before=0.7,
        mean_after=0.6,
        magnitude=0.1,
        p_value=0.03,
        method="page_hinkley",
    )
    assert e.method == "page_hinkley"
    assert e.detected_at_timestamp is None


def test_benjamini_hochberg_smoke() -> None:
    """Smoke check that the BH correction is wired up."""
    result = caliber.benjamini_hochberg([0.001, 0.5])
    assert result == [True, False]
