"""Pydantic models for Caliber's public API surface.

All inputs/outputs that cross module boundaries are Pydantic models — never dicts.
Verdict is a string Literal so it serializes cleanly to JSON for CLI/web consumers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["BETTER", "WORSE", "INCONCLUSIVE", "NO_CHANGE"]


class CompareResult(BaseModel):
    """Verdict from comparing two arrays of paired eval scores.

    `verdict` is the headline. `ci_lower`/`ci_upper` form the confidence interval
    on the mean paired difference (new - old). If the CI excludes zero, the
    verdict is BETTER (Δ > 0) or WORSE (Δ < 0). If `|Δ| < practical_threshold`,
    the verdict is NO_CHANGE regardless of significance — practical equivalence
    overrides statistical significance for trivially small effects.
    """

    verdict: Verdict
    mean_difference: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = Field(ge=0.0, le=1.0)
    p_value: float = Field(ge=0.0, le=1.0)
    n: int = Field(gt=0)
    method: str
    sample_size_needed: int | None = Field(default=None, gt=0)
    recommendation: str

    @property
    def ci(self) -> tuple[float, float]:
        return (self.ci_lower, self.ci_upper)


class SampleSizeResult(BaseModel):
    """Required per-arm sample size to detect `effect_size` with given power."""

    n_per_arm: int = Field(gt=0)
    effect_size: float
    power: float = Field(ge=0.0, le=1.0)
    confidence_level: float = Field(ge=0.0, le=1.0)


class DriftEvent(BaseModel):
    """A detected change-point in a streaming score series."""

    detected_at_index: int = Field(ge=0)
    detected_at_timestamp: float | None = None
    mean_before: float
    mean_after: float
    magnitude: float
    p_value: float = Field(ge=0.0, le=1.0)
    method: Literal["page_hinkley", "cusum"]
