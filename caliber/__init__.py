"""Caliber — statistical decision layer for AI/LLM evaluations.

Public API:
    compare(old, new) -> CompareResult
    sample_size(effect_size, ...) -> SampleSizeResult
    SequentialTester(...)
    PageHinkleyDetector(...), CUSUMDetector(...)
    benjamini_hochberg(p_values, ...) -> list[bool]

Adapters (optional dependencies):
    from_braintrust, from_langfuse, from_langsmith
"""

from __future__ import annotations

import contextlib

from caliber.core.compare import compare
from caliber.core.drift import CUSUMDetector, PageHinkleyDetector
from caliber.core.explain import explain
from caliber.core.judge import LLMProvider, OllamaProvider, judge_hypothesis
from caliber.core.multiple import benjamini_hochberg
from caliber.core.sample_size import sample_size
from caliber.core.sequential import SequentialTester
from caliber.core.types import (
    CompareResult,
    DriftEvent,
    ExampleFlip,
    ExplanationResult,
    JudgedHypothesis,
    SampleSizeResult,
    Stratum,
    Verdict,
)
from caliber.version import __version__

__all__ = [
    "CUSUMDetector",
    "CompareResult",
    "DriftEvent",
    "ExampleFlip",
    "ExplanationResult",
    "JudgedHypothesis",
    "LLMProvider",
    "OllamaProvider",
    "PageHinkleyDetector",
    "SampleSizeResult",
    "SequentialTester",
    "Stratum",
    "Verdict",
    "__version__",
    "benjamini_hochberg",
    "compare",
    "explain",
    "judge_hypothesis",
    "sample_size",
]


# Optional adapter imports - fail silently if extras not installed.
with contextlib.suppress(ImportError):
    from caliber.adapters.braintrust import from_braintrust  # noqa: F401

with contextlib.suppress(ImportError):
    from caliber.adapters.langfuse import from_langfuse  # noqa: F401

with contextlib.suppress(ImportError):
    from caliber.adapters.langsmith import from_langsmith  # noqa: F401
