"""Pure statistical primitives. No I/O, no side effects, deterministic given a seed.

Public API surface — these symbols are re-exported from `caliber` itself.
"""

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
    "benjamini_hochberg",
    "compare",
    "explain",
    "judge_hypothesis",
    "sample_size",
]
