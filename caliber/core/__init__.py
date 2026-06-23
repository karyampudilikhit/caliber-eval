"""Pure statistical primitives. No I/O, no side effects, deterministic given a seed.

Public API surface — these symbols are re-exported from `caliber` itself.
"""

from caliber.core.compare import compare
from caliber.core.drift import CUSUMDetector, PageHinkleyDetector
from caliber.core.multiple import benjamini_hochberg
from caliber.core.sample_size import sample_size
from caliber.core.sequential import SequentialTester
from caliber.core.types import (
    CompareResult,
    DriftEvent,
    SampleSizeResult,
    Verdict,
)

__all__ = [
    "CUSUMDetector",
    "CompareResult",
    "DriftEvent",
    "PageHinkleyDetector",
    "SampleSizeResult",
    "SequentialTester",
    "Verdict",
    "benjamini_hochberg",
    "compare",
    "sample_size",
]
