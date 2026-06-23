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
from caliber.core.multiple import benjamini_hochberg
from caliber.core.sample_size import sample_size
from caliber.core.sequential import SequentialTester
from caliber.core.types import (
    CompareResult,
    DriftEvent,
    SampleSizeResult,
    Verdict,
)
from caliber.version import __version__

__all__ = [
    "CUSUMDetector",
    "CompareResult",
    "DriftEvent",
    "PageHinkleyDetector",
    "SampleSizeResult",
    "SequentialTester",
    "Verdict",
    "__version__",
    "benjamini_hochberg",
    "compare",
    "sample_size",
]


# Optional adapter imports - fail silently if extras not installed.
with contextlib.suppress(ImportError):
    from caliber.adapters.braintrust import from_braintrust  # noqa: F401

with contextlib.suppress(ImportError):
    from caliber.adapters.langfuse import from_langfuse  # noqa: F401

with contextlib.suppress(ImportError):
    from caliber.adapters.langsmith import from_langsmith  # noqa: F401
