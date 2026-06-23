"""Langfuse adapter — implementation lands in session 6."""

from __future__ import annotations

from typing import Any

import numpy as np


def from_langfuse(
    client: Any,
    dataset_name: str,
    run_a: str,
    run_b: str,
    metric: str,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("from_langfuse arrives in session 6")
