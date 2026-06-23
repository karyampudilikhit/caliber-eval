"""Braintrust adapter — implementation lands in session 6."""

from __future__ import annotations

from typing import Any

import numpy as np


def from_braintrust(
    project_id: str,
    experiment_a: str,
    experiment_b: str,
    scorer: str,
    api_key: str | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("from_braintrust arrives in session 6")
