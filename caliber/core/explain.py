"""Explain WHY two prompt versions differ — stratification + driving examples.

`compare()` answers "is the difference real?". `explain()` answers two further
questions:

  - **Where did the gain come from?** Break the delta down by user-supplied
    category labels. If you tagged each eval example with `category="PEMDAS"`
    or `category="word-problem"`, you can see exactly which slice moved.

  - **Which specific examples drove the verdict?** Sort the per-example deltas
    and surface the top-N improvements and regressions — including the
    original input text and both model outputs if you provided them.

The output is pure data analysis — no LLM call. (A future LLM-judge layer
could read the surfaced examples and propose a textual hypothesis; that's
deliberately not part of v1.)

Works on **any score type**, not just binary:
  - Binary 0/1 accuracy → "wins" and "losses"
  - Continuous scores (0.0-1.0, BLEU, judge ratings) → biggest improvements
    and biggest regressions

Examples
--------
>>> import caliber
>>> r = caliber.explain(
...     old_scores=[0, 0, 1, 1, 0],
...     new_scores=[1, 1, 1, 1, 0],
...     inputs=["a", "b", "c", "d", "e"],
...     categories=["A", "A", "B", "B", "B"],
... )
>>> r.verdict          # doctest: +SKIP
'INCONCLUSIVE'
>>> len(r.top_improvements)  # doctest: +SKIP
2
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from caliber.core.compare import ArrayLike, _validate_array, compare
from caliber.core.types import (
    CompareResult,
    ExampleFlip,
    ExplanationResult,
    Stratum,
)


def explain(
    old_scores: ArrayLike,
    new_scores: ArrayLike,
    *,
    inputs: Sequence[str] | None = None,
    old_outputs: Sequence[str] | None = None,
    new_outputs: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    top_n_examples: int = 3,
    confidence_level: float = 0.95,
    method: str = "auto",
    n_bootstrap: int = 10_000,
    seed: int | None = None,
    metric_name: str = "score",
    practical_threshold: float = 0.0,
) -> ExplanationResult:
    """Decide whether ``new_scores`` differs from ``old_scores`` AND explain why.

    Parameters
    ----------
    old_scores, new_scores : array-like of float, shape (n,)
        Paired score arrays — same constraints as `caliber.compare`.
    inputs : sequence of str, optional
        The original prompts/questions, one per example. Used only to enrich
        the surfaced driving examples; doesn't affect the statistics.
    old_outputs, new_outputs : sequence of str, optional
        The model's actual responses for each example. Useful for human
        review of what changed in the driving examples.
    categories : sequence of str, optional
        A category label per example (e.g. "math", "code", "summary"). If
        provided, results are stratified per category so you can see where
        the delta came from.
    top_n_examples : int, default 3
        How many driving improvements and regressions to surface.
    confidence_level, method, n_bootstrap, seed, metric_name, practical_threshold
        Forwarded to `caliber.compare()` for the underlying statistical verdict.

    Returns
    -------
    ExplanationResult
        Statistical verdict + per-category strata + top-N improvements/regressions.

    Raises
    ------
    ValueError
        If any optional sequence is the wrong length, or if the inputs are
        otherwise invalid (delegates to `compare()` for that).

    Examples
    --------
    >>> import caliber
    >>> r = caliber.explain(
    ...     old_scores=[0, 0, 1, 1, 0],
    ...     new_scores=[1, 1, 1, 1, 0],
    ...     inputs=["q1", "q2", "q3", "q4", "q5"],
    ...     categories=["A", "A", "B", "B", "B"],
    ... )
    >>> r.strata[0].name
    'A'
    """
    # Statistical verdict first — same call the user would make directly.
    cmp: CompareResult = compare(
        old_scores,
        new_scores,
        confidence_level=confidence_level,
        method=method,  # type: ignore[arg-type]
        n_bootstrap=n_bootstrap,
        seed=seed,
        metric_name=metric_name,
        practical_threshold=practical_threshold,
    )

    old = _validate_array(old_scores, "old_scores")
    new = _validate_array(new_scores, "new_scores")
    n = len(old)

    # Validate optional sequence lengths.
    _validate_optional_length(inputs, n, "inputs")
    _validate_optional_length(old_outputs, n, "old_outputs")
    _validate_optional_length(new_outputs, n, "new_outputs")
    _validate_optional_length(categories, n, "categories")

    if top_n_examples < 0:
        raise ValueError(f"top_n_examples must be ≥ 0; got {top_n_examples}")

    # Stratification — empty if no categories supplied.
    strata = _stratify(old, new, categories) if categories is not None else []
    biggest, smallest = _extreme_category_names(strata)

    # Driving examples — top-N improvements and top-N regressions by delta.
    top_improvements, top_regressions = _find_driving_examples(
        old=old,
        new=new,
        inputs=inputs,
        old_outputs=old_outputs,
        new_outputs=new_outputs,
        categories=categories,
        top_n=top_n_examples,
    )

    summary = _build_summary(
        cmp=cmp,
        strata=strata,
        biggest=biggest,
        smallest=smallest,
        top_improvements=top_improvements,
        top_regressions=top_regressions,
        metric_name=metric_name,
    )

    return ExplanationResult(
        verdict=cmp.verdict,
        mean_difference=cmp.mean_difference,
        ci_lower=cmp.ci_lower,
        ci_upper=cmp.ci_upper,
        confidence_level=cmp.confidence_level,
        p_value=cmp.p_value,
        n=cmp.n,
        method=cmp.method,
        strata=strata,
        biggest_gain_category=biggest,
        smallest_gain_category=smallest,
        top_improvements=top_improvements,
        top_regressions=top_regressions,
        summary=summary,
    )


def _validate_optional_length(
    seq: Sequence[str] | None, expected_n: int, name: str
) -> None:
    if seq is None:
        return
    if len(seq) != expected_n:
        raise ValueError(
            f"{name} length ({len(seq)}) must match score length ({expected_n})"
        )


def _stratify(
    old: np.ndarray, new: np.ndarray, categories: Sequence[str]
) -> list[Stratum]:
    """Compute one Stratum per unique category in ``categories``."""
    result: list[Stratum] = []
    # sorted() over set() for deterministic ordering across runs.
    for cat in sorted(set(categories)):
        idx = [i for i, c in enumerate(categories) if c == cat]
        old_acc = float(old[idx].mean())
        new_acc = float(new[idx].mean())
        result.append(
            Stratum(
                name=cat,
                n=len(idx),
                old_accuracy=old_acc,
                new_accuracy=new_acc,
                delta=new_acc - old_acc,
            )
        )
    return result


def _extreme_category_names(
    strata: list[Stratum],
) -> tuple[str | None, str | None]:
    """Return (biggest-delta category name, smallest-delta category name)."""
    if not strata:
        return None, None
    biggest = max(strata, key=lambda s: s.delta).name
    smallest = min(strata, key=lambda s: s.delta).name
    return biggest, smallest


def _find_driving_examples(
    *,
    old: np.ndarray,
    new: np.ndarray,
    inputs: Sequence[str] | None,
    old_outputs: Sequence[str] | None,
    new_outputs: Sequence[str] | None,
    categories: Sequence[str] | None,
    top_n: int,
) -> tuple[list[ExampleFlip], list[ExampleFlip]]:
    """Surface up to `top_n` strict improvements and `top_n` strict regressions."""
    deltas = new - old
    # argsort ascending; negate for descending order.
    improvement_order = np.argsort(-deltas)
    regression_order = np.argsort(deltas)

    improvements: list[ExampleFlip] = []
    for i in improvement_order:
        if deltas[i] <= 0 or len(improvements) >= top_n:
            break
        improvements.append(
            _build_flip(int(i), old, new, inputs, old_outputs, new_outputs, categories)
        )

    regressions: list[ExampleFlip] = []
    for i in regression_order:
        if deltas[i] >= 0 or len(regressions) >= top_n:
            break
        regressions.append(
            _build_flip(int(i), old, new, inputs, old_outputs, new_outputs, categories)
        )

    return improvements, regressions


def _build_flip(
    i: int,
    old: np.ndarray,
    new: np.ndarray,
    inputs: Sequence[str] | None,
    old_outputs: Sequence[str] | None,
    new_outputs: Sequence[str] | None,
    categories: Sequence[str] | None,
) -> ExampleFlip:
    return ExampleFlip(
        index=i,
        input=inputs[i] if inputs is not None else None,
        old_score=float(old[i]),
        new_score=float(new[i]),
        delta=float(new[i] - old[i]),
        old_output=old_outputs[i] if old_outputs is not None else None,
        new_output=new_outputs[i] if new_outputs is not None else None,
        category=categories[i] if categories is not None else None,
    )


def _build_summary(
    *,
    cmp: CompareResult,
    strata: list[Stratum],
    biggest: str | None,
    smallest: str | None,
    top_improvements: list[ExampleFlip],
    top_regressions: list[ExampleFlip],
    metric_name: str,
) -> str:
    """Human-readable multi-section summary."""
    lines: list[str] = []
    lines.append(
        f"Verdict: {cmp.verdict} ({metric_name} Δ={cmp.mean_difference:+.4f}, "
        f"95% CI [{cmp.ci_lower:+.4f}, {cmp.ci_upper:+.4f}], p={cmp.p_value:.3g}, n={cmp.n})"
    )

    if strata:
        lines.append("")
        lines.append("By category (sorted by delta):")
        for s in sorted(strata, key=lambda x: -x.delta):
            lines.append(
                f"  {s.name:<14} n={s.n:<3}  "
                f"old={s.old_accuracy:.0%}  new={s.new_accuracy:.0%}  "
                f"Δ {s.delta:+.0%}"
            )
        if biggest and smallest and biggest != smallest:
            lines.append("")
            lines.append(f"Largest gain: {biggest}.  Smallest gain: {smallest}.")

    if top_improvements:
        lines.append("")
        lines.append(f"Top {len(top_improvements)} improvement(s):")
        for ex in top_improvements:
            lines.append(_format_flip(ex))

    if top_regressions:
        lines.append("")
        lines.append(f"Top {len(top_regressions)} regression(s):")
        for ex in top_regressions:
            lines.append(_format_flip(ex))

    return "\n".join(lines)


def _format_flip(ex: ExampleFlip) -> str:
    cat = f" ({ex.category})" if ex.category else ""
    parts = [f"  [#{ex.index}]{cat}"]
    if ex.input:
        inp = ex.input if len(ex.input) <= 60 else ex.input[:57] + "..."
        parts.append(inp)
    parts.append(
        f"Δ {ex.delta:+.2f} (old={ex.old_score:.2f}, new={ex.new_score:.2f})"
    )
    return "  ".join(parts)
