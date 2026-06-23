"""Command-line interface for Caliber.

Three commands mirroring the library's public API:

    caliber compare old.csv new.csv      — verdict + CI on paired eval scores
    caliber sample-size --effect 0.05    — required n for a target effect/power
    caliber drift scores.csv             — drift events in a streaming series

Every command accepts ``--json`` to emit a Pydantic-serialised result instead
of a human-readable summary; pipe it to ``jq`` or another tool for scripts.

CSV inputs are expected to be one number per line. A header row is detected
automatically (first line not parseable as a float → treated as header).
Multi-column CSVs use the first column only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from caliber.core.compare import compare
from caliber.core.drift import CUSUMDetector, PageHinkleyDetector
from caliber.core.sample_size import sample_size
from caliber.core.types import CompareResult, DriftEvent, SampleSizeResult
from caliber.version import __version__

app = typer.Typer(
    name="caliber",
    help="Statistical decision layer for AI/LLM evaluations.",
    no_args_is_help=True,
    add_completion=False,
)


def _load_csv(path: Path) -> np.ndarray:
    """Load a 1-D float array from a CSV file.

    Header row is auto-detected. Multi-column files use the first column only.
    """
    if not path.is_file():
        typer.echo(f"Error: file not found: {path}", err=True)
        raise typer.Exit(code=1)

    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    try:
        float(first_line.split(",")[0])
        skiprows = 0
    except (ValueError, IndexError):
        skiprows = 1

    try:
        arr = np.loadtxt(
            path,
            delimiter=",",
            skiprows=skiprows,
            dtype=np.float64,
            ndmin=1,
        )
    except ValueError as e:
        typer.echo(f"Error reading {path}: {e}", err=True)
        raise typer.Exit(code=1) from None

    if arr.ndim > 1:
        arr = arr[:, 0]
    return arr


# ----------------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------------


@app.command(name="compare")
def cmd_compare(
    old_scores: Annotated[
        Path,
        typer.Argument(help="CSV of baseline (old) scores, one per line."),
    ],
    new_scores: Annotated[
        Path,
        typer.Argument(help="CSV of new scores, one per line. Same length as old."),
    ],
    confidence: Annotated[
        float,
        typer.Option("--confidence", "-c", help="CI level in (0, 1)."),
    ] = 0.95,
    method: Annotated[
        str,
        typer.Option(
            "--method", "-m", help="paired_t | paired_bootstrap | auto"
        ),
    ] = "auto",
    n_bootstrap: Annotated[
        int,
        typer.Option("--n-bootstrap", help="Bootstrap resamples."),
    ] = 10_000,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Bootstrap RNG seed."),
    ] = None,
    metric_name: Annotated[
        str,
        typer.Option("--metric", help="Metric label used in the recommendation."),
    ] = "score",
    practical_threshold: Annotated[
        float,
        typer.Option(
            "--practical-threshold",
            help="|mean difference| below this -> NO_CHANGE.",
        ),
    ] = 0.0,
    target_effect: Annotated[
        float | None,
        typer.Option(
            "--target-effect",
            help="If verdict is INCONCLUSIVE, return required n to detect this effect.",
        ),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a human summary."),
    ] = False,
) -> None:
    """Compare two arrays of paired eval scores and print a verdict."""
    old = _load_csv(old_scores)
    new = _load_csv(new_scores)

    if method not in ("paired_t", "paired_bootstrap", "auto"):
        typer.echo(
            f"Error: --method must be paired_t | paired_bootstrap | auto; "
            f"got {method!r}",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        result = compare(
            old,
            new,
            confidence_level=confidence,
            method=method,  # type: ignore[arg-type]
            n_bootstrap=n_bootstrap,
            seed=seed,
            metric_name=metric_name,
            practical_threshold=practical_threshold,
            target_effect=target_effect,
        )
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None

    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        _render_compare(result)


def _render_compare(r: CompareResult) -> None:
    typer.echo(f"verdict:           {r.verdict}")
    typer.echo(f"mean difference:   {r.mean_difference:+.6f}")
    typer.echo(
        f"{int(r.confidence_level * 100)}% CI:            "
        f"({r.ci_lower:+.6f}, {r.ci_upper:+.6f})"
    )
    typer.echo(f"p-value:           {r.p_value:.6g}")
    typer.echo(f"n pairs:           {r.n}")
    typer.echo(f"method:            {r.method}")
    if r.sample_size_needed is not None:
        typer.echo(f"n needed:          {r.sample_size_needed}")
    typer.echo("")
    typer.echo(r.recommendation)


# ----------------------------------------------------------------------------
# sample-size
# ----------------------------------------------------------------------------


@app.command(name="sample-size")
def cmd_sample_size(
    effect: Annotated[
        float,
        typer.Option(
            "--effect", "-e", help="Effect size (Cohen's d by default)."
        ),
    ],
    effect_type: Annotated[
        str,
        typer.Option("--effect-type", help="cohens_d | absolute"),
    ] = "cohens_d",
    baseline_std: Annotated[
        float | None,
        typer.Option(
            "--baseline-std",
            help="Required when --effect-type absolute.",
        ),
    ] = None,
    power: Annotated[
        float,
        typer.Option("--power", "-p", help="Target statistical power."),
    ] = 0.8,
    confidence: Annotated[
        float,
        typer.Option("--confidence", "-c", help="Confidence level in (0, 1)."),
    ] = 0.95,
    test: Annotated[
        str,
        typer.Option("--test", help="paired_t | unpaired_t | proportion"),
    ] = "paired_t",
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a human summary."),
    ] = False,
) -> None:
    """Compute the required sample size to detect a target effect."""
    if effect_type not in ("cohens_d", "absolute"):
        typer.echo(
            f"Error: --effect-type must be cohens_d | absolute; got {effect_type!r}",
            err=True,
        )
        raise typer.Exit(code=2)
    if test not in ("paired_t", "unpaired_t", "proportion"):
        typer.echo(
            f"Error: --test must be paired_t | unpaired_t | proportion; "
            f"got {test!r}",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        result = sample_size(
            effect_size=effect,
            effect_type=effect_type,  # type: ignore[arg-type]
            baseline_std=baseline_std,
            power=power,
            confidence_level=confidence,
            test=test,  # type: ignore[arg-type]
        )
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None

    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        _render_sample_size(result)


def _render_sample_size(r: SampleSizeResult) -> None:
    typer.echo(f"n per arm:          {r.n_per_arm}")
    typer.echo(f"effect (Cohen d):   {r.effect_size:.4f}")
    typer.echo(f"target power:       {r.power:.2f}")
    typer.echo(f"confidence level:   {r.confidence_level:.2f}")


# ----------------------------------------------------------------------------
# drift
# ----------------------------------------------------------------------------


@app.command(name="drift")
def cmd_drift(
    scores: Annotated[
        Path,
        typer.Argument(help="CSV of streaming scores, one per line."),
    ],
    detector: Annotated[
        str,
        typer.Option("--detector", help="page_hinkley | cusum"),
    ] = "page_hinkley",
    delta: Annotated[
        float,
        typer.Option("--delta", help="Page-Hinkley tolerance."),
    ] = 0.05,
    threshold: Annotated[
        float,
        typer.Option("--threshold", help="Page-Hinkley decision threshold."),
    ] = 50.0,
    target_mean: Annotated[
        float | None,
        typer.Option("--target-mean", help="Required for --detector cusum."),
    ] = None,
    target_std: Annotated[
        float,
        typer.Option("--target-std", help="CUSUM reference std."),
    ] = 1.0,
    k: Annotated[
        float,
        typer.Option("--k", help="CUSUM reference (in σ units)."),
    ] = 0.5,
    h: Annotated[
        float,
        typer.Option("--h", help="CUSUM decision threshold (in σ units)."),
    ] = 5.0,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of a human summary."),
    ] = False,
) -> None:
    """Run a drift detector over a CSV of streaming scores; print any events."""
    arr = _load_csv(scores)

    if detector == "page_hinkley":
        det: PageHinkleyDetector | CUSUMDetector = PageHinkleyDetector(
            delta=delta, threshold=threshold
        )
    elif detector == "cusum":
        if target_mean is None:
            typer.echo(
                "Error: --target-mean is required for --detector cusum.",
                err=True,
            )
            raise typer.Exit(code=2)
        det = CUSUMDetector(
            target_mean=target_mean, target_std=target_std, k=k, h=h
        )
    else:
        typer.echo(
            f"Error: --detector must be page_hinkley | cusum; got {detector!r}",
            err=True,
        )
        raise typer.Exit(code=2)

    events: list[DriftEvent] = []
    for score in arr.tolist():
        ev = det.add(float(score))
        if ev is not None:
            events.append(ev)
            det.reset()

    if as_json:
        out = [json.loads(ev.model_dump_json()) for ev in events]
        typer.echo(json.dumps(out, indent=2))
    else:
        _render_drift(events, len(arr))


def _render_drift(events: list[DriftEvent], n_samples: int) -> None:
    if not events:
        typer.echo(f"No drift detected in {n_samples} samples.")
        return
    typer.echo(f"{len(events)} drift event(s) in {n_samples} samples:")
    for i, ev in enumerate(events, start=1):
        typer.echo(
            f"  [{i}] index {ev.detected_at_index}: "
            f"mean {ev.mean_before:.4f} -> {ev.mean_after:.4f} "
            f"(magnitude {ev.magnitude:.4f}, p={ev.p_value:.4g}, "
            f"method={ev.method})"
        )


# ----------------------------------------------------------------------------
# version
# ----------------------------------------------------------------------------


@app.command(name="version")
def cmd_version() -> None:
    """Print the installed caliber version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
