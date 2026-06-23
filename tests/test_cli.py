"""Tests for the Caliber CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from caliber import __version__
from caliber.cli import app

runner = CliRunner()


def _write_csv(path: Path, values: list[float], header: bool = False) -> None:
    lines = []
    if header:
        lines.append("score")
    lines.extend(str(v) for v in values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================
# version
# ============================================================================


class TestVersion:
    def test_version_prints(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout


# ============================================================================
# compare
# ============================================================================


class TestCompareCommand:
    def test_clear_better(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        old_path = tmp_path / "old.csv"
        new_path = tmp_path / "new.csv"
        _write_csv(old_path, rng.normal(0.6, 0.05, 100).tolist())
        _write_csv(new_path, rng.normal(0.8, 0.05, 100).tolist())

        result = runner.invoke(app, ["compare", str(old_path), str(new_path)])
        assert result.exit_code == 0, result.stdout
        assert "verdict:" in result.stdout
        assert "BETTER" in result.stdout

    def test_json_output_is_parseable(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        old_path = tmp_path / "old.csv"
        new_path = tmp_path / "new.csv"
        _write_csv(old_path, rng.normal(0.6, 0.05, 100).tolist())
        _write_csv(new_path, rng.normal(0.8, 0.05, 100).tolist())

        result = runner.invoke(
            app, ["compare", str(old_path), str(new_path), "--json"]
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["verdict"] == "BETTER"
        assert payload["n"] == 100
        assert "ci_lower" in payload
        assert "ci_upper" in payload
        assert "method" in payload

    def test_header_row_is_auto_detected(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        old_path = tmp_path / "old.csv"
        new_path = tmp_path / "new.csv"
        _write_csv(old_path, rng.normal(0.6, 0.05, 50).tolist(), header=True)
        _write_csv(new_path, rng.normal(0.65, 0.05, 50).tolist(), header=True)

        result = runner.invoke(app, ["compare", str(old_path), str(new_path)])
        assert result.exit_code == 0
        assert "n pairs:           50" in result.stdout

    def test_missing_file_errors(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["compare", str(tmp_path / "nope.csv"), str(tmp_path / "nope.csv")]
        )
        assert result.exit_code == 1
        assert "not found" in result.stderr.lower()

    def test_invalid_method_errors(self, tmp_path: Path) -> None:
        old_path = tmp_path / "old.csv"
        new_path = tmp_path / "new.csv"
        _write_csv(old_path, [0.1, 0.2, 0.3])
        _write_csv(new_path, [0.2, 0.3, 0.4])
        result = runner.invoke(
            app,
            ["compare", str(old_path), str(new_path), "--method", "garbage"],
        )
        assert result.exit_code == 2
        assert "method" in result.stderr.lower()

    def test_mismatched_lengths_errors(self, tmp_path: Path) -> None:
        old_path = tmp_path / "old.csv"
        new_path = tmp_path / "new.csv"
        _write_csv(old_path, [0.1, 0.2, 0.3])
        _write_csv(new_path, [0.2, 0.3])
        result = runner.invoke(app, ["compare", str(old_path), str(new_path)])
        assert result.exit_code == 1
        assert "same length" in result.stderr.lower()

    def test_seed_makes_bootstrap_reproducible(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        old_path = tmp_path / "old.csv"
        new_path = tmp_path / "new.csv"
        _write_csv(old_path, rng.normal(0.5, 0.1, 20).tolist())
        _write_csv(new_path, rng.normal(0.55, 0.1, 20).tolist())
        args = [
            "compare",
            str(old_path),
            str(new_path),
            "--method",
            "paired_bootstrap",
            "--seed",
            "42",
            "--json",
        ]
        r1 = runner.invoke(app, args)
        r2 = runner.invoke(app, args)
        assert r1.stdout == r2.stdout


# ============================================================================
# sample-size
# ============================================================================


class TestSampleSizeCommand:
    def test_cohens_d_default(self) -> None:
        result = runner.invoke(app, ["sample-size", "--effect", "0.5"])
        assert result.exit_code == 0, result.stdout
        assert "n per arm:" in result.stdout

    def test_absolute_with_baseline(self) -> None:
        result = runner.invoke(
            app,
            [
                "sample-size",
                "--effect",
                "0.05",
                "--effect-type",
                "absolute",
                "--baseline-std",
                "0.1",
            ],
        )
        assert result.exit_code == 0
        # Same Cohen's d as 0.5
        assert "0.5000" in result.stdout

    def test_absolute_without_baseline_errors(self) -> None:
        result = runner.invoke(
            app,
            ["sample-size", "--effect", "0.05", "--effect-type", "absolute"],
        )
        assert result.exit_code == 1
        assert "baseline_std" in result.stderr.lower()

    def test_json_output(self) -> None:
        result = runner.invoke(
            app, ["sample-size", "--effect", "0.5", "--json"]
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert "n_per_arm" in payload
        assert payload["n_per_arm"] > 0

    def test_invalid_test_errors(self) -> None:
        result = runner.invoke(
            app, ["sample-size", "--effect", "0.5", "--test", "garbage"]
        )
        assert result.exit_code == 2

    def test_invalid_effect_type_errors(self) -> None:
        result = runner.invoke(
            app, ["sample-size", "--effect", "0.5", "--effect-type", "garbage"]
        )
        assert result.exit_code == 2


# ============================================================================
# drift
# ============================================================================


class TestDriftCommand:
    def test_no_drift_on_stationary_stream(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        scores_path = tmp_path / "scores.csv"
        _write_csv(scores_path, rng.normal(0.7, 0.05, 200).tolist())

        result = runner.invoke(
            app, ["drift", str(scores_path), "--threshold", "10"]
        )
        assert result.exit_code == 0
        assert "No drift detected" in result.stdout

    def test_detects_drift(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        scores_path = tmp_path / "scores.csv"
        stationary = rng.normal(0.5, 0.1, 200)
        drift = rng.normal(0.8, 0.1, 400)
        _write_csv(scores_path, np.concatenate([stationary, drift]).tolist())

        result = runner.invoke(
            app, ["drift", str(scores_path), "--threshold", "5"]
        )
        assert result.exit_code == 0
        assert "drift event" in result.stdout.lower()

    def test_json_output(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        scores_path = tmp_path / "scores.csv"
        stationary = rng.normal(0.5, 0.1, 100)
        drift = rng.normal(0.8, 0.1, 300)
        _write_csv(scores_path, np.concatenate([stationary, drift]).tolist())

        result = runner.invoke(
            app, ["drift", str(scores_path), "--threshold", "5", "--json"]
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert isinstance(payload, list)
        if payload:
            ev = payload[0]
            assert "detected_at_index" in ev
            assert "mean_before" in ev
            assert "mean_after" in ev
            assert ev["method"] == "page_hinkley"

    def test_cusum_detector(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        scores_path = tmp_path / "scores.csv"
        stationary = rng.normal(0.5, 0.1, 100)
        drift = rng.normal(0.9, 0.1, 200)
        _write_csv(scores_path, np.concatenate([stationary, drift]).tolist())

        result = runner.invoke(
            app,
            [
                "drift",
                str(scores_path),
                "--detector",
                "cusum",
                "--target-mean",
                "0.5",
                "--target-std",
                "0.1",
                "--h",
                "5",
            ],
        )
        assert result.exit_code == 0
        assert "method=cusum" in result.stdout

    def test_cusum_without_target_errors(self, tmp_path: Path) -> None:
        scores_path = tmp_path / "scores.csv"
        _write_csv(scores_path, [0.5, 0.6, 0.7])
        result = runner.invoke(
            app, ["drift", str(scores_path), "--detector", "cusum"]
        )
        assert result.exit_code == 2
        assert "target-mean" in result.stderr.lower()

    def test_invalid_detector_errors(self, tmp_path: Path) -> None:
        scores_path = tmp_path / "scores.csv"
        _write_csv(scores_path, [0.5, 0.6, 0.7])
        result = runner.invoke(
            app, ["drift", str(scores_path), "--detector", "garbage"]
        )
        assert result.exit_code == 2

    def test_missing_file_errors(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["drift", str(tmp_path / "nope.csv")])
        assert result.exit_code == 1


# ============================================================================
# top-level
# ============================================================================


class TestTopLevel:
    def test_help_lists_all_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ("compare", "sample-size", "drift", "version"):
            assert cmd in result.stdout

    def test_no_args_prints_help(self) -> None:
        result = runner.invoke(app, [])
        # no_args_is_help=True → typer returns 2 and prints help
        assert "compare" in result.stdout
        assert "sample-size" in result.stdout


@pytest.fixture(autouse=True)
def _isolate_runner_stderr() -> None:
    """Ensure CliRunner captures stderr separately so error tests see it.

    Click's CliRunner defaults to mixing stderr into stdout; we want them
    separated so error-path assertions can target stderr explicitly.
    """
    runner.mix_stderr = False  # type: ignore[attr-defined]
