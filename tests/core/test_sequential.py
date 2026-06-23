"""Tests for caliber.core.sequential — group-sequential A/B tester."""

from __future__ import annotations

import numpy as np
import pytest

from caliber import CompareResult, SequentialTester


class TestValidation:
    def test_max_n_below_n_looks_raises(self) -> None:
        with pytest.raises(ValueError, match="max_n"):
            SequentialTester(max_n=3, n_looks=5)

    def test_zero_looks_raises(self) -> None:
        with pytest.raises(ValueError, match="n_looks"):
            SequentialTester(max_n=100, n_looks=0)

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            SequentialTester(max_n=100, n_looks=5, alpha=1.5)

    def test_unknown_boundary_raises(self) -> None:
        with pytest.raises(ValueError, match="boundary"):
            SequentialTester(max_n=100, n_looks=5, boundary="naive")  # type: ignore[arg-type]

    def test_pocock_non_default_alpha_raises(self) -> None:
        with pytest.raises(NotImplementedError, match=r"α=0\.05"):
            SequentialTester(max_n=100, n_looks=5, alpha=0.01, boundary="pocock")

    def test_pocock_unsupported_k_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="Pocock"):
            SequentialTester(max_n=100, n_looks=15, boundary="pocock")

    def test_mismatched_batch_lengths_raises(self) -> None:
        t = SequentialTester(max_n=100, n_looks=5)
        with pytest.raises(ValueError, match="batch lengths"):
            t.update([0.1, 0.2], [0.1])


class TestBasicShape:
    def test_returns_compare_result(self) -> None:
        t = SequentialTester(max_n=100, n_looks=5)
        rng = np.random.default_rng(0)
        r = t.update(
            rng.normal(0.5, 0.1, 10).tolist(),
            rng.normal(0.5, 0.1, 10).tolist(),
        )
        assert isinstance(r, CompareResult)

    def test_method_field_records_boundary(self) -> None:
        t = SequentialTester(max_n=100, n_looks=5, boundary="obrien_fleming")
        rng = np.random.default_rng(0)
        r = t.update(
            rng.normal(0.5, 0.1, 10).tolist(),
            rng.normal(0.5, 0.1, 10).tolist(),
        )
        assert "sequential" in r.method
        assert "obrien_fleming" in r.method

    def test_is_done_starts_false(self) -> None:
        t = SequentialTester(max_n=100, n_looks=5)
        assert not t.is_done()


class TestEarlyStopping:
    def test_stops_early_on_clear_better(self) -> None:
        t = SequentialTester(max_n=500, n_looks=5)
        rng = np.random.default_rng(0)
        # Huge effect — should cross OBF boundary at the very first look.
        old = rng.normal(0.0, 0.1, 100).tolist()
        new = rng.normal(2.0, 0.1, 100).tolist()
        r = t.update(old, new)
        assert r.verdict == "BETTER"
        assert t.is_done()

    def test_subsequent_updates_after_stop_return_same_result(self) -> None:
        t = SequentialTester(max_n=500, n_looks=5)
        rng = np.random.default_rng(0)
        r1 = t.update(
            rng.normal(0.0, 0.1, 100).tolist(),
            rng.normal(2.0, 0.1, 100).tolist(),
        )
        assert t.is_done()
        r2 = t.update([0.0], [0.0])  # post-stop call, no-op
        assert r2.verdict == r1.verdict
        assert r2.n == r1.n


class TestReachesFinalLook:
    def test_terminates_at_final_look_under_h0(self) -> None:
        # Under H₀, OBF should not reject; tester completes K looks and stops.
        rng = np.random.default_rng(0)
        t = SequentialTester(max_n=500, n_looks=5)
        for _ in range(5):
            old = rng.normal(0.5, 0.1, 100).tolist()
            new = rng.normal(0.5, 0.1, 100).tolist()
            r = t.update(old, new)
        assert t.is_done()
        assert r.verdict == "INCONCLUSIVE"


class TestFamilyWiseAlphaControl:
    """Critical: under H₀, P(any look rejects) ≤ α + slack across 1000 sims."""

    def test_obf_family_wise_alpha(self) -> None:
        rng = np.random.default_rng(42)
        n_sims = 500
        n_looks = 5
        batch_size = 30
        rejections = 0
        for _ in range(n_sims):
            t = SequentialTester(
                max_n=n_looks * batch_size, n_looks=n_looks, alpha=0.05,
            )
            for _ in range(n_looks):
                old = rng.normal(0.5, 0.1, batch_size).tolist()
                new = rng.normal(0.5, 0.1, batch_size).tolist()
                r = t.update(old, new)
                if r.verdict in ("BETTER", "WORSE"):
                    rejections += 1
                    break
        fwer = rejections / n_sims
        # OBF is slightly conservative; allow nominal α + a small slack.
        assert fwer <= 0.08, (
            f"OBF family-wise type-I rate {fwer:.3f}; "
            f"expected ≤ 0.08 (nominal α=0.05 plus slack)"
        )

    def test_pocock_family_wise_alpha(self) -> None:
        rng = np.random.default_rng(43)
        n_sims = 500
        n_looks = 5
        batch_size = 30
        rejections = 0
        for _ in range(n_sims):
            t = SequentialTester(
                max_n=n_looks * batch_size,
                n_looks=n_looks,
                alpha=0.05,
                boundary="pocock",
            )
            for _ in range(n_looks):
                old = rng.normal(0.5, 0.1, batch_size).tolist()
                new = rng.normal(0.5, 0.1, batch_size).tolist()
                r = t.update(old, new)
                if r.verdict in ("BETTER", "WORSE"):
                    rejections += 1
                    break
        fwer = rejections / n_sims
        assert fwer <= 0.08, (
            f"Pocock family-wise type-I rate {fwer:.3f}; "
            f"expected ≤ 0.08 (nominal α=0.05 plus slack)"
        )


class TestBoundaryShape:
    def test_obf_boundary_decreases_with_looks(self) -> None:
        """OBF starts very conservative, ends at the single-test threshold."""
        t = SequentialTester(max_n=500, n_looks=5)
        # Internal boundaries — exposed indirectly via test stat thresholds.
        boundaries = t._boundaries  # type: ignore[attr-defined]
        # Monotone decreasing — k=1 highest, k=K lowest.
        assert all(boundaries[i] > boundaries[i + 1] for i in range(len(boundaries) - 1))
        # Final boundary equals z_{α/2} ≈ 1.96 (since K/K = 1).
        assert abs(boundaries[-1] - 1.959964) < 1e-3

    def test_pocock_boundary_is_constant(self) -> None:
        t = SequentialTester(max_n=500, n_looks=5, boundary="pocock")
        boundaries = t._boundaries  # type: ignore[attr-defined]
        assert all(b == boundaries[0] for b in boundaries)
