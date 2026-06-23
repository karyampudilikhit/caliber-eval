"""Tests for caliber.core.drift — PageHinkleyDetector and CUSUMDetector.

Critical test (per PRD): on a stationary stream of 10k samples, the FPR
must be acceptably low (≤ ~5% in standard units) with default conservative
parameters. Both detectors are exercised against this bound.
"""

from __future__ import annotations

import numpy as np
import pytest

from caliber import CUSUMDetector, DriftEvent, PageHinkleyDetector

# ============================================================================
# Page-Hinkley
# ============================================================================


class TestPageHinkleyValidation:
    def test_negative_delta_raises(self) -> None:
        with pytest.raises(ValueError, match="delta"):
            PageHinkleyDetector(delta=-0.1)

    def test_nonpositive_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            PageHinkleyDetector(threshold=0.0)


class TestPageHinkleyBasics:
    def test_first_few_observations_return_none(self) -> None:
        det = PageHinkleyDetector(delta=0.05, threshold=5.0)
        # Stationary noise — nothing should fire in the first few samples.
        rng = np.random.default_rng(0)
        for _ in range(20):
            assert det.add(rng.normal(0.5, 0.05)) is None

    def test_reset_clears_state(self) -> None:
        det = PageHinkleyDetector(delta=0.05, threshold=5.0)
        rng = np.random.default_rng(0)
        # Push enough drifty data to fire.
        for _ in range(100):
            det.add(rng.normal(0.5, 0.05))
        for _ in range(1000):
            event = det.add(rng.normal(2.0, 0.05))
            if event is not None:
                break
        # After reset, internal state is fresh.
        det.reset()
        # First call after reset should not fire on a single sample.
        assert det.add(rng.normal(0.5, 0.05)) is None

    def test_timestamp_is_round_tripped(self) -> None:
        det = PageHinkleyDetector(delta=0.05, threshold=1.0)
        # Force a fast firing with extreme drift.
        det.add(0.0, timestamp=100.0)
        for i in range(1, 50):
            event = det.add(10.0, timestamp=100.0 + i)
            if event is not None:
                assert event.detected_at_timestamp == 100.0 + i
                break
        else:
            pytest.fail("expected a drift event with a large mean shift")


class TestPageHinkleyDetection:
    def test_detects_upward_mean_shift(self) -> None:
        rng = np.random.default_rng(0)
        det = PageHinkleyDetector(delta=0.05, threshold=5.0)
        # Pre-drift: stable at 0.5.
        for _ in range(300):
            det.add(rng.normal(0.5, 0.1))
        # Post-drift: shift to 0.8.
        fired_at: int | None = None
        for i in range(1000):
            event = det.add(rng.normal(0.8, 0.1))
            if event is not None:
                fired_at = i
                assert event.mean_after > event.mean_before
                assert event.magnitude > 0.1
                assert event.method == "page_hinkley"
                break
        assert fired_at is not None, "Page-Hinkley failed to detect a +0.3 mean shift"

    def test_detects_downward_mean_shift(self) -> None:
        rng = np.random.default_rng(0)
        det = PageHinkleyDetector(delta=0.05, threshold=5.0)
        for _ in range(300):
            det.add(rng.normal(0.7, 0.1))
        fired = False
        for _ in range(1000):
            event = det.add(rng.normal(0.4, 0.1))
            if event is not None:
                assert event.mean_after < event.mean_before
                assert event.magnitude > 0.1
                fired = True
                break
        assert fired, "Page-Hinkley failed to detect a -0.3 mean shift"

    def test_returns_drift_event_with_all_fields(self) -> None:
        rng = np.random.default_rng(0)
        det = PageHinkleyDetector(delta=0.05, threshold=1.0)
        # Pump aggressive drift to force quick firing.
        for _ in range(20):
            det.add(rng.normal(0.0, 0.05))
        for _ in range(500):
            event = det.add(rng.normal(2.0, 0.05))
            if event is not None:
                assert isinstance(event, DriftEvent)
                assert event.detected_at_index >= 0
                assert event.method == "page_hinkley"
                assert 0.0 <= event.p_value <= 1.0
                return
        pytest.fail("Page-Hinkley failed to fire on aggressive drift")


class TestPageHinkleyFalsePositiveRate:
    """Critical: on a 10k stationary stream with default params, FPR is low."""

    def test_fpr_on_stationary_stream(self) -> None:
        rng = np.random.default_rng(42)
        # Default params: delta=0.05, threshold=50 — very conservative.
        det = PageHinkleyDetector()
        false_positives = 0
        for _ in range(10_000):
            score = rng.normal(0.7, 0.1)
            if det.add(score) is not None:
                false_positives += 1
                det.reset()
        # Default params chosen for very low FPR — expect zero or near-zero.
        assert false_positives <= 5, (
            f"FPR too high: {false_positives} false alarms in 10k stationary "
            f"samples at default thresholds"
        )


# ============================================================================
# CUSUM
# ============================================================================


class TestCUSUMValidation:
    def test_nonpositive_std_raises(self) -> None:
        with pytest.raises(ValueError, match="target_std"):
            CUSUMDetector(target_mean=0.0, target_std=0.0)

    def test_negative_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k"):
            CUSUMDetector(target_mean=0.0, k=-1.0)

    def test_nonpositive_h_raises(self) -> None:
        with pytest.raises(ValueError, match="h"):
            CUSUMDetector(target_mean=0.0, h=0.0)


class TestCUSUMBasics:
    def test_no_fire_at_target_mean(self) -> None:
        # CUSUM at h=5 has a finite in-control ARL (~465); over 50 samples a
        # false alarm is possible. Use h=15 for this sanity check — the real
        # FPR bound lives in TestCUSUMFalsePositiveRate below.
        rng = np.random.default_rng(0)
        det = CUSUMDetector(target_mean=0.7, target_std=0.1, k=0.5, h=15.0)
        for _ in range(50):
            assert det.add(rng.normal(0.7, 0.1)) is None

    def test_reset_clears_state(self) -> None:
        rng = np.random.default_rng(0)
        det = CUSUMDetector(target_mean=0.5, target_std=0.1, k=0.5, h=4.0)
        for _ in range(500):
            det.add(rng.normal(1.0, 0.05))  # well above target
        det.reset()
        # Fresh — a single in-target sample at h=15 cannot fire.
        det2 = CUSUMDetector(target_mean=0.5, target_std=0.1, k=0.5, h=15.0)
        # Use a separate det to keep the reset test pure; reuse it on `det` too.
        assert det.add(0.5) is None
        assert det2.add(0.5) is None


class TestCUSUMDetection:
    def test_detects_upward_shift(self) -> None:
        rng = np.random.default_rng(0)
        det = CUSUMDetector(target_mean=0.5, target_std=0.1, k=0.5, h=5.0)
        for _ in range(100):
            det.add(rng.normal(0.5, 0.1))
        fired = False
        for _ in range(500):
            event = det.add(rng.normal(0.8, 0.1))
            if event is not None:
                assert event.mean_after > event.mean_before
                assert event.method == "cusum"
                fired = True
                break
        assert fired, "CUSUM failed to detect upward shift"

    def test_detects_downward_shift(self) -> None:
        rng = np.random.default_rng(0)
        det = CUSUMDetector(target_mean=0.7, target_std=0.1, k=0.5, h=5.0)
        for _ in range(100):
            det.add(rng.normal(0.7, 0.1))
        fired = False
        for _ in range(500):
            event = det.add(rng.normal(0.4, 0.1))
            if event is not None:
                assert event.mean_after < event.mean_before
                fired = True
                break
        assert fired, "CUSUM failed to detect downward shift"


class TestCUSUMFalsePositiveRate:
    """Critical: with conservative h=5, false alarms on stationary stream are rare."""

    def test_fpr_on_stationary_stream(self) -> None:
        rng = np.random.default_rng(42)
        det = CUSUMDetector(target_mean=0.7, target_std=0.1, k=0.5, h=5.0)
        false_positives = 0
        for _ in range(10_000):
            if det.add(rng.normal(0.7, 0.1)) is not None:
                false_positives += 1
                det.reset()
        # Standard tabular CUSUM at h=5 has an average run length of ~465 in
        # the standardised normal case; over 10k samples we'd expect roughly
        # 10k/465 ≈ 22 false alarms. Cap empirically at 40 with slack.
        assert false_positives <= 40, (
            f"CUSUM FPR too high: {false_positives} alarms in 10k stationary "
            f"samples at h=5"
        )
