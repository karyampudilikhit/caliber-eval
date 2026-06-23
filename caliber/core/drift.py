"""Streaming change-point detectors for production score drift.

Two detectors are exposed:

`PageHinkleyDetector`
    The Page-Hinkley test (Page 1954). Maintains two cumulative sums — one
    sensitive to upward drift, one to downward — each biased by a small
    constant ``delta`` so a stationary stream wanders without alarming. When
    the cumulative sum exceeds ``threshold`` above (or below) its running
    extremum, drift is declared. Does NOT require knowing the target mean
    in advance; it learns the baseline from the stream.

`CUSUMDetector`
    The classical tabular CUSUM (Page 1954). Requires a known ``target_mean``
    and reference noise scale ``target_std``. Standardises observations and
    accumulates only the portion that exceeds the slack parameter ``k``;
    fires when the cumulative deviation crosses ``h``. Tighter and more
    sensitive than Page-Hinkley when you know the target; useless when
    you don't.

Both detectors keep an internal log of observed scores so that, when drift
fires, they can estimate ``mean_before`` and ``mean_after`` using the
approximate change-point — the index where the cumulative sum reached its
extremum. The log is cleared on ``reset()``. Memory grows linearly in
observations between resets; fine for typical eval-stream sizes.

References
----------
.. [1] Page, E.S. "Continuous inspection schemes." Biometrika 41 (1954): 100-115.
.. [2] Hawkins, D.M. & Olwell, D.H. *Cumulative Sum Charts and Charting for
       Quality Improvement* (1998).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from caliber.core.types import DriftEvent


class PageHinkleyDetector:
    """Two-sided Page-Hinkley change-point detector.

    Parameters
    ----------
    delta : float, default 0.05
        Tolerance — subtracted from each cumulative-sum increment so that under
        H₀ the cumsum drifts toward zero and resets frequently. Should be a
        fraction of the noise scale of your scores; the default 0.05 is sized
        for eval scores in [0, 1] with noise σ ≈ 0.1. Smaller ``delta`` →
        more sensitive but slower change-point recovery; larger → more
        conservative.
    threshold : float, default 50.0
        Cumulative deviation required to declare drift. Larger ``threshold``
        means longer expected run length (fewer false alarms, slower detection).
        Must be tuned to the noise scale of your scores.

    Examples
    --------
    >>> from caliber import PageHinkleyDetector
    >>> det = PageHinkleyDetector(delta=0.05, threshold=5.0)
    >>> for score in stream:                       # doctest: +SKIP
    ...     event = det.add(score)
    ...     if event is not None:
    ...         print(f"Drift at idx {event.detected_at_index}")
    """

    def __init__(self, delta: float = 0.05, threshold: float = 50.0) -> None:
        if delta < 0.0:
            raise ValueError(f"delta must be ≥ 0; got {delta}")
        if threshold <= 0.0:
            raise ValueError(f"threshold must be > 0; got {threshold}")
        self.delta = delta
        self.threshold = threshold
        self._scores: list[float] = []
        self._timestamps: list[float | None] = []
        self._n = 0
        self._running_sum = 0.0
        # Resetting tabular CUSUM-style accumulators (Page 1954, modern form).
        # Each is clamped at 0; the index of the last reset is the change-point
        # estimate, since under H₀ the −delta term keeps the cumsum at 0.
        self._u_pos = 0.0
        self._u_pos_reset_idx = 0
        self._u_neg = 0.0
        self._u_neg_reset_idx = 0

    def reset(self) -> None:
        """Drop all state and treat subsequent observations as a fresh stream."""
        self._scores.clear()
        self._timestamps.clear()
        self._n = 0
        self._running_sum = 0.0
        self._u_pos = 0.0
        self._u_pos_reset_idx = 0
        self._u_neg = 0.0
        self._u_neg_reset_idx = 0

    def add(
        self, score: float, timestamp: float | None = None
    ) -> DriftEvent | None:
        """Feed one observation. Returns a `DriftEvent` if drift just fired, else `None`.

        The detector continues running after firing — call ``reset()`` if you
        want a fresh window (typical pattern: reset after handling the event).
        """
        self._scores.append(float(score))
        self._timestamps.append(timestamp)
        self._n += 1
        self._running_sum += float(score)
        running_mean = self._running_sum / self._n
        idx = self._n - 1

        # Standard tabular Page-Hinkley form. Under H₀ the −delta drift keeps
        # both cumsums clamped at 0 most of the time, so the last-reset index
        # is a clean change-point estimate. Under a genuine shift, one side
        # accumulates rapidly until it exceeds `threshold`.
        u_pos_new = max(0.0, self._u_pos + (float(score) - running_mean) - self.delta)
        if u_pos_new == 0.0:
            self._u_pos_reset_idx = idx
        self._u_pos = u_pos_new

        u_neg_new = max(0.0, self._u_neg - (float(score) - running_mean) - self.delta)
        if u_neg_new == 0.0:
            self._u_neg_reset_idx = idx
        self._u_neg = u_neg_new

        if self._u_pos > self.threshold:
            return self._build_event(self._u_pos_reset_idx)
        if self._u_neg > self.threshold:
            return self._build_event(self._u_neg_reset_idx)
        return None

    def _build_event(self, change_point_idx: int) -> DriftEvent:
        """Construct the DriftEvent. Welch's t-test on the before/after split is
        reported as a post-hoc p-value; the firing decision is the PH statistic."""
        mean_before, mean_after, p_value = _split_stats(
            self._scores, change_point_idx
        )
        return DriftEvent(
            detected_at_index=self._n - 1,
            detected_at_timestamp=self._timestamps[-1],
            mean_before=mean_before,
            mean_after=mean_after,
            magnitude=abs(mean_after - mean_before),
            p_value=p_value,
            method="page_hinkley",
        )


class CUSUMDetector:
    """Tabular two-sided CUSUM for streams with a known target mean.

    Parameters
    ----------
    target_mean : float
        The mean you expect under H₀.
    target_std : float, default 1.0
        Reference scale for standardisation. Should be the known/expected
        standard deviation of the stream under H₀.
    k : float, default 0.5
        Reference value (allowance), in units of ``target_std``. Half the
        size of the shift you want to detect quickly. Smaller ``k`` →
        faster detection of small shifts, more false alarms.
    h : float, default 5.0
        Decision threshold, in units of ``target_std``. Typical choice 4–5;
        higher → fewer false alarms, slower detection.

    Notes
    -----
    CUSUM is more sensitive than Page-Hinkley when the target mean and noise
    scale are known. Use it for stationary processes with known targets;
    use Page-Hinkley when you need to learn the baseline from data.
    """

    def __init__(
        self,
        target_mean: float,
        target_std: float = 1.0,
        k: float = 0.5,
        h: float = 5.0,
    ) -> None:
        if target_std <= 0.0:
            raise ValueError(f"target_std must be > 0; got {target_std}")
        if k < 0.0:
            raise ValueError(f"k must be ≥ 0; got {k}")
        if h <= 0.0:
            raise ValueError(f"h must be > 0; got {h}")
        self.target_mean = target_mean
        self.target_std = target_std
        self.k = k
        self.h = h
        self._scores: list[float] = []
        self._timestamps: list[float | None] = []
        self._n = 0
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._last_zero_pos = 0  # last index where s_pos reset to 0
        self._last_zero_neg = 0

    def reset(self) -> None:
        self._scores.clear()
        self._timestamps.clear()
        self._n = 0
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._last_zero_pos = 0
        self._last_zero_neg = 0

    def add(
        self, score: float, timestamp: float | None = None
    ) -> DriftEvent | None:
        self._scores.append(float(score))
        self._timestamps.append(timestamp)
        self._n += 1

        z = (float(score) - self.target_mean) / self.target_std
        # Two-sided tabular CUSUM:
        #   S_pos_t = max(0, S_pos_{t-1} + z_t - k)
        #   S_neg_t = max(0, S_neg_{t-1} - z_t - k)
        s_pos_new = max(0.0, self._s_pos + z - self.k)
        s_neg_new = max(0.0, self._s_neg - z - self.k)
        if s_pos_new == 0.0:
            self._last_zero_pos = self._n - 1
        if s_neg_new == 0.0:
            self._last_zero_neg = self._n - 1
        self._s_pos = s_pos_new
        self._s_neg = s_neg_new

        if self._s_pos > self.h:
            return self._build_event(self._last_zero_pos)
        if self._s_neg > self.h:
            return self._build_event(self._last_zero_neg)
        return None

    def _build_event(self, change_point_idx: int) -> DriftEvent:
        mean_before, mean_after, p_value = _split_stats(
            self._scores, change_point_idx
        )
        return DriftEvent(
            detected_at_index=self._n - 1,
            detected_at_timestamp=self._timestamps[-1],
            mean_before=mean_before,
            mean_after=mean_after,
            magnitude=abs(mean_after - mean_before),
            p_value=p_value,
            method="cusum",
        )


def _split_stats(
    scores: list[float], change_point_idx: int
) -> tuple[float, float, float]:
    """Compute mean_before, mean_after, and the Welch's-t p-value for the split.

    Clamps ``change_point_idx`` so both sides have at least one element. If the
    stream has fewer than 2 observations there's nothing to split on — we return
    the single observation as both means with p=1.
    """
    n = len(scores)
    if n < 2:
        single = float(scores[0]) if scores else 0.0
        return single, single, 1.0
    # Clamp into [0, n-2] so before has ≥1 element and after has ≥1 element.
    cp = max(0, min(n - 2, change_point_idx))
    before = np.asarray(scores[: cp + 1], dtype=np.float64)
    after = np.asarray(scores[cp + 1 :], dtype=np.float64)
    mean_before = float(np.mean(before))
    mean_after = float(np.mean(after))
    if before.size > 1 and after.size > 1:
        res = stats.ttest_ind(after, before, equal_var=False)
        p_value = float(res.pvalue)
        if not np.isfinite(p_value):
            p_value = 1.0
    else:
        p_value = 1.0
    return mean_before, mean_after, p_value
