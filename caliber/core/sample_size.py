"""Sample-size calculator — entry point: `sample_size()`.

Answers "how many paired evals do I need to detect an effect of size X with
80% power?" via standard power analysis. Powered by `statsmodels.stats.power`.

Three test families are supported:
    paired_t     — paired t-test on differences (the default; matches the
                   `caliber.compare(..., method="paired_t")` setup).
    unpaired_t   — independent two-sample t-test, n per arm.
    proportion   — two-proportion z-test, n per arm; effect size must be
                   Cohen's h (use `effect_type="cohens_d"` — see notes).

Two effect-size conventions:
    cohens_d     — standardised effect (mean / std).
    absolute     — raw-units effect; we standardise by `baseline_std` for you.

References
----------
.. [1] Cohen, J. *Statistical Power Analysis for the Behavioral Sciences* (1988).
.. [2] statsmodels.stats.power.TTestPower / TTestIndPower / NormalIndPower.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from statsmodels.stats.power import NormalIndPower, TTestIndPower, TTestPower

from caliber.core.types import SampleSizeResult

Test = Literal["paired_t", "unpaired_t", "proportion"]
EffectType = Literal["cohens_d", "absolute"]


def sample_size(
    effect_size: float,
    *,
    effect_type: EffectType = "cohens_d",
    baseline_std: float | None = None,
    power: float = 0.8,
    confidence_level: float = 0.95,
    test: Test = "paired_t",
) -> SampleSizeResult:
    """Return the per-arm sample size needed to detect ``effect_size`` at the
    given ``power`` and ``confidence_level``.

    Parameters
    ----------
    effect_size : float
        The effect you want to detect, interpreted via ``effect_type``.
        For ``cohens_d`` this is Cohen's d (or Cohen's h for proportions —
        see notes). For ``absolute`` this is the raw difference in units of
        the metric (e.g. 0.05 = "5 accuracy points").
    effect_type : {"cohens_d", "absolute"}, default "cohens_d"
        How to interpret ``effect_size``. ``absolute`` requires ``baseline_std``
        so we can standardise.
    baseline_std : float | None, default None
        Required when ``effect_type="absolute"``. Standard deviation of the
        observations (paired differences for ``paired_t``, raw score std for
        ``unpaired_t``). Not used for ``proportion``.
    power : float, default 0.8
        Target statistical power (1 - β).
    confidence_level : float, default 0.95
        Two-sided significance level α = 1 - confidence_level.
    test : {"paired_t", "unpaired_t", "proportion"}, default "paired_t"
        Which test family to size for.

    Returns
    -------
    SampleSizeResult
        Required ``n_per_arm`` (number of paired observations for ``paired_t``;
        n per arm for the other tests), along with the standardised effect
        size that was actually used.

    Raises
    ------
    ValueError
        If parameters are out of range, ``effect_size`` is zero, or
        ``effect_type="absolute"`` was used without ``baseline_std``, or
        ``effect_type="absolute"`` was combined with ``test="proportion"``
        (the conversion is not well-defined without a baseline rate).

    Examples
    --------
    >>> from caliber import sample_size
    >>> r = sample_size(0.5, power=0.8)         # Cohen's d = 0.5
    >>> r.n_per_arm                              # doctest: +SKIP
    34
    >>> r = sample_size(0.05, effect_type="absolute", baseline_std=0.1)
    >>> r.n_per_arm                              # doctest: +SKIP
    34

    Notes
    -----
    For ``test="proportion"`` the standardised effect is Cohen's h, not d:
        h = 2·arcsin(√p₁) − 2·arcsin(√p₂)
    Pass ``effect_size`` as Cohen's h with ``effect_type="cohens_d"``.

    Formula (paired t-test, large n):
        n ≈ (z_{α/2} + z_β)² / d²
    statsmodels solves the exact iterative form (t-distribution rather than
    the normal approximation) via ``solve_power``.
    """
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1); got {power}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1); got {confidence_level}")
    if effect_size == 0.0:
        raise ValueError("effect_size must be nonzero (no effect is unbounded n)")

    if effect_type == "absolute":
        if test == "proportion":
            raise ValueError(
                "effect_type='absolute' is not supported for test='proportion' "
                "(Cohen's h conversion needs a baseline rate, not a single std). "
                "Pass effect_size as Cohen's h with effect_type='cohens_d'."
            )
        if baseline_std is None or baseline_std <= 0.0:
            raise ValueError(
                "baseline_std must be supplied and positive when "
                "effect_type='absolute'."
            )
        d = abs(effect_size) / baseline_std
    else:
        d = abs(effect_size)

    alpha = 1.0 - confidence_level

    analysis: TTestPower | TTestIndPower | NormalIndPower
    if test == "paired_t":
        analysis = TTestPower()
    elif test == "unpaired_t":
        analysis = TTestIndPower()
    elif test == "proportion":
        analysis = NormalIndPower()
    else:
        raise ValueError(f"unknown test family: {test!r}")

    n_raw = analysis.solve_power(
        effect_size=d,
        alpha=alpha,
        power=power,
        alternative="two-sided",
    )
    if not np.isfinite(float(n_raw)) or float(n_raw) <= 0.0:
        raise ValueError(
            f"power analysis returned non-positive n ({n_raw}); "
            f"check that effect_size and power are sensible."
        )
    n_per_arm = max(2, int(np.ceil(float(n_raw))))

    return SampleSizeResult(
        n_per_arm=n_per_arm,
        effect_size=d,
        power=power,
        confidence_level=confidence_level,
    )
