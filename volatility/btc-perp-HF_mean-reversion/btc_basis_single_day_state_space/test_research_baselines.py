from __future__ import annotations

import numpy as np
import pytest

from research_baselines import (
    ar1_ou_signal_z,
    causal_rolling_z,
    fit_ar1_ou_baseline,
)


def test_causal_rolling_z_prefix_is_invariant_to_future_changes():
    rng = np.random.default_rng(1)
    values = rng.normal(size=500)
    base = causal_rolling_z(values, window=50, min_periods=20)
    changed = values.copy()
    changed[300:] += 100.0
    altered = causal_rolling_z(changed, window=50, min_periods=20)
    np.testing.assert_allclose(base[:300], altered[:300], equal_nan=True)


def test_causal_rolling_z_excludes_current_observation_from_reference():
    values = np.array([0.0, 0.0, 1.0, 100.0])
    signal = causal_rolling_z(values, window=3, min_periods=2)
    expected = (100.0 - np.mean([0.0, 0.0, 1.0])) / np.std(
        [0.0, 0.0, 1.0], ddof=1
    )
    assert signal[-1] == pytest.approx(expected)


def test_ar1_fit_recovers_stationary_structure_and_signal():
    rng = np.random.default_rng(2)
    phi = 0.92
    mean = 0.003
    residual_sd = 0.0002
    values = np.empty(8_000)
    values[0] = mean
    for i in range(1, values.size):
        values[i] = mean * (1.0 - phi) + phi * values[i - 1] + rng.normal(
            scale=residual_sd
        )

    params = fit_ar1_ou_baseline(values, min_pairs=500)
    assert params.phi == pytest.approx(phi, rel=0.03)
    assert params.unconditional_mean == pytest.approx(mean, rel=0.15)
    assert params.half_life_seconds > 0.0

    signal = ar1_ou_signal_z(values, params)
    assert signal.shape == values.shape
    assert np.isfinite(signal).all()


def test_ar1_fit_rejects_nonstationary_positive_persistence_contract():
    values = np.arange(500, dtype=float)
    with pytest.raises(ValueError):
        fit_ar1_ou_baseline(values)
