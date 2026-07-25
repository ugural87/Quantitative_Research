"""Tests for the state-space basis model.

Two properties make this module usable for a backtest: (1) the filter is
strictly causal, so no reported out-of-sample signal can depend on a future
observation, and (2) fitting on data simulated from known parameters recovers
the underlying structure rather than fitting noise. The suite also locks the
online filter to the batch filter and checks the parameter validation contract.

The estimator runs a multi-start MLE, so a single fit is reused across the
tests that need one via a module-scoped fixture; without the optional numba
dependency one fit takes tens of seconds and re-fitting per test would be
wasteful.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from state_space import (
    OnlineBasisFilter,
    StateSpaceParams,
    filter_state_space,
    fit_state_space,
    no_lookahead_check,
    simulate_state_space,
)

# A persistent transient on a slowly drifting level, at 1 Hz: a half-life of a
# few seconds and a level that moves far more slowly than the transient. This
# is the regime the notebook studies.
TRUE_PARAMS = StateSpaceParams(
    b=0.90,
    sigma_level=2.0e-5,
    sigma_transient=8.0e-5,
    sigma_observation=1.0e-5,
    dt_seconds=1.0,
)

N_FIT = 6_000
BURN_IN = 50


@pytest.fixture(scope="module")
def simulated():
    return simulate_state_space(N_FIT, TRUE_PARAMS, seed=3)


@pytest.fixture(scope="module")
def fit(simulated):
    return fit_state_space(
        simulated["observed"], dt=1.0, burn_in=BURN_IN, compare_null=True
    )


# --------------------------------------------------------------------------- #
# Parameter object contract
# --------------------------------------------------------------------------- #
def test_params_reject_non_stationary_persistence():
    with pytest.raises(ValueError):
        StateSpaceParams(b=1.0, sigma_level=1e-5, sigma_transient=1e-5,
                         sigma_observation=1e-5)


def test_params_reject_non_positive_sigma():
    with pytest.raises(ValueError):
        StateSpaceParams(b=0.9, sigma_level=0.0, sigma_transient=1e-5,
                         sigma_observation=1e-5)


def test_half_life_matches_persistence():
    expected = TRUE_PARAMS.dt_seconds * math.log(2.0) / -math.log(TRUE_PARAMS.b)
    assert TRUE_PARAMS.half_life_seconds == pytest.approx(expected, rel=1e-12)


def test_expected_fraction_reverted_is_monotone():
    p = TRUE_PARAMS
    fractions = [p.expected_fraction_reverted(h) for h in (0, 1, 5, 30, 300)]
    assert fractions[0] == pytest.approx(0.0)
    assert all(b >= a for a, b in zip(fractions, fractions[1:]))
    assert fractions[-1] < 1.0


# --------------------------------------------------------------------------- #
# Causality — the property a backtest relies on
# --------------------------------------------------------------------------- #
def test_no_lookahead_exact():
    y = simulate_state_space(4_000, TRUE_PARAMS, seed=1)["observed"]
    assert no_lookahead_check(y, TRUE_PARAMS, split=2_000,
                              perturbation=1_000.0, atol=1e-12)


def test_filter_prefix_is_invariant_to_the_future():
    y = simulate_state_space(3_000, TRUE_PARAMS, seed=2)["observed"]
    split = 1_500
    base = filter_state_space(y, TRUE_PARAMS).transient
    altered = y.copy()
    altered[split:] += 0.05  # a large, structured shock to the future
    moved = filter_state_space(altered, TRUE_PARAMS).transient
    np.testing.assert_allclose(base[:split], moved[:split], atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------- #
# Estimation recovers structure and prefers the true model to the null
# --------------------------------------------------------------------------- #
def test_fit_converges_and_recovers_half_life(fit):
    assert fit.converged
    # Half-life is the economically meaningful quantity; require it within a
    # factor of two of truth rather than pinning individual variance terms,
    # which are only weakly identified in this model class.
    ratio = fit.half_life / TRUE_PARAMS.half_life_seconds
    assert 0.5 <= ratio <= 2.0


def test_fitted_transient_tracks_the_true_transient(simulated, fit):
    est = filter_state_space(simulated["observed"], fit).transient
    # Compare on the tail, after the diffuse initial uncertainty has washed out.
    corr = np.corrcoef(est[1_000:], simulated["transient"][1_000:])[0, 1]
    assert corr > 0.5


def test_transient_model_beats_local_level_null(fit):
    # Positive delta => the AR(1) transient lowers BIC versus a pure local-level
    # model, the correct verdict for this data-generating process.
    assert fit.delta_bic_vs_local_level is not None
    assert fit.delta_bic_vs_local_level > 0.0


def test_fit_rejects_series_shorter_than_min_obs():
    y = simulate_state_space(400, TRUE_PARAMS, seed=6)["observed"]
    with pytest.raises(ValueError):
        fit_state_space(y, dt=1.0, min_obs=500)


# --------------------------------------------------------------------------- #
# Online filter equals the batch filter observation-for-observation
# --------------------------------------------------------------------------- #
def test_online_filter_matches_batch_filter():
    y = simulate_state_space(2_000, TRUE_PARAMS, seed=7)["observed"]
    batch = filter_state_space(y, TRUE_PARAMS)

    online = OnlineBasisFilter(TRUE_PARAMS, initial_observation=float(y[0]))
    online_transient = [online.update(float(v))["transient"] for v in y]

    np.testing.assert_allclose(
        np.asarray(online_transient), batch.transient, atol=1e-9, rtol=1e-6
    )


def test_online_filter_handles_missing_observation():
    online = OnlineBasisFilter(TRUE_PARAMS, initial_observation=0.0)
    online.update(1e-4)
    state = online.update(None)  # a gap: prediction carries the state forward
    assert math.isfinite(state["transient"])
    assert math.isfinite(state["level"])


# --------------------------------------------------------------------------- #
# Innovations are close to white when the model is correct
# --------------------------------------------------------------------------- #
def test_standardized_innovations_are_approximately_unit_variance(simulated, fit):
    diag = filter_state_space(simulated["observed"], fit).diagnostics(max_lag=20)
    assert abs(diag["mean"]) < 0.1
    assert 0.8 < diag["sd"] < 1.2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
