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

# --------------------------------------------------------------------------- #
# Explicit signal semantics and research gates
# --------------------------------------------------------------------------- #
def test_signal_scalings_are_explicit():
    y = simulate_state_space(2_000, TRUE_PARAMS, seed=11)["observed"]
    result = filter_state_space(y, TRUE_PARAMS)

    np.testing.assert_allclose(
        result.transient_stationary_z(TRUE_PARAMS),
        result.transient / TRUE_PARAMS.transient_sd,
    )
    np.testing.assert_allclose(
        result.transient_filter_t,
        result.transient / np.sqrt(result.filtered_variance_transient),
    )


def test_online_signal_matches_stationary_batch_definition():
    y = simulate_state_space(1_000, TRUE_PARAMS, seed=12)["observed"]
    batch = filter_state_space(y, TRUE_PARAMS)
    online = OnlineBasisFilter(TRUE_PARAMS, initial_observation=float(y[0]))
    online_z = [online.update(float(v))["transient_stationary_z"] for v in y]

    np.testing.assert_allclose(
        np.asarray(online_z),
        batch.transient_stationary_z(TRUE_PARAMS),
        atol=1e-9,
        rtol=1e-6,
    )


def test_adequacy_gate_rejects_serially_correlated_innovations():
    rng = np.random.default_rng(13)
    z = np.zeros(2_000)
    for i in range(1, z.size):
        z[i] = 0.85 * z[i - 1] + rng.normal(scale=0.5)

    result = filter_state_space(
        simulate_state_space(2_000, TRUE_PARAMS, seed=14)["observed"],
        TRUE_PARAMS,
    )
    result.standardized_innovation = z
    report = result.adequacy_report(max_lag=20, alpha=0.01)
    assert not report.dynamic_passed
    assert any("autocorrelation" in item for item in report.failures)


def test_fit_json_round_trip_preserves_parameters_and_metadata(tmp_path, fit):
    path = tmp_path / "fit.json"
    fit.save_json(path)
    restored = type(fit).load_json(path)

    assert restored.params == fit.params
    assert restored.loglik == pytest.approx(fit.loglik)
    assert restored.sample_mode == fit.sample_mode
    assert restored.optimizer_runs == fit.optimizer_runs
    assert restored.null_bic == pytest.approx(fit.null_bic)


def test_uniform_sample_mode_is_rejected_even_without_a_cap(simulated):
    with pytest.raises(ValueError, match="uniform thinning"):
        fit_state_space(
            simulated["observed"],
            sample_mode="uniform",  # type: ignore[arg-type]
            compare_null=False,
        )


def test_head_and_tail_caps_remain_contiguous(simulated):
    head = fit_state_space(
        simulated["observed"],
        max_obs=1_000,
        sample_mode="head",
        compare_null=False,
    )
    tail = fit_state_space(
        simulated["observed"],
        max_obs=1_000,
        sample_mode="tail",
        compare_null=False,
    )
    assert head.n_obs == 1_000
    assert tail.n_obs == 1_000
    assert head.source_n_obs == simulated["observed"].size
    assert tail.source_n_obs == simulated["observed"].size


def test_adequacy_report_rejects_insufficient_innovations():
    result = filter_state_space(
        simulate_state_space(20, TRUE_PARAMS, seed=31)["observed"],
        TRUE_PARAMS,
    )
    report = result.adequacy_report(max_lag=10)
    assert not report.passed
    assert "insufficient finite innovations" in report.failures


def test_online_snapshot_and_history_contract(simulated):
    history = simulated["observed"][:500]
    online = OnlineBasisFilter.from_history(TRUE_PARAMS, history)
    snapshot = online.snapshot()
    assert online.n_updates == 500
    assert set(snapshot) == {"level", "transient", "p11", "p12", "p22"}
    assert all(math.isfinite(value) for value in snapshot.values())


def test_leading_missing_states_do_not_use_first_future_observation():
    a = np.array([np.nan, np.nan, 1.0, 1.1])
    b = np.array([np.nan, np.nan, 2.0, 2.1])
    result_a = filter_state_space(a, TRUE_PARAMS)
    result_b = filter_state_space(b, TRUE_PARAMS)

    assert np.isnan(result_a.level[:2]).all()
    assert np.isnan(result_b.level[:2]).all()
    assert np.isnan(result_a.transient[:2]).all()
    assert np.isnan(result_b.transient[:2]).all()


def test_gap_aware_diagnostics_do_not_compress_missing_clock_steps():
    from state_space import FilterResult

    n = 200
    z = np.full(n, np.nan)
    z[::2] = np.random.default_rng(41).normal(size=n // 2)
    zeros = np.zeros(n)
    result = FilterResult(
        level=zeros,
        transient=zeros,
        innovation=z,
        innovation_variance=np.ones(n),
        standardized_innovation=z,
        filtered_variance_level=zeros,
        filtered_covariance=zeros,
        filtered_variance_transient=zeros,
        observed=zeros,
        loglik_contribution=zeros,
    )
    diagnostics = result.diagnostics(max_lag=1)
    assert diagnostics["minimum_exact_lag_pairs"] == 0.0
    assert math.isnan(diagnostics["ljung_box_p"])
    report = result.adequacy_report(max_lag=1)
    assert not report.dynamic_passed
    assert "insufficient exact-clock innovation pairs" in report.failures


def test_filter_rejects_negative_burn_in(simulated):
    with pytest.raises(ValueError, match="burn_in"):
        filter_state_space(simulated["observed"], TRUE_PARAMS, burn_in=-1)


def test_burn_in_is_counted_from_first_observation_not_from_leading_gap():
    from state_space import _filter_core

    observed = simulate_state_space(100, TRUE_PARAMS, seed=52)["observed"]
    y = np.r_[np.full(7, np.nan), observed]
    _, _, _, effective = _filter_core(
        y,
        TRUE_PARAMS,
        burn_in=10,
        store=False,
    )
    assert effective == 90


def test_gap_aware_autocorrelation_is_not_diluted_by_random_missingness():
    from state_space import FilterResult

    rng = np.random.default_rng(53)
    n = 8_000
    phi = 0.8
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + rng.normal()
    z[rng.random(n) < 0.5] = np.nan
    zeros = np.zeros(n)
    result = FilterResult(
        level=zeros,
        transient=zeros,
        innovation=z,
        innovation_variance=np.ones(n),
        standardized_innovation=z,
        filtered_variance_level=zeros,
        filtered_covariance=zeros,
        filtered_variance_transient=zeros,
        observed=zeros,
        loglik_contribution=zeros,
    )
    diagnostics = result.diagnostics(max_lag=1)
    assert diagnostics["minimum_exact_lag_pairs"] > 1_000
    assert diagnostics["lag1_autocorrelation"] > 0.70
    assert diagnostics["ljung_box_p"] < 1e-12
