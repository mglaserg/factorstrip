import numpy as np
import pandas as pd

from factorstrip.feasibility import (
    GateConfig,
    beta_adjusted_residuals,
    beta_neutralize_weights,
    estimate_lagged_shrunk_market_beta,
    raw_12_1_signal,
    run_blind_rho_gate,
    spanning_alpha_power_requirement,
    standardized_residual_12_1_signal,
)


def test_12_1_raw_signal_skips_current_month():
    idx = pd.date_range("2024-01-31", periods=13, freq="ME")
    monthly = pd.DataFrame({"A": [0.01] * 12 + [5.0]}, index=idx)

    signal = raw_12_1_signal(monthly, formation_months=11, skip_months=1)

    expected = (1.01**11) - 1.0
    assert np.isclose(signal.iloc[-1, 0], expected)


def test_residual_standardization_skips_current_month():
    idx = pd.date_range("2024-01-31", periods=13, freq="ME")
    values = np.arange(1.0, 14.0)
    monthly = pd.DataFrame({"A": values}, index=idx)

    signal = standardized_residual_12_1_signal(
        monthly, formation_months=11, skip_months=1
    )

    formation = values[-12:-1]
    expected = formation.sum() / formation.std(ddof=1)
    assert np.isclose(signal.iloc[-1, 0], expected)


def test_beta_is_lagged_before_residualization():
    idx = pd.bdate_range("2025-01-01", periods=8)
    market = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.015, -0.005, 0.01], index=idx)
    stocks = pd.DataFrame({"A": 2.0 * market.values}, index=idx)

    beta = estimate_lagged_shrunk_market_beta(
        stocks,
        market,
        window=4,
        min_periods=3,
        shrinkage=0.0,
        prior=1.0,
    )

    # First estimable beta occurs at the third observation, but shift(1) means
    # it cannot be used until the fourth observation.
    assert np.isnan(beta.loc[idx[2], "A"])
    assert np.isclose(beta.loc[idx[3], "A"], 2.0)

    residual = beta_adjusted_residuals(stocks, market, beta)
    assert np.isclose(residual.loc[idx[3], "A"], 0.0)


def test_beta_neutral_projection_enforces_constraints():
    idx = pd.to_datetime(["2026-01-31"])
    weights = pd.DataFrame(
        [[0.25, 0.25, -0.25, -0.25]], index=idx, columns=list("ABCD")
    )
    beta = pd.DataFrame(
        [[1.6, 1.2, 0.8, 0.5]], index=idx, columns=list("ABCD")
    )

    adjusted = beta_neutralize_weights(weights, beta, gross_exposure=1.0).iloc[0]
    assert np.isclose(adjusted.sum(), 0.0, atol=1e-12)
    assert np.isclose((adjusted * beta.iloc[0]).sum(), 0.0, atol=1e-12)
    assert np.isclose(adjusted.abs().sum(), 1.0, atol=1e-12)


def test_power_requirement_uses_two_trial_family_adjustment():
    result = spanning_alpha_power_requirement(
        rho=0.90,
        incremental_return_hurdle=0.03,
        target_vol=0.10,
        desired_power=0.80,
        family_significance=0.05,
        registered_trials=2,
    )

    assert np.isclose(result["effective_test_significance"], 0.025)
    assert 18.0 < result["required_years"] < 22.0


def test_blind_gate_output_contract_has_no_performance_metrics():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2000-01-03", "2026-08-31")
    n_names = 40
    market = pd.Series(rng.normal(0.0002, 0.01, len(dates)), index=dates)

    betas = np.linspace(0.6, 1.5, n_names)
    noise = rng.normal(0.0, 0.012, (len(dates), n_names))
    stocks = pd.DataFrame(
        market.to_numpy()[:, None] * betas[None, :] + noise,
        index=dates,
        columns=[f"S{i:02d}" for i in range(n_names)],
    )

    cfg = GateConfig(
        beta_window_days=126,
        beta_min_periods=63,
        bootstrap_samples=100,
        bootstrap_block_months=3,
        vol_window_months=6,
    )
    summary = run_blind_rho_gate(stocks, market, config=cfg)
    keys = {k.lower() for k in summary.to_dict()}

    forbidden = {"alpha", "sharpe", "cagr", "information_coefficient", "drawdown", "returns", "pnl"}
    for word in forbidden:
        assert not any(word in key for key in keys)

    assert summary.paired_months > 100
    assert -1.0 < summary.rho < 1.0
    if summary.rho_ci_lower > 0:
        assert summary.rho_for_power == summary.rho_ci_lower
    elif summary.rho_ci_upper < 0:
        assert summary.rho_for_power == summary.rho_ci_upper
    else:
        assert summary.rho_for_power == 0.0
