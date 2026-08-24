from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .loader import TLAQData


TRADING_DAYS = 252
ASSETS = ["VTI", "TLT", "GLD", "SVXY", "VIXY"]


@dataclass
class OverlayResult:
    name: str
    returns: pd.Series
    target_weights: pd.DataFrame
    trigger: pd.Series
    incremental_turnover: pd.Series
    rolling_beta: pd.Series | None = None


def performance_summary(returns: pd.Series) -> pd.Series:
    r = returns.dropna()

    if r.empty:
        return pd.Series(dtype=float)

    equity = (1.0 + r).cumprod()
    years = len(r) / TRADING_DAYS

    cagr = (
        equity.iloc[-1] ** (1.0 / years) - 1.0
        if years > 0 and equity.iloc[-1] > 0
        else np.nan
    )

    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (
        r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)
        if r.std(ddof=1) > 0
        else np.nan
    )

    dd = equity / equity.cummax() - 1.0

    return pd.Series(
        {
            "CAGR": cagr,
            "Annualized Vol": vol,
            "Sharpe": sharpe,
            "Max Drawdown": dd.min(),
            "Ending Multiple": equity.iloc[-1],
        }
    )


def rolling_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Rolling beta using observations through each date.

    The overlay functions shift this beta by one day before using it as a
    trading trigger, so the test is look-ahead safe under a close-to-close
    backtest convention.
    """
    if min_periods is None:
        min_periods = max(20, window // 2)

    cov = strategy_returns.rolling(
        window,
        min_periods=min_periods,
    ).cov(benchmark_returns)

    var = benchmark_returns.rolling(
        window,
        min_periods=min_periods,
    ).var()

    return (cov / var).rename("rolling_beta")


def _base_return_from_weights(
    data: TLAQData,
    target_weights: pd.DataFrame,
    incremental_cost_bps: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Reconstruct close-to-close returns from target end-of-day weights.

    The end-of-day target at t earns the asset return from t to t+1.

    Existing commissions/interest/borrow costs from TLAQ are retained.
    An optional additional cost is charged only for incremental SVXY
    turnover caused by the overlay.
    """
    target_weights = target_weights.reindex(
        index=data.weights.index,
        columns=ASSETS,
    ).fillna(0.0)

    asset_component = (
        target_weights.shift(1)
        * data.asset_returns[ASSETS]
    ).sum(axis=1, min_count=1)

    fee_return = (
        -data.commissions
        + data.interest
        - data.short_borrow_cost
    ).div(data.nav.shift(1))

    # Difference between overlay and original SVXY target.
    adjustment = (
        target_weights["SVXY"]
        - data.weights["SVXY"]
    )

    incremental_turnover = (
        adjustment.diff().abs()
        .fillna(adjustment.abs())
        .rename("incremental_svxy_turnover")
    )

    overlay_cost = (
        incremental_turnover
        * (incremental_cost_bps / 10_000.0)
    )

    result = (
        asset_component
        + fee_return
        - overlay_cost
    ).rename("overlay_return")

    return result, incremental_turnover


def beta_conditioned_svxy_overlay(
    data: TLAQData,
    beta_threshold: float = 1.50,
    svxy_scale: float = 0.50,
    beta_window: int = 60,
    incremental_cost_bps: float = 0.0,
) -> OverlayResult:
    """
    Reduce only SVXY when TLAQ's PRIOR-DAY rolling VTI beta is high.

    Example:
        beta_threshold = 1.50
        svxy_scale = 0.50

    means:
        if yesterday's 60d TLAQ/VTI beta > 1.50
        and today's original TLAQ target is SVXY,
        hold 50% of the prescribed SVXY weight.

    Freed exposure is implicitly held as cash. VIXY is never reduced.

    Important:
        beta.shift(1) is used deliberately. This avoids using today's
        closing return to change a position assumed to be established
        at today's close.
    """
    if not 0.0 <= svxy_scale <= 1.0:
        raise ValueError("svxy_scale must be between 0 and 1.")

    beta = rolling_beta(
        data.strategy_returns,
        data.asset_returns["VTI"],
        window=beta_window,
    )

    prior_beta = beta.shift(1)

    svxy_state = (
        data.weights["SVXY"].abs() > 1e-12
    )

    trigger = (
        svxy_state
        & prior_beta.notna()
        & (prior_beta > beta_threshold)
    ).rename("trigger")

    target = data.weights[ASSETS].copy()

    target.loc[trigger, "SVXY"] *= svxy_scale

    returns, incremental_turnover = _base_return_from_weights(
        data,
        target,
        incremental_cost_bps=incremental_cost_bps,
    )

    return OverlayResult(
        name=(
            f"Beta>{beta_threshold:.2f}; "
            f"SVXY x{svxy_scale:.2f}"
        ),
        returns=returns,
        target_weights=target,
        trigger=trigger,
        incremental_turnover=incremental_turnover,
        rolling_beta=beta,
    )


def lagged_joint_down_svxy_overlay(
    data: TLAQData,
    svxy_scale: float = 0.50,
    incremental_cost_bps: float = 0.0,
) -> OverlayResult:
    """
    Exploratory diversification-breakdown rule.

    If VTI, TLT and GLD were ALL down on the PRIOR trading day,
    reduce today's SVXY target.

    The one-day lag is intentional and avoids same-close look-ahead.
    """
    if not 0.0 <= svxy_scale <= 1.0:
        raise ValueError("svxy_scale must be between 0 and 1.")

    joint_down = (
        (data.asset_returns["VTI"] < 0)
        & (data.asset_returns["TLT"] < 0)
        & (data.asset_returns["GLD"] < 0)
    )

    prior_joint_down = (
        joint_down.shift(1)
        .fillna(False)
        .astype(bool)
    )

    svxy_state = (
        data.weights["SVXY"].abs() > 1e-12
    )

    trigger = (
        svxy_state
        & prior_joint_down
    ).rename("trigger")

    target = data.weights[ASSETS].copy()
    target.loc[trigger, "SVXY"] *= svxy_scale

    returns, incremental_turnover = _base_return_from_weights(
        data,
        target,
        incremental_cost_bps=incremental_cost_bps,
    )

    return OverlayResult(
        name=f"Prior joint-down; SVXY x{svxy_scale:.2f}",
        returns=returns,
        target_weights=target,
        trigger=trigger,
        incremental_turnover=incremental_turnover,
    )


def baseline_result(
    data: TLAQData,
) -> OverlayResult:
    target = data.weights[ASSETS].copy()

    returns, incremental_turnover = _base_return_from_weights(
        data,
        target,
        incremental_cost_bps=0.0,
    )

    trigger = pd.Series(
        False,
        index=data.weights.index,
        name="trigger",
    )

    return OverlayResult(
        name="Original TLAQ",
        returns=returns,
        target_weights=target,
        trigger=trigger,
        incremental_turnover=incremental_turnover,
    )


def compare_results(
    results: list[OverlayResult],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    rows = []

    for result in results:
        r = result.returns

        if start is not None:
            r = r.loc[start:]

        if end is not None:
            r = r.loc[:end]

        stats = performance_summary(r)

        trigger = result.trigger
        turnover = result.incremental_turnover

        if start is not None:
            trigger = trigger.loc[start:]
            turnover = turnover.loc[start:]

        if end is not None:
            trigger = trigger.loc[:end]
            turnover = turnover.loc[:end]

        row = stats.to_dict()
        row["Trigger Days"] = int(trigger.sum())
        row["Avg Incremental Turnover"] = turnover.mean()
        row["Total Incremental Turnover"] = turnover.sum()
        row["Strategy"] = result.name

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .set_index("Strategy")
    )
