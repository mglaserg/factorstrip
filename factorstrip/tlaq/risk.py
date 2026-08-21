from __future__ import annotations

import numpy as np
import pandas as pd


def _portfolio_weights_for_active_sleeves(weights: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
    """Map SVXY/VIXY weights into one active VOL sleeve."""
    out = pd.DataFrame(index=weights.index, columns=["VTI", "TLT", "GLD", "VOL"], dtype=float)
    out[["VTI", "TLT", "GLD"]] = weights[["VTI", "TLT", "GLD"]]
    out["VOL"] = 0.0
    out.loc[state == "SVXY", "VOL"] = weights.loc[state == "SVXY", "SVXY"]
    out.loc[state == "VIXY", "VOL"] = weights.loc[state == "VIXY", "VIXY"]
    return out


def risk_contribution_from_cov(
    weights: pd.Series,
    covariance: pd.DataFrame,
    annualization: int = 252,
) -> pd.Series:
    """Variance contribution shares plus annualized portfolio volatility."""
    names = [c for c in covariance.columns if c in weights.index and pd.notna(weights[c])]
    if not names:
        return pd.Series(dtype=float)

    cov = covariance.loc[names, names].to_numpy(dtype=float)
    w = weights.loc[names].to_numpy(dtype=float)

    if np.isnan(cov).any():
        return pd.Series(dtype=float)

    portfolio_var = float(w @ cov @ w)
    if portfolio_var <= 0:
        return pd.Series(dtype=float)

    marginal = cov @ w
    component_var = w * marginal
    share = component_var / portfolio_var

    result = pd.Series(share, index=names, name="risk_contribution")
    result.loc["PORTFOLIO_VOL"] = np.sqrt(portfolio_var * annualization)
    result.loc["PORTFOLIO_VAR_DAILY"] = portfolio_var
    return result


def rolling_portfolio_risk(
    active_returns: pd.DataFrame,
    weights: pd.DataFrame,
    vol_state: pd.Series,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Rolling covariance risk contribution using the portfolio's end-of-day
    weights as the risk forecast for the next interval.
    """
    if min_periods is None:
        min_periods = max(20, window // 2)

    active_weights = _portfolio_weights_for_active_sleeves(weights, vol_state)
    rows = []

    for pos, date in enumerate(active_returns.index):
        start = max(0, pos - window + 1)
        hist = active_returns.iloc[start : pos + 1]

        # Only include sleeves with a non-zero current exposure. This avoids
        # needing VOL history on dates when no volatility sleeve is held.
        current_w = active_weights.loc[date]
        names = [c for c in active_returns.columns if abs(current_w.get(c, 0.0)) > 1e-12]
        if not names:
            continue

        hist = hist[names].dropna()
        if len(hist) < min_periods:
            continue

        cov = hist.cov()
        rc = risk_contribution_from_cov(current_w[names], cov)
        if rc.empty:
            continue

        row = {"date": date, "n_obs": len(hist)}
        for name in names:
            row[f"risk_contribution::{name}"] = rc.get(name, np.nan)
        row["portfolio_vol"] = rc.get("PORTFOLIO_VOL", np.nan)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").sort_index()


def latest_risk_contributions(
    active_returns: pd.DataFrame,
    weights: pd.DataFrame,
    vol_state: pd.Series,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.Series:
    rolling = rolling_portfolio_risk(
        active_returns=active_returns,
        weights=weights,
        vol_state=vol_state,
        window=window,
        min_periods=min_periods,
    )
    if rolling.empty:
        return pd.Series(dtype=float)
    return rolling.iloc[-1].rename(rolling.index[-1])
