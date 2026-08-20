from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def performance_summary(
    returns: pd.Series,
    annualization: int = TRADING_DAYS,
) -> pd.Series:
    """
    Basic strategy metrics. Risk-free rate is assumed to be zero.
    """
    r = returns.dropna()

    if r.empty:
        return pd.Series(dtype=float)

    equity = (1.0 + r).cumprod()
    years = len(r) / annualization

    if years > 0 and equity.iloc[-1] > 0:
        cagr = equity.iloc[-1] ** (1.0 / years) - 1.0
    else:
        cagr = np.nan

    vol = r.std(ddof=1) * np.sqrt(annualization)

    sharpe = (
        r.mean() / r.std(ddof=1) * np.sqrt(annualization)
        if r.std(ddof=1) > 0
        else np.nan
    )

    drawdown = equity / equity.cummax() - 1.0

    return pd.Series(
        {
            "CAGR": cagr,
            "Annualized Vol": vol,
            "Sharpe": sharpe,
            "Max Drawdown": drawdown.min(),
            "Avg Daily Turnover": np.nan,
        }
    )
