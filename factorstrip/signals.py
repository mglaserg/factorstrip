from __future__ import annotations

import pandas as pd


def residual_momentum(
    residuals: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Sum recent idiosyncratic returns.

    The signal is formed at the close of date t. The backtester shifts
    positions one day, so the signal is not traded until t+1.
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    return residuals.rolling(
        lookback,
        min_periods=lookback,
    ).sum()


def residual_reversal(
    residuals: pd.DataFrame,
    lookback: int = 5,
) -> pd.DataFrame:
    """
    Short-horizon reversal: opposite of cumulative residual return.
    """
    return -residual_momentum(
        residuals=residuals,
        lookback=lookback,
    )


def raw_momentum(
    returns: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Benchmark signal: ordinary raw-return momentum.
    """
    return returns.rolling(
        lookback,
        min_periods=lookback,
    ).sum()
