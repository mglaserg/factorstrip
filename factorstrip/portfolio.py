from __future__ import annotations

import numpy as np
import pandas as pd


def sector_neutral_weights(
    signal: pd.DataFrame,
    sectors: pd.Series,
    quantile: float = 0.20,
    gross_exposure: float = 1.0,
    min_names_per_side: int = 1,
) -> pd.DataFrame:
    """
    Build a dollar-neutral AND sector-dollar-neutral long/short portfolio.

    Within every sector, long the strongest signals and short the weakest.
    Each active sector receives equal gross exposure before the final
    portfolio-wide normalization.

    This is deliberately simple. V1 is about testing the residual signal,
    not optimizing the portfolio.
    """
    if not 0 < quantile < 0.5:
        raise ValueError("quantile must be between 0 and 0.5.")

    weights = pd.DataFrame(
        0.0,
        index=signal.index,
        columns=signal.columns,
    )

    sectors = sectors.reindex(signal.columns)

    for date, row in signal.iterrows():
        frame = pd.DataFrame(
            {
                "signal": row,
                "sector": sectors,
            }
        ).dropna()

        active_sector_weights = []

        for sector, group in frame.groupby("sector", observed=True):
            group = group.sort_values("signal")

            n = len(group)
            k = max(
                min_names_per_side,
                int(np.floor(n * quantile)),
            )

            # Must have distinct long and short groups.
            k = min(k, n // 2)
            if k < 1:
                continue

            shorts = group.index[:k]
            longs = group.index[-k:]

            sector_w = pd.Series(0.0, index=group.index)
            sector_w.loc[longs] = 1.0 / k
            sector_w.loc[shorts] = -1.0 / k

            active_sector_weights.append(sector_w)

        if not active_sector_weights:
            continue

        day = pd.concat(active_sector_weights)

        # The construction above is net-zero inside every sector.
        # Normalize whole-portfolio gross exposure.
        gross = day.abs().sum()
        if gross > 0:
            day *= gross_exposure / gross

        weights.loc[date, day.index] = day

    return weights


def backtest(
    target_weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Simple close-to-close backtest.

    Signal/target weights formed on date t are held on t+1.
    This one-day shift is important: it prevents look-ahead trading
    on the same close used to construct the signal.

    Transaction cost:
        cost_bps per 1.0 of one-way turnover.
    """
    target_weights, returns = target_weights.align(
        returns,
        join="inner",
        axis=0,
    )
    target_weights, returns = target_weights.align(
        returns,
        join="inner",
        axis=1,
    )

    positions = target_weights.shift(1).fillna(0.0)

    gross_return = (positions * returns.fillna(0.0)).sum(axis=1)

    turnover = (
        positions.diff()
        .abs()
        .sum(axis=1)
        .fillna(positions.abs().sum(axis=1))
    )

    costs = turnover * (cost_bps / 10_000.0)
    net_return = gross_return - costs

    return pd.DataFrame(
        {
            "gross_return": gross_return,
            "turnover": turnover,
            "cost": costs,
            "net_return": net_return,
        }
    )
