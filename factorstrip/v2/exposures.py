from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .schema import COL


@dataclass(frozen=True)
class BetaConfig:
    window_days: int = 252
    min_periods: int = 126
    shrinkage_to_prior: float = 0.50
    prior_beta: float = 1.0

    def validate(self) -> None:
        if self.window_days < 2:
            raise ValueError("window_days must be >= 2")
        if not 2 <= self.min_periods <= self.window_days:
            raise ValueError("min_periods must be between 2 and window_days")
        if not 0 <= self.shrinkage_to_prior <= 1:
            raise ValueError("shrinkage_to_prior must be between 0 and 1")


def rolling_beta_arrays(asset_returns: np.ndarray, market_returns: np.ndarray, config: BetaConfig) -> tuple[np.ndarray, np.ndarray]:
    """Lag-safe rolling beta helper used by the Polars wrapper.

    beta[i] uses observations strictly before i.  This pure-numpy function is
    intentionally testable without Polars.
    """

    config.validate()
    y = np.asarray(asset_returns, dtype=float)
    x = np.asarray(market_returns, dtype=float)
    if y.shape != x.shape:
        raise ValueError("asset_returns and market_returns must have the same shape")

    raw = np.full(len(y), np.nan)
    shrunk = np.full(len(y), np.nan)
    for i in range(len(y)):
        start = max(0, i - config.window_days)
        yy = y[start:i]
        xx = x[start:i]
        valid = np.isfinite(yy) & np.isfinite(xx)
        if valid.sum() < config.min_periods:
            continue
        yy = yy[valid]
        xx = xx[valid]
        var_x = np.var(xx, ddof=1)
        if not np.isfinite(var_x) or var_x <= 0:
            continue
        beta = np.cov(yy, xx, ddof=1)[0, 1] / var_x
        raw[i] = beta
        shrunk[i] = (
            (1.0 - config.shrinkage_to_prior) * beta
            + config.shrinkage_to_prior * config.prior_beta
        )
    return raw, shrunk


def estimate_market_betas(returns: Any, market_returns: Any, config: BetaConfig | None = None):
    """Estimate lagged stock betas from long-form Polars returns.

    Parameters
    ----------
    returns:
        Polars DataFrame with date, asset_id, total_return.
    market_returns:
        Polars DataFrame with date, market_return.
    """

    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("FactorStrip V2 beta estimation requires Polars") from exc

    config = config or BetaConfig()
    config.validate()
    required = {COL.date, COL.asset_id, COL.total_return}
    missing = required - set(returns.columns)
    if missing:
        raise ValueError(f"returns missing columns: {sorted(missing)}")
    if {COL.date, "market_return"} - set(market_returns.columns):
        raise ValueError("market_returns requires date and market_return")

    joined = returns.select(COL.date, COL.asset_id, COL.total_return).join(
        market_returns.select(COL.date, "market_return"), on=COL.date, how="left"
    ).sort([COL.asset_id, COL.date])

    pieces = []
    for part in joined.partition_by(COL.asset_id, maintain_order=True):
        raw, shrunk = rolling_beta_arrays(
            part[COL.total_return].to_numpy(),
            part["market_return"].to_numpy(),
            config,
        )
        pieces.append(
            part.select(COL.date, COL.asset_id).with_columns(
                pl.Series("market_beta_raw", raw),
                pl.Series("market_beta", shrunk),
            )
        )
    if not pieces:
        return pl.DataFrame(schema={COL.date: pl.Date, COL.asset_id: pl.String, "market_beta_raw": pl.Float64, "market_beta": pl.Float64})
    return pl.concat(pieces).sort([COL.date, COL.asset_id])
