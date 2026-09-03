from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BlitzReferenceConfig:
    estimation_months: int = 36
    formation_months: int = 11
    skip_months: int = 1
    min_estimation_months: int = 24

    def validate(self) -> None:
        if self.estimation_months < 12:
            raise ValueError("estimation_months must be >= 12")
        if self.formation_months < 1:
            raise ValueError("formation_months must be >= 1")
        if self.skip_months < 1:
            raise ValueError("skip_months must be >= 1")
        if self.formation_months + self.skip_months > self.estimation_months:
            raise ValueError("formation + skip must fit inside estimation window")
        if not 4 <= self.min_estimation_months <= self.estimation_months:
            raise ValueError("min_estimation_months must be within estimation window")


def blitz_residual_momentum_reference(
    monthly_stock_returns: pd.DataFrame,
    monthly_ff3_returns: pd.DataFrame,
    config: BlitzReferenceConfig | None = None,
) -> pd.DataFrame:
    """Methodology reference for Blitz/Huij/Martens-style residual momentum.

    This is deliberately called a *reference*, not a paper replication.  For
    each signal month t, a standard OLS with intercept is estimated over the
    trailing estimation window ending at t-1.  The residual-momentum numerator
    uses the formation subset t-12..t-2 under the default 12-1 convention, and
    is standardized by the residual standard deviation from the longer
    estimation window.

    Returns a date x ticker signal matrix.  No portfolio performance is
    calculated here.
    """

    cfg = config or BlitzReferenceConfig()
    cfg.validate()
    stocks = monthly_stock_returns.sort_index().astype(float)
    factors = monthly_ff3_returns.sort_index().astype(float)
    if factors.shape[1] != 3:
        raise ValueError("monthly_ff3_returns must contain exactly three factor columns")
    stocks, factors = stocks.align(factors, join="inner", axis=0)
    out = pd.DataFrame(np.nan, index=stocks.index, columns=stocks.columns, dtype=float)

    for t in range(cfg.estimation_months, len(stocks)):
        est_start = t - cfg.estimation_months
        est_end = t  # exclusive: no signal-month return enters the regression
        form_end = t - cfg.skip_months
        form_start = form_end - cfg.formation_months
        if form_start < est_start:
            continue

        f_window = factors.iloc[est_start:est_end]
        x_full = np.column_stack([np.ones(len(f_window)), f_window.to_numpy(dtype=float)])
        form_slice = slice(form_start - est_start, form_end - est_start)

        for ticker in stocks.columns:
            y_full = stocks[ticker].iloc[est_start:est_end].to_numpy(dtype=float)
            valid = np.isfinite(y_full) & np.all(np.isfinite(x_full), axis=1)
            if valid.sum() < cfg.min_estimation_months:
                continue
            x = x_full[valid]
            y = y_full[valid]
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            resid_full = np.full(len(y_full), np.nan)
            resid_full[valid] = y - x @ coef
            resid_std = float(np.nanstd(resid_full, ddof=1))
            formation = resid_full[form_slice]
            if not np.isfinite(resid_std) or resid_std <= 0 or np.isfinite(formation).sum() < cfg.formation_months:
                continue
            out.iloc[t, out.columns.get_loc(ticker)] = float(np.nansum(formation) / resid_std)

    return out
