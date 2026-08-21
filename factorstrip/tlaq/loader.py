from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ASSETS = ["VTI", "TLT", "GLD", "SVXY", "VIXY"]
PORTFOLIO_ASSETS = ["VTI", "TLT", "GLD", "VOL"]
REQUIRED_COLUMNS = {
    "ticker",
    "date",
    "close",
    "shares",
    "exposure",
    "sharetrades",
    "tradevalue",
    "commission",
    "interest",
    "short_borrow_cost",
    "margin_call",
}


@dataclass
class TLAQData:
    raw: pd.DataFrame
    prices: pd.DataFrame
    shares: pd.DataFrame
    exposures: pd.DataFrame
    weights: pd.DataFrame
    held_weights: pd.DataFrame
    nav: pd.Series
    asset_returns: pd.DataFrame
    active_returns: pd.DataFrame
    pnl_by_asset: pd.DataFrame
    return_contributions: pd.DataFrame
    strategy_returns: pd.Series
    commissions: pd.Series
    interest: pd.Series
    short_borrow_cost: pd.Series
    vol_state: pd.Series
    held_vol_state: pd.Series
    gross_exposure: pd.Series


def _pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return (
        df.pivot(index="date", columns="ticker", values=value)
        .sort_index()
        .rename_axis(columns=None)
    )


def _vol_state_from_shares(shares: pd.DataFrame) -> pd.Series:
    svxy = shares.get("SVXY", pd.Series(0.0, index=shares.index)).fillna(0.0)
    vixy = shares.get("VIXY", pd.Series(0.0, index=shares.index)).fillna(0.0)

    both = (svxy != 0) & (vixy != 0)
    if both.any():
        dates = [str(x.date()) for x in shares.index[both][:5]]
        raise ValueError(
            "TLAQ loader found dates with both SVXY and VIXY held. "
            f"First examples: {dates}"
        )

    state = pd.Series("NONE", index=shares.index, dtype="string", name="vol_state")
    state.loc[svxy != 0] = "SVXY"
    state.loc[vixy != 0] = "VIXY"
    return state


def load_trades_table(path: str | Path) -> TLAQData:
    """
    Load a TLAQ trades_table and reconstruct daily portfolio accounting.

    Important timing convention
    ---------------------------
    Rows describe end-of-day positions. The position recorded at t-1 earns
    the close-to-close price move from t-1 to t. Therefore realized return
    attribution uses lagged shares/weights.

    This keeps TLAQ's trading logic untouched; FactorStrip only diagnoses it.
    """
    path = Path(path)
    raw = pd.read_csv(path, parse_dates=["date"])

    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"trades_table is missing columns: {sorted(missing)}")

    duplicate = raw.duplicated(["date", "ticker"])
    if duplicate.any():
        examples = raw.loc[duplicate, ["date", "ticker"]].head().to_dict("records")
        raise ValueError(f"Duplicate date/ticker rows found: {examples}")

    prices = _pivot(raw, "close").reindex(columns=ASSETS)
    shares = _pivot(raw, "shares").reindex(columns=["Cash", *ASSETS]).fillna(0.0)
    exposures = _pivot(raw, "exposure").reindex(columns=["Cash", *ASSETS]).fillna(0.0)

    # Zero prices in the backtest indicate pre-inception/unavailable history.
    clean_prices = prices.where(prices > 0)
    asset_returns = clean_prices.pct_change(fill_method=None)

    nav = exposures.sum(axis=1).rename("nav")
    if (nav <= 0).any():
        bad = nav[nav <= 0].head()
        raise ValueError(f"Non-positive NAV encountered:\n{bad}")

    weights = exposures.div(nav, axis=0)
    held_weights = weights.shift(1)

    commissions = raw.groupby("date")["commission"].sum().reindex(nav.index).fillna(0.0)
    interest = raw.groupby("date")["interest"].sum().reindex(nav.index).fillna(0.0)
    short_borrow_cost = (
        raw.groupby("date")["short_borrow_cost"].sum().reindex(nav.index).fillna(0.0)
    )

    # Exact dollar P&L from positions held at the prior close.
    price_change = clean_prices - clean_prices.shift(1)
    prior_shares = shares[ASSETS].shift(1)
    pnl_by_asset = (prior_shares * price_change).fillna(0.0)

    pnl_by_asset["FEES"] = -commissions + interest - short_borrow_cost
    pnl_by_asset["TOTAL"] = pnl_by_asset.sum(axis=1)

    prior_nav = nav.shift(1)
    return_contributions = pnl_by_asset.drop(columns="TOTAL").div(prior_nav, axis=0)
    return_contributions["TOTAL"] = return_contributions.sum(axis=1)

    strategy_returns = nav.pct_change(fill_method=None).rename("TLAQ")

    # Validate accounting. Tiny floating-point differences are expected.
    accounting_error = (nav.diff() - pnl_by_asset["TOTAL"]).dropna()
    if not accounting_error.empty and accounting_error.abs().max() > 1e-5:
        raise ValueError(
            "TLAQ accounting did not reconcile. Maximum dollar error: "
            f"{accounting_error.abs().max():.6f}"
        )

    vol_state = _vol_state_from_shares(shares)
    held_vol_state = vol_state.shift(1).fillna("NONE").astype("string")
    held_vol_state.name = "held_vol_state"

    # Active volatility-sleeve return: return of the instrument actually held
    # over that close-to-close interval. NONE is NaN rather than zero because
    # zero would falsely imply an observed zero-return volatility asset.
    active_vol = pd.Series(np.nan, index=nav.index, name="VOL", dtype=float)
    active_vol.loc[held_vol_state == "SVXY"] = asset_returns.loc[
        held_vol_state == "SVXY", "SVXY"
    ]
    active_vol.loc[held_vol_state == "VIXY"] = asset_returns.loc[
        held_vol_state == "VIXY", "VIXY"
    ]

    active_returns = pd.concat(
        [asset_returns[["VTI", "TLT", "GLD"]], active_vol], axis=1
    )

    gross_exposure = (
        exposures[ASSETS].abs().sum(axis=1) / nav
    ).rename("gross_exposure")

    return TLAQData(
        raw=raw.sort_values(["date", "ticker"]).reset_index(drop=True),
        prices=clean_prices,
        shares=shares,
        exposures=exposures,
        weights=weights,
        held_weights=held_weights,
        nav=nav,
        asset_returns=asset_returns,
        active_returns=active_returns,
        pnl_by_asset=pnl_by_asset,
        return_contributions=return_contributions,
        strategy_returns=strategy_returns,
        commissions=commissions,
        interest=interest,
        short_borrow_cost=short_borrow_cost,
        vol_state=vol_state,
        held_vol_state=held_vol_state,
        gross_exposure=gross_exposure,
    )
