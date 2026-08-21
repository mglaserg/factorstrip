from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from factorstrip.data import (
    get_sp500_universe,
    download_returns,
)
from factorstrip.metrics import performance_summary
from factorstrip.model import HierarchicalRiskModel
from factorstrip.portfolio import backtest
from factorstrip.signals import (
    residual_momentum,
    residual_reversal,
    raw_momentum,
)


# =====================================================================
# PORTFOLIO CONSTRUCTION
# =====================================================================

def global_long_short_weights(
    signal: pd.DataFrame,
    quantile: float = 0.20,
    gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """
    Rank the ENTIRE stock universe together.

    Long:
        top quantile of signals

    Short:
        bottom quantile of signals

    Portfolio is dollar neutral:

        +0.50 gross long
        -0.50 gross short
        ----------------
         1.00 gross exposure

    Example:
        quantile = 0.20

        Long top 20%
        Short bottom 20%
    """

    if not 0 < quantile < 0.5:
        raise ValueError(
            "quantile must be greater than 0 and less than 0.5"
        )

    weights = pd.DataFrame(
        0.0,
        index=signal.index,
        columns=signal.columns,
    )

    for date, row in signal.iterrows():

        valid = row.dropna().sort_values()

        n = len(valid)

        if n < 2:
            continue

        k = int(np.floor(n * quantile))

        if k < 1:
            continue

        # Prevent overlap between long and short groups
        k = min(k, n // 2)

        shorts = valid.index[:k]
        longs = valid.index[-k:]

        long_weight = (
            gross_exposure / 2.0 / k
        )

        short_weight = (
            -gross_exposure / 2.0 / k
        )

        weights.loc[date, longs] = long_weight
        weights.loc[date, shorts] = short_weight

    return weights


# =====================================================================
# INFORMATION COEFFICIENT
# =====================================================================

def daily_information_coefficient(
    signal: pd.DataFrame,
    future_returns: pd.DataFrame,
) -> pd.Series:
    """
    Cross-sectional Spearman correlation between today's signal
    and NEXT DAY'S return.

    Positive IC:
        high-signal stocks tend to outperform.

    Negative IC:
        high-signal stocks tend to underperform.

    The return matrix is shifted backward so:

        signal[t] -> return[t + 1]
    """

    future = future_returns.shift(-1)

    signal, future = signal.align(
        future,
        join="inner",
        axis=0,
    )

    signal, future = signal.align(
        future,
        join="inner",
        axis=1,
    )

    ic_values = {}

    for date in signal.index:

        frame = pd.DataFrame(
            {
                "signal": signal.loc[date],
                "future": future.loc[date],
            }
        ).dropna()

        if len(frame) < 10:
            ic_values[date] = np.nan
            continue

        ic_values[date] = frame["signal"].corr(
            frame["future"],
            method="spearman",
        )

    return pd.Series(
        ic_values,
        name="IC",
        dtype=float,
    )


# =====================================================================
# PERFORMANCE HELPER
# =====================================================================

def strategy_stats(
    backtest_result: pd.DataFrame,
) -> pd.Series:
    """
    Build performance summary and include turnover.
    """

    stats = performance_summary(
        backtest_result["net_return"]
    )

    stats["Avg Daily Turnover"] = (
        backtest_result["turnover"].mean()
    )

    return stats


# =====================================================================
# COMMAND LINE ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "FactorStrip V1: sector risk model + "
            "raw/residual momentum research"
        )
    )

    parser.add_argument(
        "--start",
        default="2022-01-01",
        help="Backtest start date",
    )

    parser.add_argument(
        "--end",
        default=None,
        help="Backtest end date",
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=20,
        help="Signal lookback in trading days",
    )

    parser.add_argument(
        "--quantile",
        type=float,
        default=0.20,
        help="Fraction of universe long and short",
    )

    parser.add_argument(
        "--cost-bps",
        type=float,
        default=0.0,
        help="Transaction cost in basis points",
    )

    parser.add_argument(
        "--output",
        default="output",
        help="Output directory",
    )

    return parser.parse_args()


# =====================================================================
# MAIN RESEARCH PIPELINE
# =====================================================================

def main():

    args = parse_args()

    output = Path(args.output)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("FACTORSTRIP V1")
    print("=" * 60)

    # =================================================================
    # 1. LOAD CURRENT S&P 500 UNIVERSE
    # =================================================================

    universe = get_sp500_universe(
        cache_path=output / "sp500_current.csv"
    )

    sector_map = (
        universe
        .set_index("ticker")["sector"]
    )

    # =================================================================
    # 2. DOWNLOAD RETURNS
    # =================================================================

    print()
    print("Downloading price history...")

    returns = download_returns(
        universe=universe,
        start=args.start,
        end=args.end,
    )

    print(
        f"Downloaded returns for "
        f"{len(returns.columns)} stocks."
    )

    # Restrict classifications to stocks that actually downloaded
    sector_map = sector_map.reindex(
        returns.columns
    )

    # =================================================================
    # 3. BUILD SECTOR-ONLY RISK MODEL
    #
    # IMPORTANT:
    #
    # We deliberately pass sector_map as BOTH sector and industry.
    #
    # That makes:
    #
    #     industry effect = 0
    #
    # Therefore the model becomes:
    #
    #     Stock
    #       = Market
    #       + Sector
    #       + Residual
    #
    # This is intentionally simpler than our original version.
    # =================================================================

    print()
    print("Fitting sector-only risk model...")

    model = HierarchicalRiskModel(
        sector_map=sector_map,
        industry_map=sector_map,
    )

    result = model.fit(
        returns
    )

    # =================================================================
    # 4. CREATE THREE SIGNALS
    # =================================================================

    print()
    print(
        f"Building {args.lookback}-day signals..."
    )

    # -------------------------------------------------------------
    # A. RAW MOMENTUM
    #
    # Stocks that have gone up most.
    # -------------------------------------------------------------

    raw_signal = raw_momentum(
        returns,
        lookback=args.lookback,
    )

    # -------------------------------------------------------------
    # B. RESIDUAL MOMENTUM
    #
    # Stocks that have outperformed their sector most.
    # -------------------------------------------------------------

    residual_momentum_signal = residual_momentum(
        result.residuals,
        lookback=args.lookback,
    )

    # -------------------------------------------------------------
    # C. RESIDUAL REVERSAL
    #
    # Stocks that have underperformed their sector most.
    #
    # This is simply the negative of residual momentum.
    # -------------------------------------------------------------

    residual_reversal_signal = residual_reversal(
        result.residuals,
        lookback=args.lookback,
    )

    # =================================================================
    # 5. CREATE PORTFOLIOS
    #
    # IMPORTANT:
    #
    # We rank the ENTIRE universe.
    #
    # We do NOT rank separately within each sector.
    #
    # Otherwise sector residualization and within-sector ranking become
    # mathematically redundant.
    # =================================================================

    raw_weights = global_long_short_weights(
        signal=raw_signal,
        quantile=args.quantile,
        gross_exposure=1.0,
    )

    residual_momentum_weights = global_long_short_weights(
        signal=residual_momentum_signal,
        quantile=args.quantile,
        gross_exposure=1.0,
    )

    residual_reversal_weights = global_long_short_weights(
        signal=residual_reversal_signal,
        quantile=args.quantile,
        gross_exposure=1.0,
    )

    # =================================================================
    # 6. BACKTEST
    # =================================================================

    print()
    print("Running backtests...")

    raw_bt = backtest(
        raw_weights,
        returns,
        cost_bps=args.cost_bps,
    )

    residual_momentum_bt = backtest(
        residual_momentum_weights,
        # returns,
        result.residuals,
        cost_bps=args.cost_bps,
    )

    residual_reversal_bt = backtest(
        residual_reversal_weights,
        # returns,
        result.residuals,
        cost_bps=args.cost_bps,
    )

    # =================================================================
    # 7. PERFORMANCE STATISTICS
    # =================================================================

    raw_stats = strategy_stats(
        raw_bt
    )

    residual_momentum_stats = strategy_stats(
        residual_momentum_bt
    )

    residual_reversal_stats = strategy_stats(
        residual_reversal_bt
    )

    comparison = pd.DataFrame(
        {
            "Raw Momentum":
                raw_stats,

            "Residual Momentum":
                residual_momentum_stats,

            "Residual Reversal":
                residual_reversal_stats,
        }
    )

    # =================================================================
    # 8. INFORMATION COEFFICIENT
    #
    # For raw momentum:
    #
    #     signal -> next day's RAW stock return
    #
    # For residual signals:
    #
    #     signal -> next day's RESIDUAL stock return
    #
    # This asks whether the ranking itself contains predictive
    # information before worrying about portfolio construction.
    # =================================================================

    raw_ic = daily_information_coefficient(
        raw_signal,
        returns,
    )

    residual_momentum_ic = daily_information_coefficient(
        residual_momentum_signal,
        result.residuals,
    )

    residual_reversal_ic = daily_information_coefficient(
        residual_reversal_signal,
        result.residuals,
    )

    ic_summary = pd.DataFrame(
        {
            "Raw Momentum": [
                raw_ic.mean(),
                raw_ic.std(),
                (
                    raw_ic.mean()
                    / raw_ic.std()
                    * np.sqrt(252)
                    if raw_ic.std() > 0
                    else np.nan
                ),
            ],

            "Residual Momentum": [
                residual_momentum_ic.mean(),
                residual_momentum_ic.std(),
                (
                    residual_momentum_ic.mean()
                    / residual_momentum_ic.std()
                    * np.sqrt(252)
                    if residual_momentum_ic.std() > 0
                    else np.nan
                ),
            ],

            "Residual Reversal": [
                residual_reversal_ic.mean(),
                residual_reversal_ic.std(),
                (
                    residual_reversal_ic.mean()
                    / residual_reversal_ic.std()
                    * np.sqrt(252)
                    if residual_reversal_ic.std() > 0
                    else np.nan
                ),
            ],
        },
        index=[
            "Mean Daily IC",
            "IC Std",
            "IC Information Ratio",
        ],
    )

    # =================================================================
    # 9. RISK MODEL DIAGNOSTICS
    # =================================================================

    stacked_residuals = (
        result.residuals
        .stack()
    )

    diagnostics = pd.Series(
        {
            "Mean cross-sectional R2":
                result.r2.mean(),

            "Median cross-sectional R2":
                result.r2.median(),

            "Mean abs residual":
                stacked_residuals.abs().mean(),

            "Residual daily std":
                stacked_residuals.std(),
        }
    )

    # =================================================================
    # 10. SAVE OUTPUTS
    # =================================================================

    returns.to_csv(
        output / "returns.csv.gz",
        compression="gzip",
    )

    result.residuals.to_csv(
        output / "residuals.csv.gz",
        compression="gzip",
    )

    result.fitted.to_csv(
        output / "fitted_returns.csv.gz",
        compression="gzip",
    )

    result.factor_returns.to_csv(
        output / "factor_returns.csv.gz",
        compression="gzip",
    )

    # Signals
    raw_signal.to_csv(
        output / "raw_momentum_signal.csv.gz",
        compression="gzip",
    )

    residual_momentum_signal.to_csv(
        output / "residual_momentum_signal.csv.gz",
        compression="gzip",
    )

    residual_reversal_signal.to_csv(
        output / "residual_reversal_signal.csv.gz",
        compression="gzip",
    )

    # Weights
    raw_weights.to_csv(
        output / "raw_momentum_weights.csv.gz",
        compression="gzip",
    )

    residual_momentum_weights.to_csv(
        output / "residual_momentum_weights.csv.gz",
        compression="gzip",
    )

    residual_reversal_weights.to_csv(
        output / "residual_reversal_weights.csv.gz",
        compression="gzip",
    )

    # Backtests
    raw_bt.to_csv(
        output / "raw_momentum_backtest.csv"
    )

    residual_momentum_bt.to_csv(
        output / "residual_momentum_backtest.csv"
    )

    residual_reversal_bt.to_csv(
        output / "residual_reversal_backtest.csv"
    )

    # IC series
    raw_ic.to_csv(
        output / "raw_momentum_ic.csv"
    )

    residual_momentum_ic.to_csv(
        output / "residual_momentum_ic.csv"
    )

    residual_reversal_ic.to_csv(
        output / "residual_reversal_ic.csv"
    )

    # Summaries
    comparison.to_csv(
        output / "comparison.csv"
    )

    ic_summary.to_csv(
        output / "ic_summary.csv"
    )

    diagnostics.to_csv(
        output / "diagnostics.csv"
    )

    # =================================================================
    # 11. PRINT RESULTS
    # =================================================================

    print()
    print("PERFORMANCE")
    print("=" * 60)

    print(
        comparison.to_string(
            float_format=lambda x: f"{x:,.4f}"
        )
    )

    print()
    print("INFORMATION COEFFICIENT")
    print("=" * 60)

    print(
        ic_summary.to_string(
            float_format=lambda x: f"{x:,.4f}"
        )
    )

    print()
    print("RISK-MODEL DIAGNOSTICS")
    print("=" * 60)

    print(
        diagnostics.to_string(
            float_format=lambda x: f"{x:,.4f}"
        )
    )

    print()
    print(
        f"Lookback:          {args.lookback} days"
    )
    print(
        f"Long/short tails:  {args.quantile:.0%}"
    )
    print(
        f"Transaction costs: {args.cost_bps:.1f} bps"
    )

    print()
    print(
        "IMPORTANT: this prototype uses the CURRENT "
        "S&P 500 universe. Historical results therefore "
        "contain survivorship bias."
    )

    print()
    print(
        f"Results saved to: {output.resolve()}"
    )


if __name__ == "__main__":
    main()