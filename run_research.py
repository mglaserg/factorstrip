from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factorstrip.data import get_sp500_universe, download_returns
from factorstrip.metrics import performance_summary
from factorstrip.model import HierarchicalRiskModel
from factorstrip.portfolio import sector_neutral_weights, backtest
from factorstrip.signals import residual_momentum, raw_momentum


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FactorStrip V1 residual-momentum research."
    )
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--quantile", type=float, default=0.20)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--output", default="output")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # 1) Universe + classifications
    # ---------------------------------------------------------------
    universe = get_sp500_universe(
        cache_path=output / "sp500_current.csv"
    )

    sector_map = universe.set_index("ticker")["sector"]
    industry_map = universe.set_index("ticker")["industry"]

    # ---------------------------------------------------------------
    # 2) Returns
    # ---------------------------------------------------------------
    returns = download_returns(
        universe=universe,
        start=args.start,
        end=args.end,
    )

    # Restrict classifications to successfully downloaded names.
    sector_map = sector_map.reindex(returns.columns)
    industry_map = industry_map.reindex(returns.columns)

    # ---------------------------------------------------------------
    # 3) Cross-sectional risk model
    #
    # stock = market + sector + industry + residual
    # ---------------------------------------------------------------
    model = HierarchicalRiskModel(
        sector_map=sector_map,
        industry_map=industry_map,
    )

    result = model.fit(returns)

    # ---------------------------------------------------------------
    # 4) Alpha hypothesis: residual momentum
    # ---------------------------------------------------------------
    residual_signal = residual_momentum(
        result.residuals,
        lookback=args.lookback,
    )

    # Compare against ordinary raw-return momentum.
    raw_signal = raw_momentum(
        returns,
        lookback=args.lookback,
    )

    # Same portfolio construction for both signals.
    residual_weights = sector_neutral_weights(
        signal=residual_signal,
        sectors=sector_map,
        quantile=args.quantile,
        gross_exposure=1.0,
    )

    raw_weights = sector_neutral_weights(
        signal=raw_signal,
        sectors=sector_map,
        quantile=args.quantile,
        gross_exposure=1.0,
    )

    # ---------------------------------------------------------------
    # 5) Backtests
    # ---------------------------------------------------------------
    residual_bt = backtest(
        residual_weights,
        returns,
        cost_bps=args.cost_bps,
    )

    raw_bt = backtest(
        raw_weights,
        returns,
        cost_bps=args.cost_bps,
    )

    residual_stats = performance_summary(
        residual_bt["net_return"]
    )
    residual_stats["Avg Daily Turnover"] = (
        residual_bt["turnover"].mean()
    )

    raw_stats = performance_summary(
        raw_bt["net_return"]
    )
    raw_stats["Avg Daily Turnover"] = (
        raw_bt["turnover"].mean()
    )

    comparison = pd.DataFrame(
        {
            "Residual Momentum": residual_stats,
            "Raw Momentum": raw_stats,
        }
    )

    # ---------------------------------------------------------------
    # 6) Diagnostics
    # ---------------------------------------------------------------
    # How much cross-sectional variation is explained by common groups?
    diagnostics = pd.Series(
        {
            "Mean cross-sectional R2": result.r2.mean(),
            "Median cross-sectional R2": result.r2.median(),
            "Mean abs residual": result.residuals.abs().stack().mean(),
        }
    )

    # Save everything so you can inspect it rather than trust a black box.
    returns.to_csv(output / "returns.csv.gz", compression="gzip")
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
    residual_signal.to_csv(
        output / "residual_momentum_signal.csv.gz",
        compression="gzip",
    )
    residual_weights.to_csv(
        output / "residual_momentum_weights.csv.gz",
        compression="gzip",
    )
    residual_bt.to_csv(output / "residual_momentum_backtest.csv")
    raw_bt.to_csv(output / "raw_momentum_backtest.csv")
    comparison.to_csv(output / "comparison.csv")
    diagnostics.to_csv(output / "diagnostics.csv")

    print("\nFACTORSTRIP V1")
    print("=" * 60)
    print("\nPerformance comparison:")
    print(comparison.to_string(float_format=lambda x: f"{x:,.4f}"))

    print("\nRisk-model diagnostics:")
    print(diagnostics.to_string(float_format=lambda x: f"{x:,.4f}"))

    print(
        "\nIMPORTANT: this prototype uses the CURRENT S&P 500 universe. "
        "A historical performance test therefore has survivorship bias. "
        "Do not interpret it as a production-quality backtest."
    )


if __name__ == "__main__":
    main()
