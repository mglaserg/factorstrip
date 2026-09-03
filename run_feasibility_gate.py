from __future__ import annotations

import argparse
import json
from pathlib import Path

from factorstrip.feasibility import GateConfig, run_blind_rho_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FactorStrip V2 blinded feasibility gate. Outputs only rho, turnover, "
            "coverage, and prospective power/design quantities."
        )
    )
    parser.add_argument("--start", default="2000-01-01", help="Daily price-history start date")
    parser.add_argument("--end", default=None, help="Daily price-history end date")
    parser.add_argument("--market", default="SPY", help="Market benchmark ticker")
    parser.add_argument("--quantile", type=float, default=0.20)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--beta-window", type=int, default=252)
    parser.add_argument("--beta-min-periods", type=int, default=126)
    parser.add_argument(
        "--beta-shrinkage",
        type=float,
        default=0.50,
        help="Weight on beta prior of 1.0; 0=no shrinkage, 1=all prior",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-block-months", type=int, default=6)
    parser.add_argument(
        "--available-clean-years",
        type=float,
        default=None,
        help=(
            "Optional clean PIT history available from a contemplated data source. "
            "If supplied, the gate emits GREENLIGHT or UNRESOLVABLE."
        ),
    )
    parser.add_argument("--output", default="feasibility_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print("FACTORSTRIP V2 — BLINDED FEASIBILITY GATE")
    print("=" * 60)
    print("This path intentionally does NOT compute alpha, Sharpe, CAGR, IC, or cumulative P&L.")

    # Lazy import keeps --help and unit-test discovery independent of network/data packages.
    from factorstrip.data import download_adjusted_close, download_returns, get_sp500_universe

    universe = get_sp500_universe(cache_path=output / "sp500_current.csv")
    stock_returns = download_returns(universe=universe, start=args.start, end=args.end)

    market_close = download_adjusted_close([args.market], start=args.start, end=args.end)
    market_returns = market_close.iloc[:, 0].pct_change(fill_method=None).rename(args.market)

    common = stock_returns.index.intersection(market_returns.dropna().index)
    stock_returns = stock_returns.loc[common]
    market_returns = market_returns.loc[common]

    config = GateConfig(
        quantile=args.quantile,
        cost_bps=args.cost_bps,
        beta_window_days=args.beta_window,
        beta_min_periods=args.beta_min_periods,
        beta_shrinkage=args.beta_shrinkage,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_block_months=args.bootstrap_block_months,
    )

    summary = run_blind_rho_gate(
        stock_returns,
        market_returns,
        config=config,
        available_clean_years=args.available_clean_years,
    )

    summary_path = output / "gate_summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n")

    print()
    print(f"rho                     : {summary.rho:.4f}")
    print(f"rho 95% block CI        : [{summary.rho_ci_lower:.4f}, {summary.rho_ci_upper:.4f}]")
    print(f"rho used for power      : {summary.rho_for_power:.4f}  (conservative CI value)")
    print(f"paired monthly obs      : {summary.paired_months}")
    print(f"required clean years    : {summary.required_years:.1f}")
    print(f"required clean months   : {summary.required_months}")
    print(f"raw monthly turnover    : {summary.avg_monthly_turnover_raw:.3f}")
    print(f"resid monthly turnover  : {summary.avg_monthly_turnover_residual:.3f}")
    print(f"gate status             : {summary.gate_status}")
    print()
    print(f"Saved blinded gate summary to {summary_path}")


if __name__ == "__main__":
    main()
