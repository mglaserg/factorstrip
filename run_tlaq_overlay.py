from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factorstrip.tlaq.loader import load_trades_table
from factorstrip.tlaq.overlay import (
    baseline_result,
    beta_conditioned_svxy_overlay,
    lagged_joint_down_svxy_overlay,
    compare_results,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Test look-ahead-safe SVXY risk overlays on TLAQ."
    )

    p.add_argument(
        "--trades-table",
        default="trades_table.csv",
    )

    p.add_argument(
        "--start",
        default="2012-01-05",
    )

    p.add_argument(
        "--end",
        default=None,
    )

    p.add_argument(
        "--beta-window",
        type=int,
        default=60,
    )

    p.add_argument(
        "--cost-bps",
        type=float,
        default=0.0,
        help=(
            "Additional transaction cost applied only to incremental "
            "SVXY turnover created by the overlay."
        ),
    )

    p.add_argument(
        "--output",
        default="tlaq_overlay_output",
    )

    return p.parse_args()


def main():
    args = parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    data = load_trades_table(args.trades_table)

    results = [
        baseline_result(data),

        # Fixed beta thresholds.
        beta_conditioned_svxy_overlay(
            data,
            beta_threshold=1.00,
            svxy_scale=0.75,
            beta_window=args.beta_window,
            incremental_cost_bps=args.cost_bps,
        ),
        beta_conditioned_svxy_overlay(
            data,
            beta_threshold=1.00,
            svxy_scale=0.50,
            beta_window=args.beta_window,
            incremental_cost_bps=args.cost_bps,
        ),
        beta_conditioned_svxy_overlay(
            data,
            beta_threshold=1.25,
            svxy_scale=0.75,
            beta_window=args.beta_window,
            incremental_cost_bps=args.cost_bps,
        ),
        beta_conditioned_svxy_overlay(
            data,
            beta_threshold=1.25,
            svxy_scale=0.50,
            beta_window=args.beta_window,
            incremental_cost_bps=args.cost_bps,
        ),
        beta_conditioned_svxy_overlay(
            data,
            beta_threshold=1.50,
            svxy_scale=0.75,
            beta_window=args.beta_window,
            incremental_cost_bps=args.cost_bps,
        ),
        beta_conditioned_svxy_overlay(
            data,
            beta_threshold=1.50,
            svxy_scale=0.50,
            beta_window=args.beta_window,
            incremental_cost_bps=args.cost_bps,
        ),
        beta_conditioned_svxy_overlay(
            data,
            beta_threshold=1.50,
            svxy_scale=0.00,
            beta_window=args.beta_window,
            incremental_cost_bps=args.cost_bps,
        ),

        # Exploratory diversification-breakdown rule.
        lagged_joint_down_svxy_overlay(
            data,
            svxy_scale=0.75,
            incremental_cost_bps=args.cost_bps,
        ),
        lagged_joint_down_svxy_overlay(
            data,
            svxy_scale=0.50,
            incremental_cost_bps=args.cost_bps,
        ),
        lagged_joint_down_svxy_overlay(
            data,
            svxy_scale=0.00,
            incremental_cost_bps=args.cost_bps,
        ),
    ]

    comparison = compare_results(
        results,
        start=args.start,
        end=args.end,
    )

    comparison.to_csv(
        output / "svxy_overlay_comparison.csv"
    )

    # Save return streams so every scenario can be inspected.
    returns = pd.concat(
        {
            result.name: result.returns
            for result in results
        },
        axis=1,
    )

    if args.start is not None:
        returns = returns.loc[args.start:]

    if args.end is not None:
        returns = returns.loc[:args.end]

    returns.to_csv(
        output / "svxy_overlay_returns.csv"
    )

    # Save the rolling beta from one beta scenario.
    beta_result = next(
        result
        for result in results
        if result.rolling_beta is not None
    )

    beta_result.rolling_beta.to_csv(
        output / "rolling_tlaq_vti_beta.csv"
    )

    print()
    print("TLAQ SVXY OVERLAY TEST")
    print("=" * 90)
    print()
    print(
        comparison.to_string(
            float_format=lambda x: f"{x:,.4f}"
        )
    )

    print()
    print(
        "IMPORTANT: all overlay triggers use information from the "
        "prior completed trading day. VIXY is never reduced."
    )
    print(
        "The optional --cost-bps charge applies only to incremental "
        "SVXY turnover created by the overlay."
    )
    print()
    print(f"Saved results to: {output.resolve()}")


if __name__ == "__main__":
    main()
