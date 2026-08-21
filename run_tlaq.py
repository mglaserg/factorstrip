from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factorstrip.tlaq import load_trades_table, build_tlaq_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="FactorStrip diagnostics for the TLAQ strategy."
    )
    parser.add_argument(
        "--trades-table",
        default="trades_table.csv",
        help="Path to TLAQ trades_table.csv",
    )
    parser.add_argument(
        "--start",
        default="2012-01-05",
        help=(
            "Analysis start date. Default is 2012-01-05, the first SVXY "
            "holding in the supplied TLAQ history. Use 'all' for full history."
        ),
    )
    parser.add_argument("--beta-window", type=int, default=60)
    parser.add_argument("--risk-window", type=int, default=60)
    parser.add_argument("--output", default="output/tlaq")
    return parser.parse_args()


def slice_data(data, start: str | None):
    """Slice every time-indexed member of the TLAQData dataclass."""
    if start is None or str(start).lower() == "all":
        return data

    cutoff = pd.Timestamp(start)
    for field_name in data.__dataclass_fields__:
        value = getattr(data, field_name)
        if isinstance(value, (pd.Series, pd.DataFrame)) and isinstance(value.index, pd.DatetimeIndex):
            setattr(data, field_name, value.loc[value.index >= cutoff])
    return data


def _print_table(title: str, obj, rows: int | None = None):
    print()
    print(title)
    print("=" * 72)
    if rows is not None and hasattr(obj, "head"):
        obj = obj.head(rows)
    print(obj.to_string(float_format=lambda x: f"{x:,.4f}"))


def main():
    args = parse_args()

    print("TLAQ FACTORSTRIP")
    print("=" * 72)
    print(f"Loading: {args.trades_table}")

    data = load_trades_table(args.trades_table)
    data = slice_data(data, args.start)

    if data.nav.empty:
        raise RuntimeError("No observations remain after applying --start.")

    print(
        f"Analysis period: {data.nav.index.min().date()} -> "
        f"{data.nav.index.max().date()} ({len(data.nav):,} dates)"
    )

    results = build_tlaq_report(
        data=data,
        output_dir=Path(args.output),
        beta_window=args.beta_window,
        risk_window=args.risk_window,
    )

    _print_table("PERFORMANCE", results["performance"])
    _print_table("TLAQ VTI STRESS BETA", results["stress"])

    print()
    print("CURRENT TLAQ")
    print("=" * 72)
    print(results["latest_summary"].to_string())

    _print_table("CURRENT WEIGHTS", results["latest_weights"])
    _print_table("VOLATILITY STATE STATISTICS", results["state_stats"])

    state_stress = results["vol_state_stress"].reset_index()
    state_beta = state_stress.pivot(index="state", columns="regime", values="tlaq_beta_to_VTI")
    state_order = ["ALL", "VTI_DOWN", "VTI_WORST_20", "VTI_WORST_10", "VTI_WORST_05"]
    state_beta = state_beta.reindex(columns=[c for c in state_order if c in state_beta])
    _print_table("TLAQ BETA TO VTI BY VOL STATE + STRESS", state_beta)

    # Convenient regime pivots for terminal output.
    vti_betas = results["vti_betas"].reset_index()
    beta_pivot = vti_betas.pivot(index="asset", columns="regime", values="beta_to_VTI")
    order = ["ALL", "VTI_DOWN", "VTI_WORST_20", "VTI_WORST_10", "VTI_WORST_05"]
    beta_pivot = beta_pivot.reindex(columns=[c for c in order if c in beta_pivot])
    _print_table("ASSET BETA TO VTI BY REGIME", beta_pivot)

    unique = results["unique_risk"].reset_index()
    unique_pivot = unique.pivot(index="asset", columns="regime", values="unique_variance_proxy")
    unique_pivot = unique_pivot.reindex(columns=[c for c in order if c in unique_pivot])
    _print_table("UNIQUE VARIANCE PROXY (1 - R2)", unique_pivot)

    rolling_risk = results["rolling_risk"]
    if not rolling_risk.empty:
        latest_risk = rolling_risk.iloc[-1]
        _print_table("LATEST ROLLING RISK CONTRIBUTIONS", latest_risk)

    _print_table("WORST 10 TLAQ DAYS", results["worst_days"], rows=10)

    print()
    print(f"Detailed outputs saved to: {Path(args.output).resolve()}")
    print("Open REPORT.md there for a guide to the output files.")


if __name__ == "__main__":
    main()
