from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factorstrip.crypto_pump import PumpConfig, run_crypto_pump_study
from factorstrip.crypto_pump.binance import (
    discover_current_usdt_perpetuals,
    download_recent_panel,
)
from factorstrip.crypto_pump.study import write_study


def _read_panel(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz"} or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError("input must be .csv, .csv.gz, .parquet, or .pq")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FactorStrip crypto residual-pump study")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="Long hourly OHLCV panel")
    src.add_argument(
        "--download-binance",
        action="store_true",
        help="Download a recent exploratory Binance USD-M universe (not PIT-safe)",
    )
    p.add_argument("--symbols", help="Comma-separated Binance symbols; BTCUSDT/ETHUSDT are added if absent")
    p.add_argument("--discover", type=int, default=40, help="Current high-volume contracts when downloading")
    p.add_argument("--days", type=int, default=120)
    p.add_argument("--raw-threshold", type=float, default=0.50)
    p.add_argument("--residual-threshold", type=float, default=0.30)
    p.add_argument("--min-qv-24h", type=float, default=5_000_000.0)
    p.add_argument("--cost-bps", type=float, default=20.0)
    p.add_argument("--factor-window-hours", type=int, default=24 * 14)
    p.add_argument("--cooldown-hours", type=int, default=24)
    p.add_argument("--output", type=Path, default=Path("output/crypto_pump"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.input:
        panel = _read_panel(args.input)
    else:
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
            for anchor in ("BTCUSDT", "ETHUSDT"):
                if anchor not in symbols:
                    symbols.insert(0, anchor)
        else:
            symbols = discover_current_usdt_perpetuals(args.discover)
            print("WARNING: --discover uses today's listed/liquid contracts and is exploratory only; it is not a point-in-time historical universe.")
        print(f"Downloading {len(symbols)} symbols for {args.days} days...")
        panel = download_recent_panel(symbols, args.days)
        args.output.mkdir(parents=True, exist_ok=True)
        panel.to_csv(args.output / "input_panel.csv.gz", index=False)

    cfg = PumpConfig(
        factor_window_hours=args.factor_window_hours,
        raw_pump_threshold=args.raw_threshold,
        residual_pump_threshold=args.residual_threshold,
        min_quote_volume_24h=args.min_qv_24h,
        round_trip_cost_bps=args.cost_bps,
        cooldown_hours=args.cooldown_hours,
    )
    result = run_crypto_pump_study(panel, cfg)
    out = write_study(result, args.output)
    print(result["summary"].to_string(index=False))
    print(f"\nWrote study to {out}")


if __name__ == "__main__":
    main()
