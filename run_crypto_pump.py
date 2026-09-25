from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from factorstrip.crypto_pump import PumpConfig, run_crypto_pump_study
from factorstrip.crypto_pump.binance import (
    BinanceLiveAccessError,
    discover_current_usdt_perpetuals,
    download_archive_panel,
    starter_usdt_perpetuals,
)
from factorstrip.crypto_pump.study import write_study


def _read_panel(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz"} or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError("input must be .csv, .csv.gz, .parquet, or .pq")


def _read_symbols_file(path: Path) -> list[str]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        values.extend(x.strip().upper() for x in line.replace(",", " ").split() if x.strip())
    return values


def _ensure_anchors(symbols: list[str]) -> list[str]:
    symbols = list(dict.fromkeys(s.upper() for s in symbols))
    for anchor in reversed(("BTCUSDT", "ETHUSDT")):
        if anchor not in symbols:
            symbols.insert(0, anchor)
    return symbols


def _parse_end(value: str | None):
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FactorStrip crypto residual-pump study")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="Long hourly OHLCV panel")
    src.add_argument(
        "--download-binance",
        action="store_true",
        help="Download Binance USD-M 1h klines from the public historical archive",
    )
    p.add_argument("--symbols", help="Comma-separated Binance symbols; BTCUSDT/ETHUSDT are added if absent")
    p.add_argument("--symbols-file", type=Path, help="Text file containing symbols separated by spaces, commas, or lines")
    p.add_argument(
        "--discover",
        type=int,
        default=40,
        help="Use N symbols from FactorStrip's deterministic archive research universe (no live REST call)",
    )
    p.add_argument(
        "--live-discover",
        action="store_true",
        help="Explicitly try Binance live Futures REST for current high-volume symbols; may return HTTP 451 by region",
    )
    p.add_argument("--days", type=int, default=120)
    p.add_argument(
        "--archive-end",
        help="Exclusive UTC end date/time. Default: start of current UTC month (completed monthly archives only)",
    )
    p.add_argument("--cache-dir", type=Path, default=Path("data/binance_archive_cache"))
    p.add_argument("--workers", type=int, default=8)
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
        if args.symbols and args.symbols_file:
            raise SystemExit("Use either --symbols or --symbols-file, not both.")

        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        elif args.symbols_file:
            symbols = _read_symbols_file(args.symbols_file)
        elif args.live_discover:
            try:
                symbols = discover_current_usdt_perpetuals(args.discover)
            except BinanceLiveAccessError as exc:
                print(f"WARNING: {exc}")
                print("Falling back to FactorStrip's deterministic archive research universe.")
                symbols = starter_usdt_perpetuals(args.discover)
        else:
            symbols = starter_usdt_perpetuals(args.discover)
            print(
                "NOTE: --discover now means a deterministic archive research universe; "
                "it does NOT call Binance live Futures REST and is not PIT-safe membership."
            )

        symbols = _ensure_anchors(symbols)
        archive_end = _parse_end(args.archive_end)
        print(
            f"Downloading {len(symbols)} symbols for {args.days} days from "
            "Binance public USD-M monthly archives..."
        )
        panel, missing = download_archive_panel(
            symbols,
            args.days,
            end=archive_end,
            cache_dir=args.cache_dir,
            workers=args.workers,
        )
        if missing:
            preview = ", ".join(missing[:15])
            more = "..." if len(missing) > 15 else ""
            print(f"Skipped {len(missing)} symbols with no archive data in range: {preview}{more}")
        if panel.empty:
            raise SystemExit(
                "No Binance archive rows were downloaded. Try explicit --symbols, a different "
                "--archive-end, or use --input with an existing hourly panel."
            )
        present = set(panel["symbol"].unique())
        required = {"BTCUSDT", "ETHUSDT"}
        if not required.issubset(present):
            raise SystemExit(
                f"Archive panel is missing required factors: {sorted(required - present)}"
            )
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
