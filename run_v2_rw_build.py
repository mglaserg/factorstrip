from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from factorstrip.v2.momentum import MOM_12_1, MOM_6_1, build_monthly_raw_and_residual_signals
from factorstrip.v2.rw_r1000 import (
    RwR1000Config,
    build_rw_r1000_panel,
    read_rw_frame,
    write_rw_canonical,
)
from factorstrip.v2.toraniko_engine import ToranikoMarketSizeValueEngine


FORBIDDEN_METRICS = {"alpha", "sharpe", "cagr", "ic", "drawdown", "pnl"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build blinded FactorStrip V2 inputs from RW R1000 data")
    p.add_argument("--ohlc", required=True, help="R1000 OHLC Parquet or Feather V1")
    p.add_argument("--fundamentals", required=True, help="R1000 daily fundamentals Parquet or Feather V1")
    p.add_argument("--out", default="v2_rw_output")
    p.add_argument("--coverage-floor", type=float, default=0.90)
    p.add_argument("--prepare-only", action="store_true", help="Stop after canonical panel/coverage output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("FACTORSTRIP V2 — RW PIT BUILD")
    print("=" * 60)
    print("This path intentionally does NOT compute alpha, Sharpe, CAGR, IC, drawdown, or P&L.")

    ohlc = read_rw_frame(args.ohlc)
    fundamentals = read_rw_frame(args.fundamentals)
    panel, coverage = build_rw_r1000_panel(
        ohlc,
        fundamentals,
        RwR1000Config(min_characteristic_coverage=args.coverage_floor),
    )
    paths = write_rw_canonical(panel, coverage, out)

    print(f"canonical panel rows       : {panel.height:,}")
    print(f"canonical panel dates      : {panel['date'].min()} -> {panel['date'].max()}")
    print(f"canonical panel tickers    : {panel['ticker'].n_unique():,}")
    print(f"coverage floor             : {args.coverage_floor:.1%}")

    manifest = {
        "status": "BLINDED_BUILD_ONLY",
        "source": "Robot Wealth R1000 PIT",
        "model": "Toraniko market + size + value; momentum excluded",
        "coverage_floor": args.coverage_floor,
        "panel_rows": panel.height,
        "panel_start": str(panel["date"].min()),
        "panel_end": str(panel["date"].max()),
        "unique_tickers": panel["ticker"].n_unique(),
        "files": {k: {"path": str(v), "sha256": _sha256(v)} for k, v in paths.items()},
        "forbidden_metrics": sorted(FORBIDDEN_METRICS),
    }

    if not args.prepare_only:
        engine = ToranikoMarketSizeValueEngine()
        result = engine.fit(panel)
        factor_path = out / "toraniko_factor_returns.parquet"
        residual_path = out / "toraniko_residuals.parquet"
        result["factor_returns"].write_parquet(factor_path, compression="zstd")
        result["residuals"].write_parquet(residual_path, compression="zstd")

        signal_paths = {}
        signal_counts = {}
        for spec in (MOM_12_1, MOM_6_1):
            signals = build_monthly_raw_and_residual_signals(
                result["returns_input"], result["residuals"], spec
            )
            path = out / f"momentum_signals_{spec.name.replace('-', '_')}.parquet"
            signals.write_parquet(path, compression="zstd")
            signal_paths[spec.name] = path
            signal_counts[spec.name] = {
                "rows": signals.height,
                "months": signals["signal_month"].n_unique() if signals.height else 0,
                "tickers": signals["symbol"].n_unique() if signals.height else 0,
            }

        manifest["files"].update({
            "factor_returns": {"path": str(factor_path), "sha256": _sha256(factor_path)},
            "residuals": {"path": str(residual_path), "sha256": _sha256(residual_path)},
            **{
                f"signals_{name}": {"path": str(path), "sha256": _sha256(path)}
                for name, path in signal_paths.items()
            },
        })
        manifest["signal_coverage"] = signal_counts
        print("Toraniko factor/residual build: complete")
        for name, counts in signal_counts.items():
            print(f"{name} signal coverage       : {counts['months']} months / {counts['tickers']} tickers")

    manifest_path = out / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved blinded build manifest to {manifest_path}")


if __name__ == "__main__":
    main()
