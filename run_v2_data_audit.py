from __future__ import annotations

import argparse
import json
from pathlib import Path

from factorstrip.v2.store import CanonicalStore


def parse_args():
    p = argparse.ArgumentParser(
        description="FactorStrip V2 canonical-data audit. No alpha/performance metrics are computed."
    )
    p.add_argument("store", help="Canonical store directory")
    p.add_argument("--output", default=None, help="Optional JSON summary path")
    return p.parse_args()


def main():
    args = parse_args()
    store = CanonicalStore(args.store)
    bars = store.read_bars()
    meta = store.read_security_history()
    manifest = store.read_manifest()

    import polars as pl

    date_min = bars.select(pl.col("date").min()).item()
    date_max = bars.select(pl.col("date").max()).item()
    asset_count = bars.select(pl.col("asset_id").n_unique()).item()
    row_count = bars.height
    duplicate_count = (
        bars.group_by(["date", "asset_id"]).len().filter(pl.col("len") > 1).height
    )
    null_return_count = bars.select(pl.col("total_return").is_null().sum()).item()
    years = None
    if date_min is not None and date_max is not None:
        years = (date_max - date_min).days / 365.2425

    summary = {
        "manifest": manifest,
        "bars": {
            "rows": int(row_count),
            "assets": int(asset_count),
            "start": None if date_min is None else str(date_min),
            "end": None if date_max is None else str(date_max),
            "raw_history_years": None if years is None else round(float(years), 2),
            "duplicate_date_asset_rows": int(duplicate_count),
            "null_total_returns": int(null_return_count),
        },
        "security_history": {
            "rows": int(meta.height),
            "assets": int(meta.select(pl.col("asset_id").n_unique()).item()),
        },
        "research_metrics_computed": False,
    }

    print("FACTORSTRIP V2 — CANONICAL DATA AUDIT")
    print("=" * 56)
    print(f"source               : {manifest.get('source')}")
    print(f"bars                 : {row_count:,}")
    print(f"unique assets        : {asset_count:,}")
    print(f"history              : {date_min} -> {date_max} ({years:.1f}y)" if years is not None else "history              : n/a")
    print(f"duplicate keys       : {duplicate_count}")
    print(f"null total returns   : {null_return_count:,}")
    print("alpha/Sharpe/IC      : NOT COMPUTED")

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved audit to {path}")


if __name__ == "__main__":
    main()
