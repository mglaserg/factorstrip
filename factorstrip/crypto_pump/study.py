from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PumpConfig
from .factors import make_feature_panel


def _cooldown(mask: pd.Series, hours: int) -> pd.Series:
    out = pd.Series(False, index=mask.index)
    last = None
    delta = pd.Timedelta(hours=hours)
    for ts, flag in mask.fillna(False).items():
        if not flag:
            continue
        if last is None or ts - last >= delta:
            out.loc[ts] = True
            last = ts
    return out


def _event_rows(features: dict, cfg: PumpConfig) -> pd.DataFrame:
    raw = features["raw_24h"]
    resid = features["residual_24h"]
    qv = features["quote_volume_24h"]
    open_ = features["open"]
    close = features["close"]

    excluded = {cfg.btc_symbol, cfg.eth_symbol}
    records: list[dict] = []

    for symbol in raw.columns:
        if symbol in excluded:
            continue
        liquid = qv[symbol] >= cfg.min_quote_volume_24h
        rules = {
            "raw_pump": liquid & (raw[symbol] >= cfg.raw_pump_threshold),
            "residual_pump": liquid
            & (resid[symbol] >= cfg.residual_pump_threshold),
        }
        for signal_name, mask in rules.items():
            keep = _cooldown(mask, cfg.cooldown_hours)
            for ts in keep.index[keep]:
                loc = open_.index.get_indexer([ts])[0]
                if loc < 0 or loc + 1 >= len(open_.index):
                    continue
                entry_ts = open_.index[loc + 1]
                entry = open_.iloc[loc + 1].get(symbol, np.nan)
                if not np.isfinite(entry) or entry <= 0:
                    continue
                row = {
                    "signal": signal_name,
                    "symbol": symbol,
                    "signal_timestamp": ts,
                    "entry_timestamp": entry_ts,
                    "entry_price": float(entry),
                    "raw_24h": float(raw.at[ts, symbol]),
                    "residual_24h": float(resid.at[ts, symbol])
                    if np.isfinite(resid.at[ts, symbol])
                    else np.nan,
                    "quote_volume_24h": float(qv.at[ts, symbol]),
                }
                for h in cfg.horizons_hours:
                    exit_loc = loc + h
                    if exit_loc >= len(close.index):
                        row[f"asset_fwd_{h}h"] = np.nan
                        row[f"short_net_{h}h"] = np.nan
                        continue
                    exit_px = close.iloc[exit_loc].get(symbol, np.nan)
                    if not np.isfinite(exit_px) or exit_px <= 0:
                        row[f"asset_fwd_{h}h"] = np.nan
                        row[f"short_net_{h}h"] = np.nan
                        continue
                    asset_ret = float(exit_px / entry - 1.0)
                    row[f"asset_fwd_{h}h"] = asset_ret
                    row[f"short_net_{h}h"] = (
                        -asset_ret - cfg.round_trip_cost_bps / 10_000.0
                    )
                records.append(row)

    if not records:
        cols = [
            "signal",
            "symbol",
            "signal_timestamp",
            "entry_timestamp",
            "entry_price",
            "raw_24h",
            "residual_24h",
            "quote_volume_24h",
        ]
        for h in cfg.horizons_hours:
            cols += [f"asset_fwd_{h}h", f"short_net_{h}h"]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame.from_records(records).sort_values(
        ["signal_timestamp", "symbol", "signal"]
    )


def summarize_events(events: pd.DataFrame, cfg: PumpConfig) -> pd.DataFrame:
    rows = []
    for signal, grp in events.groupby("signal", sort=True):
        for h in cfg.horizons_hours:
            s = grp[f"short_net_{h}h"].dropna()
            rows.append(
                {
                    "signal": signal,
                    "horizon_hours": h,
                    "n": int(len(s)),
                    "mean_short_net": float(s.mean()) if len(s) else np.nan,
                    "median_short_net": float(s.median()) if len(s) else np.nan,
                    "hit_rate": float((s > 0).mean()) if len(s) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run_crypto_pump_study(panel: pd.DataFrame, cfg: PumpConfig | None = None):
    cfg = cfg or PumpConfig()
    features = make_feature_panel(panel, cfg)
    events = _event_rows(features, cfg)
    summary = summarize_events(events, cfg)
    return {**features, "events": events, "summary": summary, "config": cfg}


def write_study(result: dict, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg: PumpConfig = result["config"]
    result["factors"].to_csv(out / "factors.csv.gz")
    result["events"].to_csv(out / "events.csv", index=False)
    result["summary"].to_csv(out / "summary.csv", index=False)

    lines = [
        "# FactorStrip Crypto Residual-Pump Study",
        "",
        "## Question",
        "",
        "Do extreme common-factor-adjusted 24h crypto pumps fade harder than a dumb raw +50% pump rule?",
        "",
        "## First-pass model",
        "",
        "Hourly coin returns are residualized against BTC, ETH, and the cross-sectional median alt return using a rolling model whose coefficients are estimated only through the prior hour. Residual hourly returns are compounded into a 24h residual move.",
        "",
        "Signals are entered at the next hourly open, use a per-symbol cooldown, and report short returns after the configured round-trip cost.",
        "",
        "## Important limitation",
        "",
        "This is an exploratory measurement study. If the input universe was chosen using today's listed contracts or today's liquidity, historical results have survivorship/universe-selection bias. A decision-grade test needs point-in-time contract availability and point-in-time liquidity membership.",
        "",
        "## Configuration",
        "",
        "```text",
    ]
    lines += [f"{k}={v}" for k, v in asdict(cfg).items()]
    lines += ["```", "", "## Results", ""]
    if result["summary"].empty:
        lines.append("No qualifying events were found.")
    else:
        lines.append(result["summary"].to_markdown(index=False))
    lines += [
        "",
        "## Decision rule for the next stage",
        "",
        "Do not add funding, volume-shock, sector, or ML conditioning unless residual_pump shows a materially different forward-return profile from raw_pump with enough events to justify another test.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
