from __future__ import annotations

import numpy as np
import pandas as pd

from factorstrip.crypto_pump import PumpConfig, run_crypto_pump_study
from factorstrip.crypto_pump.factors import build_return_inputs, rolling_factor_residuals


def _synthetic_panel(n=700):
    rng = np.random.default_rng(7)
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    btc_r = rng.normal(0, 0.003, n)
    eth_noise = rng.normal(0, 0.002, n)
    alt_noise = rng.normal(0, 0.002, n)

    # A broad crypto shock. Coins following the common factor structure should
    # look huge in raw-return space but ordinary in residual space.
    btc_r[400] += 0.18
    eth_r = 0.8 * btc_r + eth_noise
    alt = 0.6 * btc_r + 0.2 * eth_r + alt_noise

    symbols = ["BTCUSDT", "ETHUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"]
    rets = {
        "BTCUSDT": btc_r,
        "ETHUSDT": eth_r,
    }
    for i, s in enumerate(symbols[2:]):
        rets[s] = 0.9 * btc_r + 0.3 * eth_r + alt + rng.normal(0, 0.003 + i * 0.0002, n)

    # One genuinely idiosyncratic pump followed by a fade.
    rets["AAAUSDT"][500] += np.log(1.65)
    rets["AAAUSDT"][501:505] += np.log(0.90) / 4

    rows = []
    for s in symbols:
        close = 100 * np.exp(np.cumsum(rets[s]))
        open_ = np.r_[100.0, close[:-1]]
        for i, t in enumerate(ts):
            rows.append(
                {
                    "timestamp": t,
                    "symbol": s,
                    "open": open_[i],
                    "high": max(open_[i], close[i]),
                    "low": min(open_[i], close[i]),
                    "close": close[i],
                    "quote_volume": 2_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def test_current_bar_does_not_fit_itself():
    panel = _synthetic_panel()
    cfg = PumpConfig(factor_window_hours=240, min_factor_obs=168, min_alt_count=5)
    _, _, _, _, returns, factors = build_return_inputs(panel, cfg)
    r1 = rolling_factor_residuals(returns, factors, cfg)

    panel2 = panel.copy()
    mask = (panel2["symbol"] == "AAAUSDT") & (panel2["timestamp"] == panel2["timestamp"].sort_values().unique()[450])
    panel2.loc[mask, "close"] *= 2.0
    _, _, _, _, returns2, factors2 = build_return_inputs(panel2, cfg)
    r2 = rolling_factor_residuals(returns2, factors2, cfg)

    t = returns.index[450]
    # The altered current return should mostly flow into the residual rather than
    # being absorbed by a coefficient estimated with that same return.
    assert abs(r2.at[t, "AAAUSDT"] - r1.at[t, "AAAUSDT"]) > 0.5


def test_idiosyncratic_pump_is_detected_and_fades():
    panel = _synthetic_panel()
    cfg = PumpConfig(
        factor_window_hours=240,
        min_factor_obs=168,
        raw_pump_threshold=0.50,
        residual_pump_threshold=0.30,
        min_quote_volume_24h=1_000_000,
        round_trip_cost_bps=0,
        min_alt_count=5,
    )
    result = run_crypto_pump_study(panel, cfg)
    events = result["events"]
    hit = events[(events.symbol == "AAAUSDT") & (events.signal == "residual_pump")]
    assert not hit.empty
    assert hit.iloc[0]["residual_24h"] > 0.30
    assert hit.iloc[0]["short_net_4h"] > 0


def test_broad_market_pump_is_not_mistaken_for_residual_pump():
    panel = _synthetic_panel()
    cfg = PumpConfig(
        factor_window_hours=240,
        min_factor_obs=168,
        raw_pump_threshold=0.50,
        residual_pump_threshold=0.30,
        min_quote_volume_24h=1_000_000,
        round_trip_cost_bps=0,
        min_alt_count=5,
    )
    result = run_crypto_pump_study(panel, cfg)
    events = result["events"]
    broad_raw = events[(events.symbol == "BBBUSDT") & (events.signal == "raw_pump")]
    broad_resid = events[(events.symbol == "BBBUSDT") & (events.signal == "residual_pump")]
    assert not broad_raw.empty
    # The common-factor-adjusted move stays below the residual-pump threshold.
    assert broad_resid.empty or broad_resid["signal_timestamp"].min() > broad_raw["signal_timestamp"].min()
