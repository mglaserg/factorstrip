from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PumpConfig:
    """Configuration for the first-pass crypto residual-pump study.

    The first pass deliberately strips only three common return factors:
    BTC, ETH, and the cross-sectional median alt return.  Liquidity is a
    universe filter, not another fitted factor.  This keeps the initial
    comparison interpretable: dumb raw pump vs common-factor-adjusted pump.
    """

    factor_window_hours: int = 24 * 14
    min_factor_obs: int = 24 * 7
    pump_lookback_hours: int = 24
    raw_pump_threshold: float = 0.50
    residual_pump_threshold: float = 0.30
    min_quote_volume_24h: float = 5_000_000.0
    cooldown_hours: int = 24
    horizons_hours: tuple[int, ...] = (1, 4, 12, 24, 72)
    round_trip_cost_bps: float = 20.0
    ridge: float = 1e-8
    min_alt_count: int = 5
    btc_symbol: str = "BTCUSDT"
    eth_symbol: str = "ETHUSDT"
