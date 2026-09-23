from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

BASE_URL = "https://fapi.binance.com"


def _get(path: str, params: dict | None = None):
    url = BASE_URL + path
    if params:
        url += "?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "FactorStrip/crypto-pump"})
    with urlopen(req, timeout=30) as resp:  # nosec - public Binance market-data API
        return json.loads(resp.read().decode("utf-8"))


def discover_current_usdt_perpetuals(limit: int = 40) -> list[str]:
    """Current high-volume universe. Exploratory only; not PIT-safe historically."""
    info = _get("/fapi/v1/exchangeInfo")
    allowed = {
        s["symbol"]
        for s in info.get("symbols", [])
        if s.get("contractType") == "PERPETUAL"
        and s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
    }
    tickers = _get("/fapi/v1/ticker/24hr")
    ranked = []
    for row in tickers:
        symbol = row.get("symbol")
        if symbol in allowed:
            try:
                qv = float(row.get("quoteVolume", 0.0))
            except (TypeError, ValueError):
                qv = 0.0
            ranked.append((qv, symbol))
    ranked.sort(reverse=True)
    symbols = [s for _, s in ranked if s not in {"BTCUSDT", "ETHUSDT"}]
    selected = symbols[: max(0, limit - 2)]
    return ["BTCUSDT", "ETHUSDT", *selected]


def fetch_hourly_klines(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        data = _get(
            "/fapi/v1/klines",
            {
                "symbol": symbol,
                "interval": "1h",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,
            },
        )
        if not data:
            break
        rows.extend(data)
        nxt = int(data[-1][0]) + 60 * 60 * 1000
        if nxt <= cursor:
            break
        cursor = nxt
        time.sleep(0.03)

    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "open", "high", "low", "close", "quote_volume"]
        )
    out = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_base",
            "taker_quote",
            "ignore",
        ],
    )
    out["timestamp"] = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    out["symbol"] = symbol.upper()
    for col in ("open", "high", "low", "close", "quote_volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[["timestamp", "symbol", "open", "high", "low", "close", "quote_volume"]]


def download_recent_panel(symbols: list[str], days: int = 120) -> pd.DataFrame:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    frames = [fetch_hourly_klines(s, start, end) for s in symbols]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])
