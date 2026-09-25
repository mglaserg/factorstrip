from __future__ import annotations

import io
from datetime import datetime, timezone
from urllib.error import HTTPError
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from factorstrip.crypto_pump import binance


def _zip_kline(rows: list[list[object]], name: str = "BTCUSDT-1h-2026-08.csv") -> bytes:
    text = "\n".join(",".join(str(x) for x in row) for row in rows) + "\n"
    buf = io.BytesIO()
    with ZipFile(buf, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr(name, text)
    return buf.getvalue()


def test_archive_url_is_usdm_monthly():
    assert binance._archive_url("btcusdt", "1h", 2026, 8) == (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        "BTCUSDT/1h/BTCUSDT-1h-2026-08.zip"
    )


def test_parse_archive_zip_extracts_quote_volume_and_timestamp():
    row = [
        1785542400000,
        "100.0",
        "110.0",
        "90.0",
        "105.0",
        "12.0",
        1785545999999,
        "1260.0",
        42,
        "6.0",
        "630.0",
        "0",
    ]
    frame = binance._parse_kline_zip(_zip_kline([row]), "BTCUSDT")
    assert len(frame) == 1
    assert frame.iloc[0]["symbol"] == "BTCUSDT"
    assert frame.iloc[0]["close"] == 105.0
    assert frame.iloc[0]["quote_volume"] == 1260.0
    assert str(frame.iloc[0]["timestamp"].tz) == "UTC"


def test_fetch_archive_skips_missing_months(monkeypatch):
    row = [
        1782864000000,
        "100",
        "101",
        "99",
        "100.5",
        "10",
        1782867599999,
        "1005",
        10,
        "5",
        "502.5",
        "0",
    ]
    payload = _zip_kline([row], "BTCUSDT-1h-2026-07.csv")

    def fake_download(url, cache_path=None):
        if url.endswith("2026-07.zip"):
            return payload
        return None

    monkeypatch.setattr(binance, "_download_bytes", fake_download)
    frame = binance.fetch_archive_hourly_klines(
        "BTCUSDT",
        datetime(2026, 7, 1, tzinfo=timezone.utc),
        datetime(2026, 9, 1, tzinfo=timezone.utc),
    )
    assert len(frame) == 1


def test_live_451_becomes_actionable_error(monkeypatch):
    def blocked(*args, **kwargs):
        raise HTTPError("https://fapi.binance.com/fapi/v1/exchangeInfo", 451, "Unavailable", {}, None)

    monkeypatch.setattr(binance, "urlopen", blocked)
    with pytest.raises(binance.BinanceLiveAccessError, match="HTTP 451"):
        binance.discover_current_usdt_perpetuals(10)


def test_starter_universe_keeps_required_factors():
    symbols = binance.starter_usdt_perpetuals(40)
    assert len(symbols) == 40
    assert symbols[:2] == ["BTCUSDT", "ETHUSDT"]
    assert "SOLUSDT" in symbols
