from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd

LIVE_BASE_URL = "https://fapi.binance.com"
ARCHIVE_BASE_URL = "https://data.binance.vision/data/futures/um"

# This is a convenience universe for exploratory research, not a point-in-time
# membership definition. Missing/delisted symbols are skipped by the archive
# downloader. Keep BTC/ETH first because FactorStrip requires them as factors.
STARTER_USDT_PERPETUALS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "TRXUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "SUIUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "NEARUSDT",
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
    "INJUSDT",
    "SEIUSDT",
    "TIAUSDT",
    "FILUSDT",
    "ATOMUSDT",
    "AAVEUSDT",
    "UNIUSDT",
    "ETCUSDT",
    "WLDUSDT",
    "ENAUSDT",
    "WIFUSDT",
    "1000PEPEUSDT",
    "1000BONKUSDT",
    "1000SHIBUSDT",
    "FETUSDT",
    "RUNEUSDT",
    "JUPUSDT",
    "PYTHUSDT",
    "ORDIUSDT",
    "STXUSDT",
    "GALAUSDT",
    "SANDUSDT",
    "MANAUSDT",
    "DYDXUSDT",
    "CRVUSDT",
    "LDOUSDT",
    "PENDLEUSDT",
    "IMXUSDT",
    "ICPUSDT",
    "MKRUSDT",
    "ENSUSDT",
    "RENDERUSDT",
]

KLINE_COLUMNS = [
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
]
OUTPUT_COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close", "quote_volume"]


class BinanceLiveAccessError(RuntimeError):
    """Raised when Binance's live futures REST API is unavailable/restricted."""


def starter_usdt_perpetuals(limit: int = 40) -> list[str]:
    """Return a deterministic research universe without calling Binance REST.

    This preserves the old ``--discover N`` convenience for users whose region
    cannot call fapi.binance.com. It is intentionally *not* presented as a
    current or point-in-time universe.
    """

    if limit < 2:
        raise ValueError("--discover must be at least 2 so BTCUSDT and ETHUSDT are present")
    return STARTER_USDT_PERPETUALS[: min(limit, len(STARTER_USDT_PERPETUALS))]


def _live_get(path: str, params: dict | None = None):
    url = LIVE_BASE_URL + path
    if params:
        url += "?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "FactorStrip/crypto-pump"})
    try:
        with urlopen(req, timeout=30) as resp:  # nosec - public market-data API
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 451:
            raise BinanceLiveAccessError(
                "Binance live USD-M futures REST returned HTTP 451 for this network/region. "
                "Use the archive downloader (the default for --download-binance) or pass "
                "--symbols/--symbols-file. Historical research does not require fapi.binance.com."
            ) from exc
        raise


def discover_current_usdt_perpetuals(limit: int = 40) -> list[str]:
    """Live current high-volume universe; optional and not PIT-safe historically.

    This function is retained for users in regions where Binance permits the
    public Futures REST API. The CLI no longer calls it by default.
    """

    info = _live_get("/fapi/v1/exchangeInfo")
    allowed = {
        s["symbol"]
        for s in info.get("symbols", [])
        if s.get("contractType") == "PERPETUAL"
        and s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
    }
    tickers = _live_get("/fapi/v1/ticker/24hr")
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


def _archive_url(symbol: str, interval: str, year: int, month: int) -> str:
    symbol = symbol.upper()
    filename = f"{symbol}-{interval}-{year:04d}-{month:02d}.zip"
    return f"{ARCHIVE_BASE_URL}/monthly/klines/{symbol}/{interval}/{filename}"


def _download_bytes(url: str, cache_path: Path | None = None) -> bytes | None:
    if cache_path is not None and cache_path.exists():
        return cache_path.read_bytes()

    req = Request(url, headers={"User-Agent": "FactorStrip/crypto-pump"})
    try:
        with urlopen(req, timeout=60) as resp:  # nosec - public Binance archive
            payload = resp.read()
    except HTTPError as exc:
        if exc.code == 404:
            return None
        if exc.code == 451:
            raise RuntimeError(
                "Binance public archive returned HTTP 451. This is separate from the live "
                "Futures REST restriction; use --input with an existing hourly panel."
            ) from exc
        raise
    except URLError as exc:
        raise RuntimeError(f"could not download Binance archive file: {url}: {exc}") from exc

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(payload)
    return payload


def _parse_kline_zip(payload: bytes, symbol: str) -> pd.DataFrame:
    try:
        with ZipFile(io.BytesIO(payload)) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if not names:
                return pd.DataFrame(columns=OUTPUT_COLUMNS)
            with zf.open(names[0]) as handle:
                out = pd.read_csv(handle, header=None, names=KLINE_COLUMNS)
    except BadZipFile as exc:
        raise ValueError(f"invalid Binance archive zip for {symbol}") from exc

    # A few archive generations/tools may include a header row. Coercing the
    # timestamp is a cheap way to discard it without assuming a specific file era.
    raw_time = pd.to_numeric(out["open_time"], errors="coerce")
    out = out.loc[raw_time.notna()].copy()
    raw_time = raw_time.loc[raw_time.notna()].astype("int64")
    if out.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Futures archives are normally milliseconds. Be tolerant of microseconds
    # in case Binance changes timestamp precision as it has for other datasets.
    unit = "us" if raw_time.abs().median() >= 10**15 else "ms"
    out["timestamp"] = pd.to_datetime(raw_time, unit=unit, utc=True)
    out["symbol"] = symbol.upper()
    for col in ("open", "high", "low", "close", "quote_volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[OUTPUT_COLUMNS].dropna(subset=["timestamp", "open", "close"])


def _month_starts(start: datetime, end: datetime):
    cursor = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while cursor < end:
        yield cursor
        if cursor.month == 12:
            cursor = datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            cursor = datetime(cursor.year, cursor.month + 1, 1, tzinfo=timezone.utc)


def latest_completed_month_end(now: datetime | None = None) -> datetime:
    """Exclusive UTC endpoint at the start of the current month."""

    now = now or datetime.now(timezone.utc)
    return datetime(now.year, now.month, 1, tzinfo=timezone.utc)


def fetch_archive_hourly_klines(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch USD-M 1h klines from Binance's public monthly archive.

    ``end`` is exclusive. The CLI defaults it to the start of the current UTC
    month so only completed monthly packages are required. This avoids the
    region-restricted live Futures REST API entirely.
    """

    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start/end must be timezone-aware")
    if start >= end:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    frames: list[pd.DataFrame] = []
    for month in _month_starts(start, end):
        url = _archive_url(symbol, "1h", month.year, month.month)
        cache_path = None
        if cache_dir is not None:
            cache_path = cache_dir / symbol.upper() / Path(url).name
        payload = _download_bytes(url, cache_path)
        if payload is None:
            continue
        frame = _parse_kline_zip(payload, symbol)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return (
        out[(out["timestamp"] >= start_ts) & (out["timestamp"] < end_ts)]
        .drop_duplicates(["timestamp", "symbol"], keep="last")
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )


def download_archive_panel(
    symbols: list[str],
    days: int = 120,
    *,
    end: datetime | None = None,
    cache_dir: Path | None = None,
    workers: int = 8,
) -> tuple[pd.DataFrame, list[str]]:
    """Download a multi-symbol hourly panel from completed monthly archives.

    Returns ``(panel, missing_symbols)``. Missing symbols are normal for a
    convenience universe because listings/delistings differ through time.
    """

    if days <= 0:
        raise ValueError("days must be positive")
    end = end or latest_completed_month_end()
    start = end - timedelta(days=days)
    symbols = list(dict.fromkeys(s.upper() for s in symbols))

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    max_workers = max(1, min(workers, len(symbols))) if symbols else 1

    def one(symbol: str):
        return symbol, fetch_archive_hourly_klines(
            symbol, start, end, cache_dir=cache_dir
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(one, s): s for s in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            sym, frame = future.result()
            if frame.empty:
                missing.append(sym)
            else:
                frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), sorted(missing)
    panel = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])
    return panel.reset_index(drop=True), sorted(missing)


# Backward-compatible name used by the first add-on. It now uses the public
# archive instead of the live REST API.
def download_recent_panel(symbols: list[str], days: int = 120) -> pd.DataFrame:
    panel, _ = download_archive_panel(symbols, days)
    return panel
