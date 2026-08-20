from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf
from io import StringIO
import requests

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def get_sp500_universe(cache_path: str | Path | None = None) -> pd.DataFrame:
    """
    Download the CURRENT S&P 500 constituent table.

    WARNING:
        Using today's constituents for a historical backtest creates
        survivorship bias.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)

        if cache_path.exists():
            return pd.read_csv(cache_path)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    response = requests.get(
        SP500_URL,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    table = pd.read_html(
        StringIO(response.text),
        match="Symbol",
    )[0]

    universe = (
        table[
            [
                "Symbol",
                "Security",
                "GICS Sector",
                "GICS Sub-Industry",
            ]
        ]
        .rename(
            columns={
                "Symbol": "ticker",
                "Security": "company",
                "GICS Sector": "sector",
                "GICS Sub-Industry": "industry",
            }
        )
        .copy()
    )

    # Yahoo uses BRK-B instead of BRK.B, etc.
    universe["yahoo_ticker"] = (
        universe["ticker"]
        .str.replace(".", "-", regex=False)
    )

    universe = universe[
        [
            "ticker",
            "yahoo_ticker",
            "company",
            "sector",
            "industry",
        ]
    ]

    if cache_path is not None:
        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        universe.to_csv(
            cache_path,
            index=False,
        )

    return universe

def get_sp500_universe_pandas(cache_path: str | Path | None = None) -> pd.DataFrame:
    """
    Download the CURRENT S&P 500 constituent table.

    Columns returned:
        ticker
        yahoo_ticker
        company
        sector
        industry

    WARNING:
        Using today's constituents for a historical backtest creates
        survivorship bias. This helper is for prototyping. For serious
        historical research, supply point-in-time constituents.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return pd.read_csv(cache_path)

    table = pd.read_html(SP500_URL, match="Symbol")[0]

    universe = (
        table[
            ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]
        ]
        .rename(
            columns={
                "Symbol": "ticker",
                "Security": "company",
                "GICS Sector": "sector",
                "GICS Sub-Industry": "industry",
            }
        )
        .copy()
    )

    # Yahoo uses BRK-B rather than BRK.B, etc.
    universe["yahoo_ticker"] = (
        universe["ticker"].str.replace(".", "-", regex=False)
    )

    universe = universe[
        ["ticker", "yahoo_ticker", "company", "sector", "industry"]
    ]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        universe.to_csv(cache_path, index=False)

    return universe


def download_adjusted_close(
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download adjusted price history from Yahoo Finance.

    auto_adjust=True means the returned Close series is adjusted for
    splits/dividends by yfinance.
    """
    tickers = list(dict.fromkeys(tickers))

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )

    if raw.empty:
        raise RuntimeError("No price data returned by yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        # Standard multi-ticker yfinance layout:
        # first level = Price field, second level = ticker.
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        elif "Close" in raw.columns.get_level_values(1):
            close = raw.xs("Close", axis=1, level=1).copy()
        else:
            raise RuntimeError(
                "Could not locate Close prices in yfinance result."
            )
    else:
        if "Close" not in raw.columns:
            raise RuntimeError(
                "Could not locate Close prices in yfinance result."
            )
        if len(tickers) != 1:
            raise RuntimeError("Unexpected single-level yfinance columns.")
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})

    close.index = pd.to_datetime(close.index)
    close = close.sort_index()

    return close


def download_returns(
    universe: pd.DataFrame,
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download prices and return a date x canonical-ticker return matrix.
    """
    yahoo_to_canonical = dict(
        zip(universe["yahoo_ticker"], universe["ticker"])
    )

    close = download_adjusted_close(
        universe["yahoo_ticker"],
        start=start,
        end=end,
    )

    close = close.rename(columns=yahoo_to_canonical)

    # Keep only names actually returned.
    canonical = [x for x in universe["ticker"] if x in close.columns]
    close = close[canonical]

    returns = close.pct_change(fill_method=None)

    # Do not drop whole dates merely because one stock is missing.
    return returns
