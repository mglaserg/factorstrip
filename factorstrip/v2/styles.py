from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StyleConfig:
    """Primary V2 style model: size + value, explicitly excluding momentum."""

    winsor_fraction: float = 0.01
    min_cross_section: int = 100

    def validate(self) -> None:
        if not 0 <= self.winsor_fraction < 0.5:
            raise ValueError("winsor_fraction must be in [0, 0.5)")
        if self.min_cross_section < 10:
            raise ValueError("min_cross_section must be >= 10")


def _pl():
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("V2 style construction requires Polars") from exc
    return pl


def _winsor_zscore(expr_name: str, by: str, fraction: float):
    pl = _pl()
    x = pl.col(expr_name)
    lo = x.quantile(fraction).over(by)
    hi = x.quantile(1.0 - fraction).over(by)
    clipped = x.clip(lo, hi)
    mean = clipped.mean().over(by)
    std = clipped.std().over(by)
    return ((clipped - mean) / std).fill_nan(None)


def build_market_size_value_inputs(panel: Any, config: StyleConfig | None = None):
    """Create Toraniko inputs for the clean Market + Size + Value model.

    Size follows the Toraniko/SMB orientation: smaller firms receive higher
    scores via `-log(market_cap)`.  Value uses log book-to-price, available
    point-in-time from RW's daily price-to-book field.  Both are cross-section
    winsorized and standardized. Non-positive/missing book-to-price receives a
    neutral value score of zero after cross-sectional centering rather than
    deleting the stock from the universe. Momentum is intentionally absent
    because it is the signal under test.
    """

    pl = _pl()
    cfg = config or StyleConfig()
    cfg.validate()
    required = {"date", "ticker", "asset_returns", "market_cap", "book_price"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    base = (
        panel.select("date", "ticker", "asset_returns", "market_cap", "book_price")
        .filter(
            pl.col("asset_returns").is_finite()
            & pl.col("market_cap").is_finite()
            & (pl.col("market_cap") > 0)
        )
        .with_columns(
            (-pl.col("market_cap").log()).alias("size_raw"),
            pl.when(pl.col("book_price").is_finite() & (pl.col("book_price") > 0))
            .then(pl.col("book_price").log())
            .otherwise(None)
            .alias("value_raw"),
        )
        .with_columns(
            _winsor_zscore("size_raw", "date", cfg.winsor_fraction).alias("size"),
            _winsor_zscore("value_raw", "date", cfg.winsor_fraction)
            .fill_null(0.0)
            .fill_nan(0.0)
            .alias("value"),
        )
        .drop_nulls(["size"])
        .with_columns(pl.len().over("date").alias("n_cross_section"))
        .filter(pl.col("n_cross_section") >= cfg.min_cross_section)
        .drop("n_cross_section")
        .sort(["date", "ticker"])
    )

    returns_df = base.select(
        "date", pl.col("ticker").alias("symbol"), "asset_returns"
    )
    mkt_cap_df = base.select(
        "date", pl.col("ticker").alias("symbol"), "market_cap"
    )
    # Toraniko's constrained sector machinery needs >=1 sector column.  A
    # single ALL bucket makes the sector factor identically zero, leaving a
    # pure market intercept plus our two style factors.
    sector_df = base.select(
        "date", pl.col("ticker").alias("symbol")
    ).with_columns(pl.lit(1.0).alias("ALL"))
    style_df = base.select(
        "date", pl.col("ticker").alias("symbol"), "size", "value"
    )
    return returns_df, mkt_cap_df, sector_df, style_df
