from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RwR1000Config:
    """Frozen data-quality rules for the RW Russell 1000 research panel."""

    min_characteristic_coverage: float = 0.90
    require_positive_market_cap: bool = True
    require_positive_price_to_book: bool = False
    source_name: str = "Robot Wealth R1000 PIT"
    characteristic_lag_days: int = 1

    def validate(self) -> None:
        if not 0 < self.min_characteristic_coverage <= 1:
            raise ValueError("min_characteristic_coverage must be in (0, 1]")
        if self.characteristic_lag_days != 1:
            raise ValueError("primary RW V2 design is frozen at a one-observation characteristic lag")


def _pl():
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("Robot Wealth V2 ingestion requires Polars") from exc
    return pl


def read_rw_frame(path: str | Path):
    """Read RW Parquet or legacy Feather V1 into Polars.

    RW's archived R1000 files are Feather V1 (FEA1).  Modern Polars cannot scan
    those directly, so PyArrow is used only at this ingestion boundary.  Once
    ingested, FactorStrip writes Parquet and never depends on Feather V1 again.
    """

    pl = _pl()
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix in {".feather", ".arrow", ".ipc"}:
        try:
            import pyarrow.feather as feather
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Reading RW Feather V1 requires PyArrow. Install FactorStrip's `rw` extra."
            ) from exc
        return pl.from_arrow(feather.read_table(path, memory_map=True))
    raise ValueError(f"Unsupported RW file type: {path.suffix}")


def _normalize_date(df: Any, name: str = "date"):
    pl = _pl()
    dtype = df.schema.get(name)
    if dtype == pl.String:
        return df.with_columns(pl.col(name).str.to_date())
    return df


def normalize_rw_ohlc(df: Any):
    """Normalize RW R1000 OHLC without evaluating strategy performance."""

    pl = _pl()
    required = {
        "ticker", "date", "open", "high", "low", "close",
        "unadjusted_close", "volume", "is_universe",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"RW OHLC missing columns: {sorted(missing)}")
    df = _normalize_date(df)
    return (
        df.with_columns(
            pl.col("ticker").cast(pl.String),
            pl.col("is_universe").cast(pl.Int8),
            (pl.lit("RW_R1000::") + pl.col("ticker")).alias("asset_id"),
        )
        .sort(["ticker", "date"])
        .with_columns(
            pl.col("close").pct_change().over("ticker").alias("asset_returns")
        )
    )


def normalize_rw_fundamentals(df: Any):
    pl = _pl()
    required = {
        "ticker", "date", "market_cap", "price_to_book", "is_universe"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"RW fundamentals missing columns: {sorted(missing)}")
    df = _normalize_date(df)
    return df.with_columns(
        pl.col("ticker").cast(pl.String),
        pl.col("is_universe").cast(pl.Int8),
        (pl.lit("RW_R1000::") + pl.col("ticker")).alias("asset_id"),
    )


def lag_rw_characteristics(fundamentals: Any, lag_days: int = 1):
    """Lag price-derived/fundamental exposures before explaining day-t returns."""

    pl = _pl()
    if lag_days != 1:
        raise ValueError("primary RW V2 design is frozen at a one-observation characteristic lag")
    cols = [
        c for c in (
            "market_cap", "price_to_book", "price_to_earnings",
            "price_to_sales", "enterprise_value"
        ) if c in fundamentals.columns
    ]
    return (
        fundamentals.sort(["ticker", "date"])
        .with_columns(
            *[
                pl.col(c).shift(lag_days).over("ticker").alias(f"{c}_lag")
                for c in cols
            ]
        )
    )


def daily_characteristic_coverage(ohlc: Any, fundamentals: Any):
    """Coverage-only audit for active PIT members; computes no alpha metrics."""

    pl = _pl()
    active = ohlc.filter(pl.col("is_universe") == 1).select("ticker", "date")
    lagged = lag_rw_characteristics(fundamentals)
    coverage = active.join(
        lagged.select("ticker", "date", "market_cap_lag", "price_to_book_lag"),
        on=["ticker", "date"],
        how="left",
    )
    return (
        coverage.group_by("date")
        .agg(
            pl.len().alias("n_active"),
            pl.col("market_cap_lag").is_not_null().sum().alias("n_mcap"),
            pl.col("price_to_book_lag").is_not_null().sum().alias("n_ptb"),
        )
        .with_columns(
            (pl.col("n_mcap") / pl.col("n_active")).alias("mcap_coverage"),
            (pl.col("n_ptb") / pl.col("n_active")).alias("ptb_coverage"),
        )
        .sort("date")
    )


def build_rw_r1000_panel(
    ohlc: Any,
    fundamentals: Any,
    config: RwR1000Config | None = None,
):
    """Build the clean PIT panel used by FactorStrip V2.

    Universe membership is RW's historical `is_universe` flag.  Dates below
    the frozen 90% characteristic-coverage threshold are excluded rather than
    silently imputed.  Individual rows missing required style characteristics
    are also excluded from the factor cross-section.
    """

    pl = _pl()
    cfg = config or RwR1000Config()
    cfg.validate()
    ohlc = normalize_rw_ohlc(ohlc)
    fundamentals = normalize_rw_fundamentals(fundamentals)

    start = max(ohlc["date"].min(), fundamentals["date"].min())
    ohlc = ohlc.filter(pl.col("date") >= start)
    fundamentals = fundamentals.filter(pl.col("date") >= start)
    lagged_fundamentals = lag_rw_characteristics(fundamentals, cfg.characteristic_lag_days)
    coverage = daily_characteristic_coverage(ohlc, fundamentals)
    good_dates = coverage.filter(
        (pl.col("mcap_coverage") >= cfg.min_characteristic_coverage)
        & (pl.col("ptb_coverage") >= cfg.min_characteristic_coverage)
    ).select("date")

    panel = (
        ohlc.filter(pl.col("is_universe") == 1)
        .join(good_dates, on="date", how="inner")
        .join(
            lagged_fundamentals.select(
                "ticker", "date", "market_cap_lag", "price_to_book_lag",
                "price_to_earnings_lag", "price_to_sales_lag", "enterprise_value_lag",
            ),
            on=["ticker", "date"],
            how="left",
        )
    )
    if cfg.require_positive_market_cap:
        panel = panel.filter(pl.col("market_cap_lag").is_finite() & (pl.col("market_cap_lag") > 0))
    if cfg.require_positive_price_to_book:
        panel = panel.filter(pl.col("price_to_book_lag").is_finite() & (pl.col("price_to_book_lag") > 0))

    panel = panel.with_columns(
        pl.col("market_cap_lag").alias("market_cap"),
        pl.col("price_to_book_lag").alias("price_to_book"),
        pl.when(pl.col("price_to_book_lag").is_finite() & (pl.col("price_to_book_lag") > 0))
        .then(1.0 / pl.col("price_to_book_lag"))
        .otherwise(None)
        .alias("book_price"),
    ).sort(["date", "ticker"])
    return panel, coverage


def write_rw_canonical(
    panel: Any,
    coverage: Any,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Persist the source-normalized RW research inputs as versioned Parquet."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "panel": output_dir / "rw_r1000_panel.parquet",
        "coverage": output_dir / "rw_r1000_coverage.parquet",
    }
    panel.write_parquet(paths["panel"], compression="zstd")
    coverage.write_parquet(paths["coverage"], compression="zstd")
    return paths
