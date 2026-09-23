import pyarrow.feather as feather
import polars as pl

ohlc = pl.from_arrow(
    feather.read_table("R1000_ohlc_1d.feather", memory_map=True)
).with_columns(
    pl.col("date").str.to_date()
    if pl.from_arrow(
        feather.read_table(
            "R1000_ohlc_1d.feather",
            columns=["date"],
            memory_map=True,
        )
    ).schema["date"] == pl.String
    else pl.col("date")
)

fund = pl.from_arrow(
    feather.read_table("R1000_fundamentals_1d.feather", memory_map=True)
)

if fund.schema["date"] == pl.String:
    fund = fund.with_columns(pl.col("date").str.to_date())

active = (
    ohlc
    .filter(pl.col("is_universe") == 1)
    .select("ticker", "date")
)

coverage = (
    active.join(
        fund.select("ticker", "date", "market_cap", "price_to_book"),
        on=["ticker", "date"],
        how="left",
    )
)

print("\n=== ACTIVE PIT COVERAGE ===")
print(
    coverage.select(
        pl.len().alias("active_rows"),
        pl.col("market_cap").is_not_null().sum().alias("market_cap_rows"),
        pl.col("price_to_book").is_not_null().sum().alias("ptb_rows"),
    )
)

missing_tickers = (
    coverage
    .filter(pl.col("market_cap").is_null())
    .select("ticker")
    .unique()
)

print("\n=== ACTIVE TICKERS EVER MISSING FUNDAMENTALS ===")
print("count:", missing_tickers.height)
print(missing_tickers.head(30))

daily = (
    coverage
    .group_by("date")
    .agg(
        pl.len().alias("n_active"),
        pl.col("market_cap").is_not_null().sum().alias("n_mcap"),
        pl.col("price_to_book").is_not_null().sum().alias("n_ptb"),
    )
    .with_columns(
        (pl.col("n_mcap") / pl.col("n_active")).alias("mcap_coverage"),
        (pl.col("n_ptb") / pl.col("n_active")).alias("ptb_coverage"),
    )
    .sort("date")
)

print(
    daily.select(
        pl.col("mcap_coverage").min().alias("mcap_min"),
        pl.col("mcap_coverage").median().alias("mcap_median"),
        pl.col("mcap_coverage").quantile(0.05).alias("mcap_p05"),
        pl.col("ptb_coverage").min().alias("ptb_min"),
        pl.col("ptb_coverage").median().alias("ptb_median"),
        pl.col("ptb_coverage").quantile(0.05).alias("ptb_p05"),
    )
)