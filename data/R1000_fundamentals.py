import pyarrow.feather as feather
import polars as pl

PATH = "R1000_fundamentals_1d.feather"

df = pl.from_arrow(
    feather.read_table(PATH, memory_map=True)
)

# Normalize date if needed
if df.schema.get("date") == pl.String:
    df = df.with_columns(pl.col("date").str.to_date())

print("\n=== FUNDAMENTALS SCHEMA ===")
print(df.schema)

print("\n=== SUMMARY ===")
print(
    df.select(
        pl.len().alias("rows"),
        pl.col("ticker").n_unique().alias("unique_tickers"),
        pl.col("date").min().alias("start_date"),
        pl.col("date").max().alias("end_date"),
    )
)

print("\n=== NULL COUNTS ===")
print(
    df.select([
        pl.col(c).null_count().alias(c)
        for c in df.columns
    ])
)

print("\n=== IS_UNIVERSE COUNTS ===")
if "is_universe" in df.columns:
    print(
        df.group_by("is_universe")
          .agg(pl.len().alias("rows"))
          .sort("is_universe")
    )

print("\n=== SAMPLE ===")
print(df.head(10))