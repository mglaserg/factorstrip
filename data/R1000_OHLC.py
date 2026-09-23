import pyarrow.feather as feather
import polars as pl

PATH = "R1000_ohlc_1d.feather"

df = pl.from_arrow(
    feather.read_table(PATH, memory_map=True)
).with_columns(
    pl.col("date").str.to_date(),
    pl.col("is_universe").cast(pl.Int8),
)

print("\n=== FULL SCHEMA ===")
print(df.schema)

print("\n=== NULL COUNTS ===")
print(
    df.select([
        pl.col(c).null_count().alias(c)
        for c in df.columns
    ])
)

print("\n=== PIT UNIVERSE SIZE BY YEAR ===")
print(
    df.filter(pl.col("is_universe") == 1)
      .with_columns(pl.col("date").dt.year().alias("year"))
      .group_by("year")
      .agg([
          pl.col("ticker").n_unique().alias("names_seen"),
          pl.len().alias("rows")
      ])
      .sort("year")
)

print("\n=== MONTH-END UNIVERSE SIZE ===")
monthly = (
    df.filter(pl.col("is_universe") == 1)
      .with_columns(
          pl.col("date").dt.truncate("1mo").alias("month")
      )
      .group_by("month")
      .agg(pl.col("ticker").n_unique().alias("n_members"))
      .sort("month")
)

print(monthly.head(12))
print("...")
print(monthly.tail(12))

print("\n=== UNIVERSE SIZE SUMMARY ===")
print(
    monthly.select([
        pl.col("n_members").min().alias("min"),
        pl.col("n_members").median().alias("median"),
        pl.col("n_members").max().alias("max"),
    ])
)