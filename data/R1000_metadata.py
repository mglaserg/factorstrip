import pyarrow.feather as feather
import polars as pl

PATH = "R1000_metadata.feather"

df = pl.from_arrow(feather.read_table(PATH, memory_map=True))

print("\n=== METADATA SCHEMA ===")
print(df.schema)

print("\n=== ROWS / TICKERS ===")
print(
    df.select(
        pl.len().alias("rows"),
        pl.col("ticker").n_unique().alias("unique_tickers"),
    )
)

print("\n=== NULL COUNTS ===")
print(
    df.select([
        pl.col(c).null_count().alias(c)
        for c in df.columns
    ])
)

print("\n=== SAMPLE ===")
print(df.head(20))