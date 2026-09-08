# FactorStrip V2 — Robot Wealth PIT Build

The blinded feasibility gate greenlit the research build. The subsequent RW
R1000 data audit established a usable point-in-time research archive without
looking at alpha, Sharpe, IC, CAGR, drawdown, or strategy P&L.

## Audited source facts

The local Robot Wealth R1000 archive showed:

- OHLC: 9,438,170 rows, 2,908 historical tickers, 1998-01-02 through 2021-04-01
- 1,674 historical constituents that were no longer members at the dataset end
- PIT `is_universe` membership with roughly 1,000 active members through time
- zero nulls in OHLC/volume/membership fields
- fundamentals: 8,857,090 rows, 2,806 tickers, 1998-12-01 through 2021-04-01
- daily market cap with no source nulls
- active-characteristic coverage: ~98.9% median and ~95.1% fifth percentile

The primary V2 research sample uses RW's historical `is_universe == 1` flag.
Undated sector metadata is **not** used as a historical exposure.

One final source issue remains deliberately open before inferential evaluation:
terminal/delisting economics are not separately exposed/documented by the files
we audited. The build may construct factor residuals and signals, but the clean
alpha/Sharpe/IC experiment remains locked until that policy is resolved and the
EdgeLab registration is final.

## Primary factor model

FactorStrip now uses Toraniko as the primary daily cross-sectional estimator.
The model is intentionally small:

```text
Market + Size + Value
```

- All size/value characteristics are lagged one prior trading observation before explaining day-t returns
- Market: Toraniko's constrained common factor
- Size: 1% cross-sectional winsorization and z-score of `-log(market_cap)`
- Value: 1% cross-sectional winsorization and z-score of `log(1 / price_to_book)`; invalid/non-positive observations receive neutral exposure 0 rather than removing the stock
- Momentum: explicitly excluded because momentum is the signal under test
- Sector: explicitly excluded because the audited RW sector metadata is undated

A single synthetic `ALL` sector bucket is passed to Toraniko so its constrained
market/sector machinery reduces to a market factor with a zero sector return.
Styles are residualized against that common component.

The existing `CrossSectionalFactorEngine` remains an independent correctness and
golden-test implementation; it is no longer the production V2 estimator.

## Blinded signal construction

The build creates, but does not evaluate:

- 12-1 raw momentum
- 12-1 residual momentum standardized by pooled daily residual volatility
- 6-1 raw momentum
- 6-1 residual momentum standardized by pooled daily residual volatility

For 12-1, signal month `t` uses months `t-12 ... t-2`, skipping `t-1`.
No forward return is joined in this path.

## Install

The legacy Feather V1 source can be read directly through PyArrow, but converting
once to Parquet is preferred.

```powershell
uv sync --extra rw
```

The existing `uv.lock` may need to be refreshed on a networked machine because
the execution environment used to prepare this patch cannot resolve packages.

## Run

Using Parquet:

```powershell
uv run python run_v2_rw_build.py `
  --ohlc C:\path\R1000_ohlc_1d.parquet `
  --fundamentals C:\path\R1000_fundamentals_1d.parquet `
  --out v2_rw_output
```

Legacy Feather V1 paths are also accepted if the `rw` extra is installed.

For ingestion/coverage only:

```powershell
uv run python run_v2_rw_build.py `
  --ohlc C:\path\R1000_ohlc_1d.parquet `
  --fundamentals C:\path\R1000_fundamentals_1d.parquet `
  --out v2_rw_output `
  --prepare-only
```

## Outputs

The build writes only research inputs/diagnostics:

```text
rw_r1000_panel.parquet
rw_r1000_coverage.parquet
toraniko_factor_returns.parquet
toraniko_residuals.parquet
momentum_signals_12_1.parquet
momentum_signals_6_1.parquet
build_manifest.json
```

`build_manifest.json` records file hashes and explicitly lists the metrics this
path is forbidden to calculate.

## Next gate

1. run this build on the audited RW files
2. inspect **coverage, factor/residual correctness, and hashes only**
3. generate the Blitz/golden canary and Toraniko-vs-reference toy checks
4. resolve/document terminal-delisting return treatment
5. formally register `research/factorstrip_v2_preregistration.json` in EdgeLab
6. only then expose alpha, paired IC, delta Sharpe, or P&L
