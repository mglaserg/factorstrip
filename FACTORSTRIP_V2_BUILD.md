# FactorStrip V2 — Greenlit Build

The blinded gate returned `rho_used_for_power = 0.9177` and a prospective
requirement of about 200 clean monthly observations (16.7 years) under the
frozen 3%/10%/80%-power design.  That greenlights the *research build*; it does
not validate residual momentum.

## Architecture now in the repo

### 1. Canonical PIT data boundary

`factorstrip.v2.store.CanonicalStore` persists two source-neutral Parquet tables:

- `bars.parquet`: stable asset ID, historical symbol, unadjusted price/volume,
  and one-period total return
- `security_history.parquet`: source identity plus common-stock and historical
  major-exchange eligibility fields. PIT sector/industry classifications are
  treated as a separate factor-exposure input rather than silently backfilled.

Vendor adapters should end here.  Strategy/factor code should never call a data
vendor directly.

A stable `asset_id` is mandatory because tickers change.  Delisted names must
remain in the historical store.  `total_return` must preserve terminal/delisting
economics supplied by the vendor.

### 2. Mechanical PIT universe

`build_mechanical_universe` deliberately does **not** consume S&P 500 membership.
Defaults are a $5 lagged price floor, 63-day ADV, $5m minimum ADV, at least 252
observations, major-exchange/common-stock flags, and top 3000 by lagged ADV.
These are configuration values to freeze in EdgeLab before the clean run.

### 3. Reference implementations

- `blitz_residual_momentum_reference`: independent pandas/numpy methodology
  reference with a 36-month FF3 OLS **with intercept**, 12-1 formation, and
  residual-vol standardization. It is a canary, not a paper replication.
- `estimate_toraniko_reference`: optional Toraniko comparison path. Toraniko is
  not authoritative and remains Polars-native.
- `golden.py`: stable output hashes for a fixed reference sample.

### 4. Authoritative FactorStrip engine

`CrossSectionalFactorEngine` takes Polars long-form returns/exposures and runs a
daily cross-sectional WLS:

```
r_t = X_{t-1} f_t + epsilon_t
```

The NumPy WLS kernel exposes `X'W epsilon` as an explicit diagnostic.  The V2
engine therefore tests the thing the project claims to do: strip the modeled
exposure span cross-sectionally on each date.

`estimate_market_betas` estimates beta using observations strictly before each
return and supports fixed shrinkage toward a prior.  The raw-vs-shrunk beta
comparison remains a formal falsifier for beta-estimation-error momentum.

### 5. Research lock

`research/factorstrip_v2_preregistration.json` is a locked draft of the final
EdgeLab registration.  It intentionally contains no measured alpha/Sharpe/IC.
Do not run the clean PIT inferential experiment until the design is registered
in EdgeLab.

## Data-source implication

For Norgate, the required delisted-US-equity capability is currently in the
Platinum/Diamond US packages.  The Python integration is Windows-local through
Norgate Data Updater, which is why this repo treats Norgate as an ingestion
adapter rather than an application-wide dependency.

## Install V2 dependencies

```
uv sync
```

The V2 core adds Polars.  Toraniko is optional:

```
uv sync --extra reference
```

## Next concrete milestone

1. select/install the delisting-complete source and resolve its capability blockers
2. write its adapter into the canonical Parquet contract
3. run **data quality / coverage only** (`run_v2_data_audit.py`)
4. freeze mechanical-universe parameters in EdgeLab
5. generate and hash the Blitz/reference canary sample
6. validate FactorStrip vs Toraniko on equivalent simple exposures
7. formally register the real experiment before exposing inferential metrics

## Data-source certification

`factorstrip.v2.source_capabilities` records source capabilities explicitly.
Norgate US Platinum is **not silently certified** for the full sector model: it
has stable IDs, delisted names, raw dollar turnover, and historical major-
exchange status, but Norgate documents no delisting-return field and does not
document a PIT GICS classification series in the Python metadata API. Those
items must be resolved by policy/augmentation or a different source before the
clean inferential run.
