# FactorStrip + TLAQ

This is the TLAQ-specific diagnostic layer for FactorStrip.

It does **not** alter TLAQ's trading rules and does **not** automatically hedge VTI. It reconstructs TLAQ from the historical `trades_table.csv`, then measures return attribution, market dependency, diversification failure, and risk concentration.

## Install the add-on

Copy these paths into the root of your existing `factorstrip` repository:

```text
factorstrip/tlaq/
run_tlaq.py
tests/test_tlaq.py
```

Your repository should then contain:

```text
factorstrip/
├── factorstrip/
│   ├── ...existing FactorStrip modules...
│   └── tlaq/
│       ├── __init__.py
│       ├── loader.py
│       ├── dependency.py
│       ├── risk.py
│       └── report.py
├── tests/
│   └── test_tlaq.py
├── run_research.py
└── run_tlaq.py
```

No new package dependencies are required beyond FactorStrip's existing NumPy/Pandas stack.

## Input

Place your TLAQ `trades_table.csv` in the repository root, or pass its full path with `--trades-table`.

The loader expects:

```text
ticker
date
close
shares
exposure
sharetrades
tradevalue
commission
interest
short_borrow_cost
margin_call
```

and the TLAQ rows:

```text
Cash
VTI
TLT
GLD
SVXY
VIXY
```

## Run

From the repository root:

```powershell
python run_tlaq.py --trades-table trades_table.csv
```

The default analysis begins on `2012-01-05`, when the volatility sleeve first appears in the supplied history.

Use every available date:

```powershell
python run_tlaq.py --trades-table trades_table.csv --start all
```

Change rolling windows:

```powershell
python run_tlaq.py `
    --trades-table trades_table.csv `
    --start 2012-01-05 `
    --beta-window 60 `
    --risk-window 60
```

## What the loader reconstructs

Historical NAV:

```text
NAV[t] = sum(exposure[i,t])
```

Historical end-of-day weight:

```text
weight[i,t] = exposure[i,t] / NAV[t]
```

Realized close-to-close P&L uses the shares already held at the prior close:

```text
P&L[i,t] = shares[i,t-1] * (close[i,t] - close[i,t-1])
```

Fees are:

```text
- commission + interest - short_borrow_cost
```

The loader checks that reconstructed total P&L agrees with the NAV change to within floating-point tolerance. If it does not reconcile, it raises an error rather than continuing.

## What the analysis answers

### 1. What caused TLAQ's return?

`daily_return_attribution.csv.gz` decomposes each daily TLAQ return into:

```text
VTI
TLT
GLD
SVXY
VIXY
FEES
```

`worst_days.csv` makes this especially useful by showing the exact contribution of every sleeve on TLAQ's worst days.

### 2. Does TLAQ become more VTI-like in stress?

`strategy_stress_beta.csv` reports TLAQ/VTI beta on:

```text
all days
VTI down days
worst 20% VTI days
worst 10% VTI days
worst 5% VTI days
```

This is a diagnostic of downside/crash beta. It is not an instruction to hedge VTI.

### 3. Does the SVXY/VIXY switch change crash behavior?

`vol_state_stress.csv` repeats stress-beta analysis separately when the position actually held over the return interval was:

```text
SVXY
VIXY
NONE
```

This directly tests whether the VIXY state restores diversification or reduces tail equity exposure.

### 4. Which individual sleeve becomes VTI-like?

`asset_vti_betas.csv` measures VTI beta for:

```text
TLT
GLD
VOL
```

where `VOL` means the return of whichever volatility ETF TLAQ actually held over that interval.

### 5. Which assets stop being independent?

For each active asset, FactorStrip regresses it on the other TLAQ sleeves and computes:

```text
unique_variance_proxy = 1 - R²
```

A high value means the asset still behaves mostly independently of the others. A falling value in a VTI tail regime means diversification is collapsing.

This is stored in:

```text
asset_unique_risk.csv
```

This is a diagnostic proxy, not a full institutional specific-risk forecast.

### 6. Do correlations change in crashes?

`correlations_by_regime.csv` stores pairwise correlations among:

```text
VTI
TLT
GLD
VOL
```

for normal/down/tail VTI regimes.

### 7. What actually dominates TLAQ's risk?

Using the actual historical weights and a rolling covariance estimate:

```text
portfolio variance = w' Sigma w
```

FactorStrip computes component risk contributions in:

```text
rolling_risk_contributions.csv
```

This is the distinction between **capital exposure** and **risk exposure**.

For example, an asset can be only 25% of NAV but contribute 50% of portfolio variance.

## Output folder

Default:

```text
output/tlaq/
```

Files:

```text
REPORT.md
performance.csv
latest_summary.csv
latest_weights.csv
tlaq_nav.csv
tlaq_returns.csv
historical_weights.csv.gz
daily_return_attribution.csv.gz
dollar_attribution.csv
worst_days.csv
strategy_stress_beta.csv
rolling_tlaq_vti_beta.csv
asset_vti_betas.csv
asset_unique_risk.csv
correlations_by_regime.csv
rolling_risk_contributions.csv
vol_state_stats.csv
vol_state_stress.csv
```

## First files to open

Start with:

```text
worst_days.csv
vol_state_stress.csv
strategy_stress_beta.csv
asset_unique_risk.csv
rolling_risk_contributions.csv
```

Together they answer the main TLAQ question:

> When TLAQ has a bad period, did VTI simply fall, did SVXY amplify it, did TLT/GLD fail to diversify, did leverage make the event larger, or did several of those things happen at once?

## Timing convention

The code distinguishes two states:

- `vol_state`: the end-of-day position appearing in the table on that date.
- `held_vol_state`: the position from the prior close that actually earned the current close-to-close return.

Historical realized-return analysis uses `held_vol_state`. Current risk forecasting uses the latest end-of-day weights/state.
