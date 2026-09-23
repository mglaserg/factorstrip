# FactorStrip Crypto Residual-Pump Study

This is a **research add-on** for FactorStrip. It does not change the existing equity/Toraniko V2 pipeline.

The first question is intentionally narrow:

> Do extreme 24-hour crypto pumps fade harder after we strip the common BTC/ETH/alt-market move?

## What it does

For each coin, FactorStrip fits an hourly rolling model:

```text
coin_return ~= intercept + BTC + ETH + median_alt_market
```

At hour `t`, coefficients are estimated using data only through `t-1`. The current return is therefore out-of-sample to that rolling fit.

Hourly residuals are compounded over 24 hours. We compare:

```text
raw_pump      = raw 24h return >= 50%
residual_pump = factor-stripped 24h residual return >= 30%
```

Both require a trailing 24-hour quote-volume floor. Signals enter at the **next hourly open**, use a per-symbol cooldown, and report 1h / 4h / 12h / 24h / 72h short returns after round-trip costs.

This is the deliberate first pass. No funding, volume-shock interaction, sectors, XGBoost, or other cleverness yet.

## Install into FactorStrip

Copy these paths into the repository root:

```text
factorstrip/crypto_pump/
run_crypto_pump.py
CRYPTO_PUMP_USAGE.md
tests/test_crypto_pump.py
```

No new dependency is required beyond NumPy/Pandas for the research engine. The Binance downloader uses Python's standard library.

## Run on an existing hourly panel

Required columns:

```text
timestamp
symbol
open
close
quote_volume
```

`high` and `low` may also be present. `BTCUSDT` and `ETHUSDT` must be included because they are common factors.

```powershell
uv run python run_crypto_pump.py --input data/crypto_hourly.parquet
```

## Quick exploratory Binance run

```powershell
uv run python run_crypto_pump.py --download-binance --discover 40 --days 120
```

Or explicitly choose the contracts:

```powershell
uv run python run_crypto_pump.py `
  --download-binance `
  --symbols BTCUSDT,ETHUSDT,DOGEUSDT,WIFUSDT,1000PEPEUSDT,ENAUSDT `
  --days 180
```

**Important:** `--discover` ranks the *currently* listed/traded USD-M contracts by current 24-hour volume. That is useful for a quick recent EDA run, but it is **not** a point-in-time historical universe and must not be treated as a decision-grade backtest.

## Useful knobs

```powershell
uv run python run_crypto_pump.py `
  --input data/crypto_hourly.parquet `
  --raw-threshold 0.50 `
  --residual-threshold 0.30 `
  --min-qv-24h 5000000 `
  --cost-bps 20 `
  --factor-window-hours 336 `
  --cooldown-hours 24
```

## Outputs

Default folder:

```text
output/crypto_pump/
```

Files:

```text
REPORT.md
summary.csv
events.csv
factors.csv.gz
```

If Binance downloading is used, the exact downloaded panel is also frozen as:

```text
input_panel.csv.gz
```

## What decides whether we continue

Open `summary.csv` and compare `raw_pump` with `residual_pump`.

We are **not** looking for the prettiest threshold. We are asking whether residualization changes the forward-return distribution enough to justify another experiment.

If it does, the next experiment is residual pump + funding/volume shock. If it does not, keep the dumb raw-pump rule.

## Research caveat

A serious historical test needs point-in-time contract availability and point-in-time liquidity membership so delisted/dead contracts are not silently removed. The current downloader is intentionally labeled exploratory because current exchange membership is not sufficient for that job.
