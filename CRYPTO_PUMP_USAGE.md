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
tests/test_crypto_pump_binance.py
```

No new dependency is required beyond NumPy/Pandas for the research engine. The Binance archive downloader uses Python's standard library.

## The HTTP 451 fix

The original quick downloader used `fapi.binance.com`. Binance can return **HTTP 451** to that live Futures REST endpoint depending on region/network.

Historical research no longer depends on that endpoint.

`--download-binance` now downloads 1-hour USD-M futures klines from Binance's public archive at `data.binance.vision`. By default it stops at the beginning of the current UTC month so it can use completed **monthly** archive packages rather than making hundreds of daily requests.

The old command therefore still works:

```powershell
uv run python run_crypto_pump.py `
  --download-binance `
  --discover 40 `
  --days 120
```

`--discover 40` now means: use the first 40 symbols from FactorStrip's deterministic starter research universe. It does **not** call the live Binance API and it does not claim the universe was point-in-time correct.

Downloaded zip files are cached under:

```text
data/binance_archive_cache/
```

so reruns do not download the same archive files again.

## Better: explicitly choose the universe

For a shitcoin/pump study, explicit symbols are clearer:

```powershell
uv run python run_crypto_pump.py `
  --download-binance `
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,WIFUSDT,1000PEPEUSDT,ENAUSDT,SUIUSDT `
  --days 180
```

Or create `crypto_universe.txt`:

```text
BTCUSDT
ETHUSDT
SOLUSDT
DOGEUSDT
WIFUSDT
1000PEPEUSDT
ENAUSDT
SUIUSDT
```

and run:

```powershell
uv run python run_crypto_pump.py `
  --download-binance `
  --symbols-file crypto_universe.txt `
  --days 180
```

Missing or not-yet-listed symbols in a historical range are skipped and reported instead of crashing the run.

## Optional live discovery

If you are on a network where Binance permits its public Futures REST API, you can explicitly request the old current-volume discovery:

```powershell
uv run python run_crypto_pump.py `
  --download-binance `
  --live-discover `
  --discover 40 `
  --days 120
```

If that call gets HTTP 451, FactorStrip prints the reason and falls back to the archive starter universe rather than dumping a raw traceback.

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

## Archive endpoint

By default, a run on September 23 uses an exclusive archive end of September 1 and therefore only completed monthly files. You can freeze another exclusive UTC endpoint explicitly:

```powershell
uv run python run_crypto_pump.py `
  --download-binance `
  --discover 40 `
  --days 365 `
  --archive-end 2026-09-01
```

This makes research reruns reproducible.

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

The starter universe is convenience, not a decision-grade historical universe. A serious historical test still needs point-in-time contract availability and point-in-time liquidity membership so dead/delisted contracts are not silently removed.
