# FactorStrip

A deliberately transparent cross-sectional equity risk model and
residual-alpha research sandbox.

## V1 model

For stock `i` on day `t`:

```text
return
  = market
  + sector relative to market
  + industry relative to sector
  + stock-specific residual
```

For NVDA, conceptually:

```text
NVDA
  = MARKET
  + Information Technology
  + Semiconductors
  + NVDA-specific residual
```

NVDA is never assigned a Healthcare factor. Classifications explicitly
determine the stock's one allowed branch of the hierarchy.

## Why this exists

The risk model is not alpha. It attempts to strip common movement out of
individual stock returns so that we can test whether the remaining
idiosyncratic return contains predictable structure.

V1 tests one deliberately simple hypothesis:

```text
20-day residual momentum -> future stock returns
```

and compares it with ordinary 20-day raw-return momentum using the same
sector-neutral long/short portfolio construction.

## Install

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -e .
```

For tests:

```bash
pip install pytest
pytest
```

## Run

```bash
python run_research.py --start 2022-01-01
```

Example with different settings:

```bash
python run_research.py \
    --start 2021-01-01 \
    --lookback 20 \
    --quantile 0.20 \
    --cost-bps 5
```

Outputs are written to `output/`.

Important files:

```text
factor_returns.csv.gz
residuals.csv.gz
residual_momentum_signal.csv.gz
residual_momentum_weights.csv.gz
residual_momentum_backtest.csv
raw_momentum_backtest.csv
comparison.csv
diagnostics.csv
```

## Interpretation

If NVDA returns 4% and the model estimates:

```text
MARKET                            +0.7%
Information Technology vs market +0.6%
Semiconductors vs technology     +1.1%
```

then:

```text
NVDA residual = 4.0% - 0.7% - 0.6% - 1.1%
              = 1.6%
```

That 1.6% is the stock-specific move the alpha research layer sees.

## Critical caveat: survivorship bias

`get_sp500_universe()` downloads the CURRENT S&P 500 membership and then
uses those names for historical price research. That is convenient for
a prototype but creates survivorship bias.

Before treating historical Sharpe/CAGR as real evidence, replace that
universe with point-in-time constituent membership.

## Next sensible extensions

1. Point-in-time S&P 500 membership.
2. Point-in-time market-cap weights.
3. Size factor.
4. Value factor.
5. Momentum factor.
6. Factor covariance matrix.
7. Specific-risk estimates.
8. Portfolio risk attribution.
9. Residual mean-reversion tests.
10. Walk-forward / out-of-sample research.

The point is to add these only after the simple model behaves correctly.
