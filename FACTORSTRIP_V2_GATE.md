# FactorStrip V2 — Feasibility Gate

The `factorstrip-v1-freeze` Git tag preserves the exploratory V1 implementation.
The active V2 program begins with a deliberately blinded feasibility gate.

## What the gate asks

Before buying a delisting-complete/PIT equity dataset or building a new factor
engine, estimate the correlation between a cheap prototype of the intended raw
and residual 12-1 momentum sleeves. That correlation determines the prospective
sample requirement for detecting an economically worthwhile incremental alpha.

The prototype uses the current survivor universe only as a **measurement
instrument**. It is not evidence for or against the alpha hypothesis.

## Blinding contract

`run_feasibility_gate.py` may emit only:

- paired raw/residual portfolio-return correlation (`rho`)
- a circular block-bootstrap confidence interval for `rho`
- turnover
- coverage counts
- prospective power/sample-size quantities
- the GREENLIGHT / UNRESOLVABLE decision when available clean history is supplied

It intentionally does **not** calculate or expose alpha, Sharpe, CAGR, IC,
drawdown, cumulative P&L, or the underlying paired strategy return series.
Do not add those metrics to the feasibility path before the real experiment is
preregistered.

## Frozen economic/statistical assumptions

- Net incremental alpha hurdle: **3.0%/year**
- Target portfolio volatility: **10%/year**
- Desired power: **80%**
- Family significance: **5%**
- Registered momentum trials: **2** (`12-1` primary, `6-1` secondary)
- Family-wise control in the feasibility power approximation: Bonferroni
- Power calculation uses the **lower 95% block-bootstrap CI bound for rho** when the interval is positive; if the interval crosses zero, it uses zero (the conservative sample-size case), never the point estimate

The 3% hurdle is an economic requirement: FactorStrip must justify a second
momentum sleeve, PIT data maintenance, and a factor engine. It is not derived
from the retired delta-Sharpe framework.

## Prototype shape

The gate deliberately approximates the intended experiment more closely than V1:

1. daily stock and SPY returns from the current survivor universe
2. rolling market beta, estimated through `t-1`
3. fixed shrinkage of beta toward 1.0
4. beta-adjusted daily residuals
5. monthly aggregation
6. 12-1 raw momentum
7. 12-1 residual momentum standardized by residual volatility
8. identical global long/short construction
9. identical beta-neutral projection
10. identical lagged 10% volatility target
11. identical next-month actual stock returns and transaction-cost convention
12. correlation only

The final PIT/Barra-style residual engine may differ. The gate exists only to
answer whether the full program can plausibly be resolved with obtainable data.

## Run

```bash
uv run python run_feasibility_gate.py --start 2000-01-01
```

If evaluating a candidate data source with, for example, 35 clean usable years:

```bash
uv run python run_feasibility_gate.py \
  --start 2000-01-01 \
  --available-clean-years 35
```

Outputs are written under `feasibility_output/` by default. The gate does not
write portfolio return histories.

## Decision

- `GREENLIGHT`: available clean PIT history meets/exceeds the prospective sample
  requirement. Proceed to data-source selection, PIT universe, reference models,
  and the preregistered FactorStrip experiment.
- `UNRESOLVABLE`: do not buy data or build the full alpha program for this
  registered hurdle/test.
- `MEASUREMENT_ONLY`: no candidate clean-history length was supplied; inspect the
  required-years result before making a data decision.

## Precommitted eventual outcome interpretation

Once the real PIT experiment is run after preregistration:

- alpha <= 0: stop residual momentum as an alpha transformation
- alpha > 0 and delta-Sharpe > 0: residual momentum may replace raw momentum
- alpha > 0 and delta-Sharpe < 0: keep raw momentum as core and allocate **25% of
  portfolio risk budget** to the incremental FactorStrip component
- alpha <= 0 and delta-Sharpe > 0: treat residualization as portfolio engineering,
  not new alpha

Those interpretations are frozen before the feasibility result.
