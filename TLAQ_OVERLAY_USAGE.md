# TLAQ SVXY Overlay Test

This add-on tests whether reducing **only SVXY** when TLAQ is unusually
equity-sensitive improves the existing strategy.

It does not change the SVXY/VIXY switching rule and never reduces VIXY.

## Install

Copy these files into the existing FactorStrip repository:

```text
factorstrip/tlaq/overlay.py
run_tlaq_overlay.py
```

## Run

```powershell
python run_tlaq_overlay.py --trades-table trades_table.csv
```

Add a conservative cost for overlay-induced SVXY trading:

```powershell
python run_tlaq_overlay.py --trades-table trades_table.csv --cost-bps 5
```

## Rules tested

### Beta-conditioned haircut

Uses yesterday's rolling 60-day TLAQ/VTI beta.

Examples:

```text
beta > 1.00 -> SVXY x 75%
beta > 1.00 -> SVXY x 50%
beta > 1.25 -> SVXY x 75%
beta > 1.25 -> SVXY x 50%
beta > 1.50 -> SVXY x 75%
beta > 1.50 -> SVXY x 50%
beta > 1.50 -> SVXY x 0%
```

Freed SVXY exposure goes to cash.

### Lagged joint-down diagnostic

If VTI, TLT and GLD were all negative on the PRIOR trading day while
TLAQ's current target remains SVXY, reduce the SVXY target.

The prior-day lag is intentional. A same-close implementation looks much
better historically but uses today's exact closing return to alter a
position assumed to be established at that same close, which is not a
clean backtest.

## Interpretation

Do not select a rule simply because it has the best in-sample Sharpe.
The useful questions are:

1. Does the rule materially reduce drawdown?
2. Does it preserve most of TLAQ's CAGR?
3. Does it work in multiple subperiods?
4. Does it survive incremental trading costs?
5. Is the trigger economically interpretable rather than data-mined?
