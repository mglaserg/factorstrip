from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .dependency import (
    conditional_vti_betas,
    regression_stats,
    rolling_beta,
    unique_risk_table,
)
from .loader import TLAQData
from .risk import rolling_portfolio_risk


TRADING_DAYS = 252


def _performance_summary(r: pd.Series) -> pd.Series:
    r = r.dropna()
    if r.empty:
        return pd.Series(dtype=float)

    equity = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 and equity.iloc[-1] > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS) if r.std(ddof=1) > 0 else np.nan
    dd = equity / equity.cummax() - 1

    return pd.Series(
        {
            "CAGR": cagr,
            "Annualized Vol": vol,
            "Sharpe": sharpe,
            "Max Drawdown": dd.min(),
        }
    )


def _strategy_stress_stats(data: TLAQData) -> pd.Series:
    frame = pd.concat([data.strategy_returns, data.asset_returns["VTI"]], axis=1).dropna()
    frame.columns = ["TLAQ", "VTI"]
    if frame.empty:
        return pd.Series(dtype=float)

    q20 = frame["VTI"].quantile(0.20)
    q10 = frame["VTI"].quantile(0.10)
    q05 = frame["VTI"].quantile(0.05)
    regimes = {
        "All": pd.Series(True, index=frame.index),
        "VTI down": frame["VTI"] < 0,
        "VTI worst 20%": frame["VTI"] <= q20,
        "VTI worst 10%": frame["VTI"] <= q10,
        "VTI worst 5%": frame["VTI"] <= q05,
    }

    out = {}
    for name, mask in regimes.items():
        stats = regression_stats(frame.loc[mask, "TLAQ"], frame.loc[mask, "VTI"])
        out[f"Beta - {name}"] = stats.get("beta::VTI")
        out[f"R2 - {name}"] = stats.get("r2")
    return pd.Series(out)


def _state_stats(data: TLAQData) -> pd.DataFrame:
    rows = []
    for state in ["SVXY", "VIXY", "NONE"]:
        mask = data.held_vol_state == state
        frame = pd.concat(
            [data.strategy_returns.rename("TLAQ"), data.asset_returns["VTI"]], axis=1
        ).loc[mask].dropna()
        if frame.empty:
            continue
        reg = regression_stats(frame["TLAQ"], frame["VTI"])
        r = frame["TLAQ"]
        ann_mean = r.mean() * TRADING_DAYS
        ann_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
        sharpe = ann_mean / ann_vol if ann_vol > 0 else np.nan
        rows.append(
            {
                "state": state,
                "days": len(frame),
                "Annualized Mean Return": ann_mean,
                "Annualized Vol": ann_vol,
                "Sharpe": sharpe,
                "VTI beta": reg.get("beta::VTI"),
                "VTI R2": reg.get("r2"),
            }
        )
    return pd.DataFrame(rows).set_index("state") if rows else pd.DataFrame()


def _worst_days(data: TLAQData, n: int = 15) -> pd.DataFrame:
    daily = pd.concat(
        [
            data.strategy_returns.rename("TLAQ"),
            data.asset_returns["VTI"].rename("VTI_return"),
            data.return_contributions[["VTI", "TLT", "GLD", "SVXY", "VIXY", "FEES"]],
            data.held_vol_state,
            data.gross_exposure,
        ],
        axis=1,
    ).dropna(subset=["TLAQ"])
    return daily.nsmallest(n, "TLAQ")


def _dollar_attribution(data: TLAQData) -> pd.Series:
    cols = ["VTI", "TLT", "GLD", "SVXY", "VIXY", "FEES"]
    return data.pnl_by_asset[cols].sum().rename("dollar_pnl")



def _correlations_by_regime(data: TLAQData) -> pd.DataFrame:
    returns = data.active_returns.copy()
    vti = returns["VTI"].dropna()
    q20 = vti.quantile(0.20)
    q10 = vti.quantile(0.10)
    q05 = vti.quantile(0.05)
    masks = {
        "ALL": pd.Series(True, index=returns.index),
        "VTI_DOWN": returns["VTI"] < 0,
        "VTI_WORST_20": returns["VTI"] <= q20,
        "VTI_WORST_10": returns["VTI"] <= q10,
        "VTI_WORST_05": returns["VTI"] <= q05,
    }
    rows = []
    columns = list(returns.columns)
    for regime, mask in masks.items():
        corr = returns.loc[mask].corr()
        for i, a in enumerate(columns):
            for b in columns[i + 1 :]:
                rows.append(
                    {
                        "regime": regime,
                        "asset_a": a,
                        "asset_b": b,
                        "correlation": corr.loc[a, b],
                    }
                )
    return pd.DataFrame(rows).set_index(["regime", "asset_a", "asset_b"])


def _vol_state_stress_stats(data: TLAQData) -> pd.DataFrame:
    frame = pd.concat(
        [
            data.strategy_returns.rename("TLAQ"),
            data.asset_returns["VTI"].rename("VTI"),
            data.active_returns["VOL"].rename("VOL"),
            data.held_vol_state.rename("state"),
        ],
        axis=1,
    )
    vti_clean = frame["VTI"].dropna()
    q20 = vti_clean.quantile(0.20)
    q10 = vti_clean.quantile(0.10)
    q05 = vti_clean.quantile(0.05)
    regime_masks = {
        "ALL": pd.Series(True, index=frame.index),
        "VTI_DOWN": frame["VTI"] < 0,
        "VTI_WORST_20": frame["VTI"] <= q20,
        "VTI_WORST_10": frame["VTI"] <= q10,
        "VTI_WORST_05": frame["VTI"] <= q05,
    }

    rows = []
    for state in ["SVXY", "VIXY", "NONE"]:
        state_mask = frame["state"] == state
        for regime, regime_mask in regime_masks.items():
            sample = frame.loc[state_mask & regime_mask]
            tlaq_reg = regression_stats(sample["TLAQ"], sample["VTI"])
            vol_reg = regression_stats(sample["VOL"], sample["VTI"])
            rows.append(
                {
                    "state": state,
                    "regime": regime,
                    "n": int(sample[["TLAQ", "VTI"]].dropna().shape[0]),
                    "tlaq_beta_to_VTI": tlaq_reg.get("beta::VTI"),
                    "tlaq_r2": tlaq_reg.get("r2"),
                    "vol_beta_to_VTI": vol_reg.get("beta::VTI"),
                    "vol_r2": vol_reg.get("r2"),
                    "mean_tlaq_return": sample["TLAQ"].mean(),
                }
            )
    return pd.DataFrame(rows).set_index(["state", "regime"]).sort_index()


def build_tlaq_report(
    data: TLAQData,
    output_dir: str | Path,
    beta_window: int = 60,
    risk_window: int = 60,
) -> dict[str, object]:
    """Run the full TLAQ FactorStrip diagnostic suite and save CSV outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    performance = _performance_summary(data.strategy_returns)
    stress = _strategy_stress_stats(data)
    state_stats = _state_stats(data)
    worst_days = _worst_days(data)
    attribution = _dollar_attribution(data)

    # Asset dependence. Active VOL means whichever volatility ETF TLAQ held
    # over the return interval.
    dependence_returns = data.active_returns.copy()
    vti_betas = conditional_vti_betas(
        dependence_returns,
        benchmark="VTI",
        targets=["TLT", "GLD", "VOL"],
    )
    unique_risk = unique_risk_table(dependence_returns, stress_driver="VTI")
    correlations = _correlations_by_regime(data)
    vol_state_stress = _vol_state_stress_stats(data)

    tlaq_rolling_beta = rolling_beta(
        data.strategy_returns,
        data.asset_returns["VTI"],
        window=beta_window,
    )

    rolling_risk = rolling_portfolio_risk(
        active_returns=data.active_returns,
        weights=data.weights,
        vol_state=data.vol_state,
        window=risk_window,
    )

    latest_weights = data.weights.iloc[-1].rename("weight")
    latest_summary = pd.Series(
        {
            "Date": str(data.nav.index[-1].date()),
            "NAV": data.nav.iloc[-1],
            "Gross asset exposure": data.gross_exposure.iloc[-1],
            "Vol state": data.vol_state.iloc[-1],
            "60d TLAQ/VTI beta": tlaq_rolling_beta.iloc[-1] if len(tlaq_rolling_beta) else np.nan,
            "Total commissions": data.commissions.sum(),
            "Total interest": data.interest.sum(),
            "Total short borrow cost": data.short_borrow_cost.sum(),
        },
        dtype=object,
    )

    # Save machine-readable outputs.
    performance.to_csv(output_dir / "performance.csv")
    stress.to_csv(output_dir / "strategy_stress_beta.csv")
    state_stats.to_csv(output_dir / "vol_state_stats.csv")
    worst_days.to_csv(output_dir / "worst_days.csv")
    attribution.to_csv(output_dir / "dollar_attribution.csv")
    vti_betas.to_csv(output_dir / "asset_vti_betas.csv")
    unique_risk.to_csv(output_dir / "asset_unique_risk.csv")
    correlations.to_csv(output_dir / "correlations_by_regime.csv")
    vol_state_stress.to_csv(output_dir / "vol_state_stress.csv")
    tlaq_rolling_beta.to_csv(output_dir / "rolling_tlaq_vti_beta.csv")
    rolling_risk.to_csv(output_dir / "rolling_risk_contributions.csv")
    latest_weights.to_csv(output_dir / "latest_weights.csv")
    latest_summary.to_csv(output_dir / "latest_summary.csv")
    data.return_contributions.to_csv(output_dir / "daily_return_attribution.csv.gz", compression="gzip")
    data.weights.to_csv(output_dir / "historical_weights.csv.gz", compression="gzip")
    data.nav.to_csv(output_dir / "tlaq_nav.csv")
    data.strategy_returns.to_csv(output_dir / "tlaq_returns.csv")

    # Human-readable report.
    lines = [
        "# TLAQ FactorStrip Report",
        "",
        "FactorStrip is used here as a diagnostic layer. It does not alter TLAQ's trading rules.",
        "",
        "## Latest state",
        "",
        f"- Date: {latest_summary['Date']}",
        f"- NAV: ${float(latest_summary['NAV']):,.2f}",
        f"- Gross asset exposure: {float(latest_summary['Gross asset exposure']):.2f}x NAV",
        f"- Volatility state: {latest_summary['Vol state']}",
        f"- Rolling {beta_window}d TLAQ/VTI beta: {float(latest_summary['60d TLAQ/VTI beta']):.3f}",
        "",
        "## What to inspect",
        "",
        "1. `worst_days.csv` — exact sleeve return contributions on TLAQ's worst days.",
        "2. `strategy_stress_beta.csv` — how TLAQ's VTI beta changes as VTI gets worse.",
        "3. `asset_vti_betas.csv` — whether TLT, GLD, and the active VOL sleeve become more VTI-like in stress.",
        "4. `asset_unique_risk.csv` — 1-R² proxy for how independent each sleeve remains.",
        "5. `rolling_risk_contributions.csv` — which sleeves dominate forecast portfolio variance through time.",
        "6. `vol_state_stats.csv` — TLAQ behavior while the held vol sleeve is SVXY, VIXY, or NONE.",
        "7. `vol_state_stress.csv` — crash beta separately while SVXY vs VIXY is actually held.",
        "8. `correlations_by_regime.csv` — pairwise diversification changes as VTI enters worse tails.",
        "",
        "## Accounting convention",
        "",
        "The end-of-day position at t-1 earns the close-to-close price move into t. Commissions, interest, and short-borrow cost are included from the table. The loader reconciles reconstructed P&L to the table's NAV change.",
        "",
        "## Caveat",
        "",
        "`unique_variance_proxy = 1 - R²` is a diagnostic, not a full institutional specific-risk forecast. Conditional tail samples also contain fewer observations, so very small tail groups should be interpreted cautiously.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "performance": performance,
        "stress": stress,
        "state_stats": state_stats,
        "worst_days": worst_days,
        "attribution": attribution,
        "vti_betas": vti_betas,
        "unique_risk": unique_risk,
        "correlations": correlations,
        "vol_state_stress": vol_state_stress,
        "rolling_beta": tlaq_rolling_beta,
        "rolling_risk": rolling_risk,
        "latest_weights": latest_weights,
        "latest_summary": latest_summary,
    }
