from __future__ import annotations

import numpy as np
import pandas as pd


def regression_stats(y: pd.Series, X: pd.Series | pd.DataFrame) -> pd.Series:
    """OLS with an intercept, implemented with NumPy for transparency."""
    if isinstance(X, pd.Series):
        X = X.to_frame()

    data = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    if len(data) < max(5, X.shape[1] + 2):
        out = {"alpha": np.nan, "r2": np.nan, "n": len(data)}
        out.update({f"beta::{c}": np.nan for c in X.columns})
        return pd.Series(out, dtype=float)

    yv = data["__y__"].to_numpy(dtype=float)
    xv = data[X.columns].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(data)), xv])

    coef, *_ = np.linalg.lstsq(design, yv, rcond=None)
    fitted = design @ coef
    resid = yv - fitted

    sse = float(np.sum(resid**2))
    sst = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan

    out = {"alpha": float(coef[0]), "r2": r2, "n": float(len(data))}
    out.update({f"beta::{c}": float(b) for c, b in zip(X.columns, coef[1:])})
    return pd.Series(out)


def rolling_beta(
    y: pd.Series,
    x: pd.Series,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling single-factor beta = rolling covariance / rolling variance."""
    if min_periods is None:
        min_periods = max(20, window // 2)
    cov = y.rolling(window, min_periods=min_periods).cov(x)
    var = x.rolling(window, min_periods=min_periods).var()
    return (cov / var).rename("rolling_beta")


def stress_masks(vti_returns: pd.Series) -> dict[str, pd.Series]:
    clean = vti_returns.dropna()
    q20 = clean.quantile(0.20)
    q10 = clean.quantile(0.10)
    q05 = clean.quantile(0.05)

    idx = vti_returns.index
    return {
        "ALL": pd.Series(True, index=idx),
        "VTI_UP": vti_returns > 0,
        "VTI_DOWN": vti_returns < 0,
        "VTI_WORST_20": vti_returns <= q20,
        "VTI_WORST_10": vti_returns <= q10,
        "VTI_WORST_05": vti_returns <= q05,
    }


def conditional_vti_betas(
    returns: pd.DataFrame,
    benchmark: str = "VTI",
    targets: list[str] | None = None,
) -> pd.DataFrame:
    """Beta/R² of each target versus VTI under multiple VTI regimes."""
    if benchmark not in returns:
        raise ValueError(f"Benchmark {benchmark!r} is not present.")
    if targets is None:
        targets = [c for c in returns.columns if c != benchmark]

    masks = stress_masks(returns[benchmark])
    rows = []

    for target in targets:
        for regime, mask in masks.items():
            stats = regression_stats(returns.loc[mask, target], returns.loc[mask, benchmark])
            rows.append(
                {
                    "asset": target,
                    "regime": regime,
                    "beta_to_VTI": stats.get(f"beta::{benchmark}"),
                    "r2": stats.get("r2"),
                    "unique_variance_proxy": 1.0 - stats.get("r2") if pd.notna(stats.get("r2")) else np.nan,
                    "n": int(stats.get("n", 0)),
                }
            )

    return pd.DataFrame(rows).set_index(["asset", "regime"]).sort_index()


def unique_risk_table(
    returns: pd.DataFrame,
    stress_driver: str = "VTI",
) -> pd.DataFrame:
    """
    Explain every asset using all of the other assets.

    1 - R² is reported as a simple 'unique variance proxy'. It is not a full
    specific-risk model, but it directly answers: how much of this asset's
    variation is not linearly explained by the rest of the TLAQ complex?
    """
    if stress_driver not in returns:
        raise ValueError(f"Stress driver {stress_driver!r} is not present.")

    masks = stress_masks(returns[stress_driver])
    rows = []

    for target in returns.columns:
        predictors = [c for c in returns.columns if c != target]
        for regime, mask in masks.items():
            stats = regression_stats(
                returns.loc[mask, target],
                returns.loc[mask, predictors],
            )
            r2 = stats.get("r2")
            rows.append(
                {
                    "asset": target,
                    "regime": regime,
                    "r2_explained_by_others": r2,
                    "unique_variance_proxy": 1.0 - r2 if pd.notna(r2) else np.nan,
                    "n": int(stats.get("n", 0)),
                }
            )

    return pd.DataFrame(rows).set_index(["asset", "regime"]).sort_index()
