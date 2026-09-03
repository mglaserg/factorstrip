from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .schema import COL


@dataclass(frozen=True)
class FactorEngineConfig:
    beta_column: str = "market_beta"
    sector_column: str | None = "sector"
    style_columns: tuple[str, ...] = ()
    weight_column: str | None = None
    min_assets: int = 30
    winsor_fraction: float = 0.01

    def validate(self) -> None:
        if self.min_assets < 3:
            raise ValueError("min_assets must be >= 3")
        if not 0 <= self.winsor_fraction < 0.5:
            raise ValueError("winsor_fraction must be in [0, 0.5)")


@dataclass(frozen=True)
class WlsSolution:
    coefficients: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    r2: float
    max_abs_weighted_orthogonality: float


def _winsorize(y: np.ndarray, fraction: float) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    finite = np.isfinite(y)
    if fraction <= 0 or finite.sum() < 3:
        return y
    lo, hi = np.quantile(y[finite], [fraction, 1.0 - fraction])
    y[finite] = np.clip(y[finite], lo, hi)
    return y


def solve_wls(y: np.ndarray, x: np.ndarray, weights: np.ndarray | None = None) -> WlsSolution:
    """Solve one cross-sectional WLS and expose the orthogonality diagnostic."""

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(y) != x.shape[0]:
        raise ValueError("incompatible y/x shapes")
    if weights is None:
        w = np.ones(len(y), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != y.shape:
            raise ValueError("weights must match y")
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("weights must be finite and strictly positive")

    root_w = np.sqrt(w)
    xw = x * root_w[:, None]
    yw = y * root_w
    coef, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    fitted = x @ coef
    resid = y - fitted

    ybar = np.average(y, weights=w)
    sse = float(np.sum(w * resid**2))
    sst = float(np.sum(w * (y - ybar) ** 2))
    r2 = np.nan if sst <= 0 else 1.0 - sse / sst

    # WLS first-order condition: X' W epsilon = 0.
    orth = x.T @ (w * resid)
    scale = max(float(np.sum(w)), 1.0)
    max_abs_orth = float(np.max(np.abs(orth)) / scale) if orth.size else 0.0
    return WlsSolution(coef, fitted, resid, r2, max_abs_orth)


class CrossSectionalFactorEngine:
    """Authoritative V2 daily factor/residual engine.

    Input/output is Polars; the small linear-algebra kernel is NumPy.  Every day
    is a cross-sectional WLS on lagged exposures.  Residuals are therefore
    orthogonal to the modeled exposure span by construction (under the chosen
    weights), unlike a simple time-series beta subtraction.
    """

    def __init__(self, config: FactorEngineConfig | None = None):
        self.config = config or FactorEngineConfig()
        self.config.validate()

    def fit(self, returns: Any, exposures: Any):
        try:
            import polars as pl
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("FactorStrip V2 factor engine requires Polars") from exc

        cfg = self.config
        req_returns = {COL.date, COL.asset_id, COL.total_return}
        req_exposures = {COL.date, COL.asset_id, cfg.beta_column, *cfg.style_columns}
        if cfg.sector_column is not None:
            req_exposures.add(cfg.sector_column)
        if cfg.weight_column:
            req_exposures.add(cfg.weight_column)
        miss_r = req_returns - set(returns.columns)
        miss_x = req_exposures - set(exposures.columns)
        if miss_r:
            raise ValueError(f"returns missing columns: {sorted(miss_r)}")
        if miss_x:
            raise ValueError(f"exposures missing columns: {sorted(miss_x)}")

        data = returns.select(COL.date, COL.asset_id, COL.total_return).join(
            exposures.select(*sorted(req_exposures)), on=[COL.date, COL.asset_id], how="inner"
        ).sort([COL.date, COL.asset_id])

        factor_rows: list[dict[str, object]] = []
        residual_rows: list[dict[str, object]] = []
        fitted_rows: list[dict[str, object]] = []
        diagnostic_rows: list[dict[str, object]] = []

        for day in data.partition_by(COL.date, maintain_order=True):
            date = day[COL.date][0]
            sectors = (
                day[cfg.sector_column].to_list()
                if cfg.sector_column is not None
                else ["ALL"] * day.height
            )
            beta = np.asarray(day[cfg.beta_column].to_numpy(), dtype=float)
            y = np.asarray(day[COL.total_return].to_numpy(), dtype=float)
            styles = [np.asarray(day[c].to_numpy(), dtype=float) for c in cfg.style_columns]
            weights = None if cfg.weight_column is None else np.asarray(day[cfg.weight_column].to_numpy(), dtype=float)

            valid = np.isfinite(y) & np.isfinite(beta)
            for arr in styles:
                valid &= np.isfinite(arr)
            if cfg.sector_column is not None:
                valid &= np.asarray([s is not None for s in sectors], dtype=bool)
            if weights is not None:
                valid &= np.isfinite(weights) & (weights > 0)

            if valid.sum() < cfg.min_assets:
                diagnostic_rows.append({"date": date, "n_obs": int(valid.sum()), "r2": None, "max_abs_weighted_orthogonality": None, "status": "INSUFFICIENT_CROSS_SECTION"})
                continue

            idx = np.flatnonzero(valid)
            yv = _winsorize(y[idx], cfg.winsor_fraction)
            beta_v = beta[idx]
            sectors_v = [str(sectors[i]) for i in idx]
            sector_names = sorted(set(sectors_v))
            sector_matrix = np.column_stack([
                np.asarray([1.0 if s == name else 0.0 for s in sectors_v])
                for name in sector_names
            ])
            columns = ["MARKET_BETA", *[f"SECTOR::{s}" for s in sector_names], *cfg.style_columns]
            x_parts: list[np.ndarray] = [beta_v[:, None], sector_matrix]
            for arr in styles:
                x_parts.append(arr[idx][:, None])
            x = np.column_stack(x_parts)
            wv = None if weights is None else weights[idx]
            sol = solve_wls(yv, x, wv)

            for name, value in zip(columns, sol.coefficients, strict=True):
                factor_rows.append({"date": date, "factor": name, "factor_return": float(value)})
            asset_ids = day[COL.asset_id].to_list()
            for local_i, row_i in enumerate(idx):
                aid = asset_ids[row_i]
                residual_rows.append({"date": date, "asset_id": aid, "residual": float(sol.residuals[local_i])})
                fitted_rows.append({"date": date, "asset_id": aid, "fitted_return": float(sol.fitted[local_i])})
            diagnostic_rows.append({
                "date": date,
                "n_obs": int(len(idx)),
                "r2": None if not np.isfinite(sol.r2) else float(sol.r2),
                "max_abs_weighted_orthogonality": float(sol.max_abs_weighted_orthogonality),
                "status": "OK",
            })

        return {
            "factor_returns": pl.DataFrame(factor_rows) if factor_rows else pl.DataFrame(),
            "residuals": pl.DataFrame(residual_rows) if residual_rows else pl.DataFrame(),
            "fitted": pl.DataFrame(fitted_rows) if fitted_rows else pl.DataFrame(),
            "diagnostics": pl.DataFrame(diagnostic_rows) if diagnostic_rows else pl.DataFrame(),
        }
