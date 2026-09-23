from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PumpConfig

REQUIRED_COLUMNS = {"timestamp", "symbol", "open", "close", "quote_volume"}


def validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")

    out = panel.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["symbol"] = out["symbol"].astype(str).str.upper()
    for col in ("open", "close", "quote_volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.sort_values(["timestamp", "symbol"]).drop_duplicates(
        ["timestamp", "symbol"], keep="last"
    )
    if (out["open"] <= 0).any() or (out["close"] <= 0).any():
        raise ValueError("open/close prices must be positive")
    return out


def _wide(panel: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        panel.pivot(index="timestamp", columns="symbol", values=column)
        .sort_index()
        .sort_index(axis=1)
    )


def build_return_inputs(panel: pd.DataFrame, cfg: PumpConfig):
    panel = validate_panel(panel)
    close = _wide(panel, "close")
    open_ = _wide(panel, "open")
    quote_volume = _wide(panel, "quote_volume")

    if cfg.btc_symbol not in close.columns:
        raise ValueError(f"{cfg.btc_symbol} is required as a common factor")
    if cfg.eth_symbol not in close.columns:
        raise ValueError(f"{cfg.eth_symbol} is required as a common factor")

    log_close = np.log(close)
    returns = log_close.diff()

    alt_cols = [
        c for c in returns.columns if c not in {cfg.btc_symbol, cfg.eth_symbol}
    ]
    alt = returns[alt_cols]
    alt_count = alt.notna().sum(axis=1)
    alt_median = alt.median(axis=1, skipna=True).where(alt_count >= cfg.min_alt_count)

    factors = pd.DataFrame(
        {
            "btc": returns[cfg.btc_symbol],
            "eth": returns[cfg.eth_symbol],
            "alt_median": alt_median,
        },
        index=returns.index,
    )
    return panel, close, open_, quote_volume, returns, factors


def rolling_factor_residuals(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    cfg: PumpConfig,
) -> pd.DataFrame:
    """One-step-ahead residuals from a rolling common-factor model.

    At hour t, coefficients use only observations through t-1.  Current-hour
    returns therefore never enter the coefficient estimate used to residualize
    that same hour.
    """

    idx = returns.index.intersection(factors.index)
    y = returns.reindex(idx)
    f = factors.reindex(idx)

    x = pd.DataFrame(index=idx)
    x["intercept"] = 1.0
    for col in factors.columns:
        x[col] = f[col]
    x_cols = list(x.columns)
    k = len(x_cols)

    # Common X'X.  It is valid for an asset only when that asset has a complete
    # return history over the fitted window; we enforce that below.
    sxx = np.full((len(idx), k, k), np.nan, dtype=float)
    for i, ci in enumerate(x_cols):
        for j, cj in enumerate(x_cols):
            sxx[:, i, j] = (
                (x[ci] * x[cj])
                .shift(1)
                .rolling(cfg.factor_window_hours, min_periods=cfg.min_factor_obs)
                .sum()
                .to_numpy()
            )

    sxy = np.full((len(idx), k, y.shape[1]), np.nan, dtype=float)
    for i, ci in enumerate(x_cols):
        prod = y.mul(x[ci], axis=0)
        sxy[:, i, :] = (
            prod.shift(1)
            .rolling(cfg.factor_window_hours, min_periods=cfg.min_factor_obs)
            .sum()
            .to_numpy()
        )

    factor_row_valid = f.notna().all(axis=1)
    factor_valid = factor_row_valid.shift(1).rolling(
        cfg.factor_window_hours, min_periods=cfg.min_factor_obs
    ).sum().to_numpy()
    joint_valid = (
        y.notna()
        .mul(factor_row_valid, axis=0)
        .shift(1)
        .rolling(cfg.factor_window_hours, min_periods=cfg.min_factor_obs)
        .sum()
        .to_numpy()
    )

    x_now = x.to_numpy(dtype=float)
    y_now = y.to_numpy(dtype=float)
    residual = np.full_like(y_now, np.nan, dtype=float)
    eye = np.eye(k)
    eye[0, 0] = 0.0  # do not penalize intercept

    for t in range(len(idx)):
        if not np.isfinite(x_now[t]).all():
            continue
        if not np.isfinite(sxx[t]).all():
            continue
        if factor_valid[t] < cfg.min_factor_obs:
            continue
        try:
            inv = np.linalg.inv(sxx[t] + cfg.ridge * eye)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(sxx[t] + cfg.ridge * eye)
        betas = inv @ sxy[t]
        pred = x_now[t] @ betas
        ok = (
            np.isfinite(y_now[t])
            & np.isfinite(pred)
            & (joint_valid[t] >= factor_valid[t])
        )
        residual[t, ok] = y_now[t, ok] - pred[ok]

    return pd.DataFrame(residual, index=idx, columns=y.columns)


def make_feature_panel(panel: pd.DataFrame, cfg: PumpConfig):
    panel, close, open_, quote_volume, returns, factors = build_return_inputs(panel, cfg)
    residual_hourly = rolling_factor_residuals(returns, factors, cfg)

    raw_24h = close.div(close.shift(cfg.pump_lookback_hours)).sub(1.0)
    residual_log_24h = residual_hourly.rolling(
        cfg.pump_lookback_hours, min_periods=cfg.pump_lookback_hours
    ).sum()
    residual_24h = np.expm1(residual_log_24h)
    qv_24h = quote_volume.rolling(
        cfg.pump_lookback_hours, min_periods=cfg.pump_lookback_hours
    ).sum()

    return {
        "panel": panel,
        "close": close,
        "open": open_,
        "quote_volume": quote_volume,
        "returns": returns,
        "factors": factors,
        "residual_hourly": residual_hourly,
        "raw_24h": raw_24h,
        "residual_24h": residual_24h,
        "quote_volume_24h": qv_24h,
    }
