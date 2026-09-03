from __future__ import annotations

from typing import Any

from .config import UniverseConfig
from .schema import BARS_REQUIRED, COL, SECURITY_HISTORY_REQUIRED


def _require(columns: list[str], required: set[str], label: str) -> None:
    missing = required - set(columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def build_mechanical_universe(bars: Any, security_history: Any, config: UniverseConfig | None = None):
    """Build the V2 point-in-time mechanical universe in Polars.

    Eligibility uses *lagged* price, dollar-volume, and history counts.  No
    index-membership field is consumed.  Delisted names remain naturally in the
    historical cross-section as long as the canonical source contains them.
    """

    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("FactorStrip V2 universe construction requires Polars") from exc

    config = config or UniverseConfig()
    config.validate()
    if not isinstance(bars, pl.DataFrame) or not isinstance(security_history, pl.DataFrame):
        raise TypeError("bars and security_history must be Polars DataFrames")
    _require(bars.columns, BARS_REQUIRED, "bars")
    _require(security_history.columns, SECURITY_HISTORY_REQUIRED, "security_history")

    lag = config.information_lag_days
    joined = (
        bars.sort([COL.asset_id, COL.date])
        .with_columns(
            (pl.col(COL.close_unadjusted) * pl.col(COL.volume_unadjusted)).alias("dollar_volume")
        )
        .with_columns(
            pl.col(COL.close_unadjusted).shift(lag).over(COL.asset_id).alias("price_lag"),
            pl.col("dollar_volume")
            .rolling_mean(window_size=config.adv_window_days, min_samples=config.adv_min_periods)
            .shift(lag)
            .over(COL.asset_id)
            .alias("adv_lag"),
            pl.col(COL.total_return)
            .is_not_null()
            .cast(pl.Int64)
            .cum_sum()
            .shift(lag)
            .over(COL.asset_id)
            .alias("history_obs_lag"),
        )
        .join(
            security_history.select(
                COL.date,
                COL.asset_id,
                COL.is_common_stock,
                COL.is_major_exchange,
            ),
            on=[COL.date, COL.asset_id],
            how="left",
        )
    )

    mask = (
        (pl.col("price_lag") >= config.price_floor)
        & (pl.col("adv_lag") >= config.min_dollar_volume)
        & (pl.col("history_obs_lag") >= config.min_history_days)
    )
    if config.require_common_stock:
        mask &= pl.col(COL.is_common_stock).fill_null(False)
    if config.require_major_exchange:
        mask &= pl.col(COL.is_major_exchange).fill_null(False)

    eligible = joined.filter(mask)

    if config.top_n_by_dollar_volume is not None:
        eligible = (
            eligible.with_columns(
                pl.col("adv_lag")
                .rank(method="ordinal", descending=True)
                .over(COL.date)
                .alias("liquidity_rank")
            )
            .filter(pl.col("liquidity_rank") <= config.top_n_by_dollar_volume)
        )
    else:
        eligible = eligible.with_columns(pl.lit(None, dtype=pl.UInt32).alias("liquidity_rank"))

    return eligible.select(
        COL.date,
        COL.asset_id,
        COL.symbol,
        COL.total_return,
        "price_lag",
        "adv_lag",
        "history_obs_lag",
        "liquidity_rank",
    ).sort([COL.date, COL.asset_id])
