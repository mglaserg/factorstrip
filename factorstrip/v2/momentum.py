from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MomentumSpec:
    name: str
    formation_months: int
    skip_months: int = 1
    min_daily_residual_obs: int = 180

    def validate(self) -> None:
        if self.formation_months < 2:
            raise ValueError("formation_months must be >= 2")
        if self.skip_months < 1:
            raise ValueError("skip_months must be >= 1")
        if self.min_daily_residual_obs < 20:
            raise ValueError("min_daily_residual_obs must be >= 20")


MOM_12_1 = MomentumSpec("12-1", formation_months=11, skip_months=1, min_daily_residual_obs=180)
MOM_6_1 = MomentumSpec("6-1", formation_months=5, skip_months=1, min_daily_residual_obs=80)


def _pl():
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("V2 momentum construction requires Polars") from exc
    return pl


def build_monthly_raw_and_residual_signals(
    returns_long: Any,
    residuals_long: Any,
    spec: MomentumSpec = MOM_12_1,
):
    """Construct blinded monthly raw/residual momentum scores.

    Signal month t uses formation months t-(formation+1) through t-2, skipping
    the immediately prior month.  Raw momentum compounds actual returns.
    Residual momentum sums daily factor residuals and divides by the pooled
    daily residual standard deviation over the same formation months.

    This function constructs signals only. It never joins forward returns or
    computes alpha, IC, Sharpe, or P&L.
    """

    pl = _pl()
    spec.validate()
    req_r = {"date", "symbol", "asset_returns"}
    req_e = {"date", "symbol", "residual"}
    if req_r - set(returns_long.columns):
        raise ValueError(f"returns missing columns: {sorted(req_r - set(returns_long.columns))}")
    if req_e - set(residuals_long.columns):
        raise ValueError(f"residuals missing columns: {sorted(req_e - set(residuals_long.columns))}")

    joined = (
        returns_long.join(residuals_long, on=["date", "symbol"], how="inner")
        .sort(["symbol", "date"])
        .with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
    )
    monthly = (
        joined.group_by(["symbol", "month"])
        .agg(
            pl.col("asset_returns").log1p().sum().alias("raw_log_sum"),
            pl.col("residual").sum().alias("resid_sum"),
            (pl.col("residual") ** 2).sum().alias("resid_sumsq"),
            pl.col("residual").len().alias("resid_n"),
        )
        .sort(["symbol", "month"])
        .with_columns(
            (pl.col("month").dt.year() * 12 + pl.col("month").dt.month()).alias("month_ord")
        )
    )

    f = spec.formation_months
    shift = spec.skip_months + 1
    # For 12-1 this is 11 formation months shifted 2 rows: t-12..t-2.
    rolled = monthly.with_columns(
        pl.col("raw_log_sum").rolling_sum(f).shift(shift).over("symbol").alias("raw_form_log"),
        pl.col("resid_sum").rolling_sum(f).shift(shift).over("symbol").alias("resid_form_sum"),
        pl.col("resid_sumsq").rolling_sum(f).shift(shift).over("symbol").alias("resid_form_sumsq"),
        pl.col("resid_n").rolling_sum(f).shift(shift).over("symbol").alias("resid_form_n"),
        pl.col("month_ord").shift(shift).over("symbol").alias("last_form_month_ord"),
        pl.col("month_ord").shift(shift + f - 1).over("symbol").alias("first_form_month_ord"),
    ).with_columns(
        (
            pl.col("last_form_month_ord") - pl.col("first_form_month_ord")
        ).alias("formation_span")
    )

    expected_span = f - 1
    var_num = (
        pl.col("resid_form_sumsq")
        - (pl.col("resid_form_sum") ** 2 / pl.col("resid_form_n"))
    )
    resid_var = var_num / (pl.col("resid_form_n") - 1)
    return (
        rolled.filter(
            (pl.col("formation_span") == expected_span)
            & (pl.col("resid_form_n") >= spec.min_daily_residual_obs)
            & (resid_var > 0)
        )
        .select(
            pl.col("month").alias("signal_month"),
            "symbol",
            pl.col("raw_form_log").exp().sub(1.0).alias("raw_momentum"),
            (pl.col("resid_form_sum") / resid_var.sqrt()).alias("residual_momentum"),
            pl.col("resid_form_n").alias("formation_daily_obs"),
        )
        .sort(["signal_month", "symbol"])
    )
