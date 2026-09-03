from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, sqrt
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GateConfig:
    """Frozen design choices for the FactorStrip V2 feasibility gate."""

    formation_months: int = 11
    skip_months: int = 1
    beta_window_days: int = 252
    beta_min_periods: int = 126
    beta_shrinkage: float = 0.50
    beta_prior: float = 1.0
    quantile: float = 0.20
    gross_exposure: float = 1.0
    cost_bps: float = 5.0
    target_vol: float = 0.10
    vol_window_months: int = 12
    max_vol_scalar: float = 3.0
    incremental_return_hurdle: float = 0.03
    desired_power: float = 0.80
    family_significance: float = 0.05
    registered_trials: int = 2
    bootstrap_samples: int = 2_000
    bootstrap_block_months: int = 6
    bootstrap_seed: int = 17_291

    def validate(self) -> None:
        if self.formation_months < 1:
            raise ValueError("formation_months must be >= 1")
        if self.skip_months < 1:
            raise ValueError("skip_months must be >= 1")
        if self.beta_window_days < 2:
            raise ValueError("beta_window_days must be >= 2")
        if not 2 <= self.beta_min_periods <= self.beta_window_days:
            raise ValueError("beta_min_periods must be between 2 and beta_window_days")
        if not 0.0 <= self.beta_shrinkage <= 1.0:
            raise ValueError("beta_shrinkage must be between 0 and 1")
        if not 0.0 < self.quantile < 0.5:
            raise ValueError("quantile must be between 0 and 0.5")
        if self.gross_exposure <= 0:
            raise ValueError("gross_exposure must be positive")
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")
        if self.target_vol <= 0:
            raise ValueError("target_vol must be positive")
        if self.vol_window_months < 2:
            raise ValueError("vol_window_months must be >= 2")
        if self.max_vol_scalar <= 0:
            raise ValueError("max_vol_scalar must be positive")
        if self.incremental_return_hurdle <= 0:
            raise ValueError("incremental_return_hurdle must be positive")
        if not 0.0 < self.desired_power < 1.0:
            raise ValueError("desired_power must be between 0 and 1")
        if not 0.0 < self.family_significance < 1.0:
            raise ValueError("family_significance must be between 0 and 1")
        if self.registered_trials < 1:
            raise ValueError("registered_trials must be >= 1")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be >= 100")
        if self.bootstrap_block_months < 1:
            raise ValueError("bootstrap_block_months must be >= 1")


@dataclass(frozen=True)
class GateSummary:
    """
    Deliberately blind output contract.

    This object contains feasibility/design measurements only. It has no
    alpha, Sharpe, CAGR, IC, drawdown, cumulative-PnL, or return-series field.
    """

    rho: float
    rho_ci_lower: float
    rho_ci_upper: float
    rho_for_power: float
    required_years: float
    required_months: int
    appraisal_ratio_at_hurdle: float
    effective_test_significance: float
    avg_monthly_turnover_raw: float
    avg_monthly_turnover_residual: float
    annualized_turnover_raw: float
    annualized_turnover_residual: float
    paired_months: int
    first_paired_month: str | None
    last_paired_month: str | None
    median_raw_signal_names: float
    median_residual_signal_names: float
    median_raw_portfolio_names: float
    median_residual_portfolio_names: float
    downloaded_names: int
    gate_status: str
    available_clean_years: float | None
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def monthly_compound_returns(daily_returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Compound daily simple returns into calendar-month returns."""
    return (1.0 + daily_returns).resample("ME").prod(min_count=1) - 1.0


def estimate_lagged_shrunk_market_beta(
    stock_returns: pd.DataFrame,
    market_returns: pd.Series,
    *,
    window: int = 252,
    min_periods: int = 126,
    shrinkage: float = 0.50,
    prior: float = 1.0,
) -> pd.DataFrame:
    """
    Estimate rolling market beta and lag it one trading day before use.

    shrinkage is the weight on the prior:
        beta_shrunk = (1-shrinkage) * beta_ols + shrinkage * prior

    The final shift(1) is intentional: beta used to residualize return t was
    estimated only from data available through t-1.
    """
    stock_returns, market_returns = stock_returns.align(
        market_returns.rename("MARKET"), join="inner", axis=0
    )

    beta = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns, dtype=float)

    for ticker in stock_returns.columns:
        stock = stock_returns[ticker]
        paired_market = market_returns.where(stock.notna())
        cov = stock.rolling(window, min_periods=min_periods).cov(paired_market)
        market_var = paired_market.rolling(window, min_periods=min_periods).var()
        beta[ticker] = cov / market_var

    beta = (1.0 - shrinkage) * beta + shrinkage * prior
    return beta.shift(1)


def beta_adjusted_residuals(
    stock_returns: pd.DataFrame,
    market_returns: pd.Series,
    lagged_beta: pd.DataFrame,
) -> pd.DataFrame:
    """Daily stock return minus lagged shrunk-beta times market return."""
    stock_returns, market_returns = stock_returns.align(
        market_returns.rename("MARKET"), join="inner", axis=0
    )
    lagged_beta = lagged_beta.reindex(index=stock_returns.index, columns=stock_returns.columns)
    return stock_returns - lagged_beta.mul(market_returns, axis=0)


def raw_12_1_signal(
    monthly_returns: pd.DataFrame,
    *,
    formation_months: int = 11,
    skip_months: int = 1,
) -> pd.DataFrame:
    """
    Conventional 12-1-style raw momentum signal.

    At month-end t, skip the most recent observed month(s) and compound the
    preceding formation_months. With formation_months=11 and skip_months=1,
    target weights formed at t are held at t+1, so the holding-month signal
    uses returns t-12 through t-2.
    """
    lagged = monthly_returns.shift(skip_months)
    gross = 1.0 + lagged
    return gross.rolling(formation_months, min_periods=formation_months).apply(
        np.prod, raw=True
    ) - 1.0


def standardized_residual_12_1_signal(
    monthly_residuals: pd.DataFrame,
    *,
    formation_months: int = 11,
    skip_months: int = 1,
) -> pd.DataFrame:
    """Residual momentum scaled by residual volatility over the same window."""
    lagged = monthly_residuals.shift(skip_months)
    rolling = lagged.rolling(formation_months, min_periods=formation_months)
    numerator = rolling.sum()
    denominator = rolling.std(ddof=1).replace(0.0, np.nan)
    return numerator / denominator


def global_long_short_weights(
    signal: pd.DataFrame,
    *,
    quantile: float = 0.20,
    gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """Equal-weight top/bottom-quantile, dollar-neutral target weights."""
    if not 0.0 < quantile < 0.5:
        raise ValueError("quantile must be between 0 and 0.5")

    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    for date, row in signal.iterrows():
        valid = row.dropna().sort_values()
        n = len(valid)
        k = min(int(np.floor(n * quantile)), n // 2)
        if k < 1:
            continue

        shorts = valid.index[:k]
        longs = valid.index[-k:]
        weights.loc[date, longs] = gross_exposure / 2.0 / k
        weights.loc[date, shorts] = -gross_exposure / 2.0 / k

    return weights


def beta_neutralize_weights(
    target_weights: pd.DataFrame,
    beta_exposure: pd.DataFrame,
    *,
    gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """
    Project selected-name weights onto zero-dollar / zero-beta constraints.

    The active set is unchanged. The projection is the minimum-L2 adjustment
    to the initial long/short weights subject to sum(w)=0 and sum(w*beta)=0.
    """
    beta_exposure = beta_exposure.reindex(
        index=target_weights.index,
        columns=target_weights.columns,
    )
    out = pd.DataFrame(0.0, index=target_weights.index, columns=target_weights.columns)

    for date in target_weights.index:
        w = target_weights.loc[date]
        b = beta_exposure.loc[date]
        active = (w != 0.0) & w.notna() & b.notna()
        if active.sum() < 3:
            continue

        wv = w.loc[active].to_numpy(dtype=float)
        bv = b.loc[active].to_numpy(dtype=float)
        if np.nanstd(bv) < 1e-10:
            continue

        constraints = np.vstack([np.ones_like(bv), bv])
        correction = constraints.T @ np.linalg.pinv(constraints @ constraints.T) @ (constraints @ wv)
        adjusted = wv - correction

        gross = np.abs(adjusted).sum()
        if gross <= 0:
            continue
        adjusted *= gross_exposure / gross
        out.loc[date, active] = adjusted

    return out


def _portfolio_path(
    target_weights: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    *,
    cost_bps: float,
) -> pd.DataFrame:
    """Monthly next-period portfolio path used only inside the blind gate."""
    target_weights, monthly_returns = target_weights.align(monthly_returns, join="inner", axis=0)
    target_weights, monthly_returns = target_weights.align(monthly_returns, join="inner", axis=1)

    positions = target_weights.shift(1).fillna(0.0)
    gross = (positions * monthly_returns.fillna(0.0)).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    turnover = turnover.fillna(positions.abs().sum(axis=1))
    net = gross - turnover * (cost_bps / 10_000.0)

    return pd.DataFrame({"net": net, "turnover": turnover})


def _apply_lagged_vol_target(
    target_weights: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    *,
    target_vol: float,
    window_months: int,
    max_scalar: float,
) -> pd.DataFrame:
    """Scale target weights using strategy volatility known at each signal date."""
    base = _portfolio_path(target_weights, monthly_returns, cost_bps=0.0)
    realized_vol = base["net"].rolling(window_months, min_periods=window_months).std(ddof=1) * sqrt(12.0)

    scalar = (target_vol / realized_vol).replace([np.inf, -np.inf], np.nan)
    scalar = scalar.clip(lower=0.0, upper=max_scalar).fillna(0.0)

    # base return at month t is generated by target_weights[t-1], so by the
    # end of month t it is known and may be used to size target_weights[t].
    return target_weights.mul(scalar, axis=0)


def circular_block_bootstrap_correlation_ci(
    x: pd.Series,
    y: pd.Series,
    *,
    samples: int = 2_000,
    block_months: int = 6,
    seed: int = 17_291,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Circular block-bootstrap confidence interval for paired correlation."""
    paired = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    n = len(paired)
    if n < max(12, block_months * 2):
        raise ValueError(
            f"Need at least {max(12, block_months * 2)} paired months for bootstrap; got {n}."
        )

    values = paired.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    blocks_needed = ceil(n / block_months)

    for _ in range(samples):
        idx_parts = []
        starts = rng.integers(0, n, size=blocks_needed)
        for start in starts:
            idx_parts.extend((start + np.arange(block_months)) % n)
        idx = np.asarray(idx_parts[:n], dtype=int)
        sample = values[idx]
        rho = np.corrcoef(sample[:, 0], sample[:, 1])[0, 1]
        if np.isfinite(rho):
            draws.append(float(rho))

    if len(draws) < samples * 0.90:
        raise RuntimeError("Too many invalid bootstrap correlations.")

    alpha = 1.0 - confidence
    lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def spanning_alpha_power_requirement(
    *,
    rho: float,
    incremental_return_hurdle: float = 0.03,
    target_vol: float = 0.10,
    desired_power: float = 0.80,
    family_significance: float = 0.05,
    registered_trials: int = 2,
) -> dict[str, float | int]:
    """
    Prospective sample requirement for the spanning-alpha hurdle.

    Uses a two-sided normal approximation and Bonferroni family-wise control
    for the registered 12-1 primary and 6-1 secondary specifications.
    """
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must be strictly between -1 and 1")

    effective_sig = family_significance / registered_trials
    residual_vol = target_vol * sqrt(1.0 - rho**2)
    appraisal_ratio = incremental_return_hurdle / residual_vol

    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - effective_sig / 2.0)
    z_power = normal.inv_cdf(desired_power)
    required_years = ((z_alpha + z_power) / appraisal_ratio) ** 2

    return {
        "appraisal_ratio": float(appraisal_ratio),
        "effective_test_significance": float(effective_sig),
        "required_years": float(required_years),
        "required_months": int(ceil(required_years * 12.0)),
    }


def run_blind_rho_gate(
    stock_daily_returns: pd.DataFrame,
    market_daily_returns: pd.Series,
    *,
    config: GateConfig | None = None,
    available_clean_years: float | None = None,
) -> GateSummary:
    """
    Run the blind FactorStrip V2 feasibility gate.

    Intentionally returns only correlation, turnover, coverage, and prospective
    power/design quantities. It does not expose strategy-return series or any
    performance statistic.
    """
    cfg = config or GateConfig()
    cfg.validate()

    stock_daily_returns = stock_daily_returns.sort_index().copy()
    market_daily_returns = market_daily_returns.sort_index().copy()
    stock_daily_returns, market_daily_returns = stock_daily_returns.align(
        market_daily_returns.rename("MARKET"), join="inner", axis=0
    )

    beta_daily = estimate_lagged_shrunk_market_beta(
        stock_daily_returns,
        market_daily_returns,
        window=cfg.beta_window_days,
        min_periods=cfg.beta_min_periods,
        shrinkage=cfg.beta_shrinkage,
        prior=cfg.beta_prior,
    )
    residual_daily = beta_adjusted_residuals(
        stock_daily_returns,
        market_daily_returns,
        beta_daily,
    )

    stock_monthly = monthly_compound_returns(stock_daily_returns)
    residual_monthly = residual_daily.resample("ME").sum(min_count=1)
    beta_monthly = beta_daily.resample("ME").last()

    raw_signal = raw_12_1_signal(
        stock_monthly,
        formation_months=cfg.formation_months,
        skip_months=cfg.skip_months,
    )
    residual_signal = standardized_residual_12_1_signal(
        residual_monthly,
        formation_months=cfg.formation_months,
        skip_months=cfg.skip_months,
    )

    raw_weights = global_long_short_weights(
        raw_signal,
        quantile=cfg.quantile,
        gross_exposure=cfg.gross_exposure,
    )
    residual_weights = global_long_short_weights(
        residual_signal,
        quantile=cfg.quantile,
        gross_exposure=cfg.gross_exposure,
    )

    raw_weights = beta_neutralize_weights(
        raw_weights,
        beta_monthly,
        gross_exposure=cfg.gross_exposure,
    )
    residual_weights = beta_neutralize_weights(
        residual_weights,
        beta_monthly,
        gross_exposure=cfg.gross_exposure,
    )

    raw_weights = _apply_lagged_vol_target(
        raw_weights,
        stock_monthly,
        target_vol=cfg.target_vol,
        window_months=cfg.vol_window_months,
        max_scalar=cfg.max_vol_scalar,
    )
    residual_weights = _apply_lagged_vol_target(
        residual_weights,
        stock_monthly,
        target_vol=cfg.target_vol,
        window_months=cfg.vol_window_months,
        max_scalar=cfg.max_vol_scalar,
    )

    raw_path = _portfolio_path(raw_weights, stock_monthly, cost_bps=cfg.cost_bps)
    residual_path = _portfolio_path(residual_weights, stock_monthly, cost_bps=cfg.cost_bps)
    paired = pd.concat(
        [raw_path["net"].rename("raw"), residual_path["net"].rename("residual")],
        axis=1,
    ).dropna()

    # Months before the strategy is active can be exactly zero in both arms.
    # Exclude them from the correlation measurement.
    active = (paired["raw"].abs() + paired["residual"].abs()) > 0
    paired = paired.loc[active]
    if len(paired) < max(12, cfg.bootstrap_block_months * 2):
        raise RuntimeError(
            "Insufficient paired active months after warm-up for the feasibility gate. "
            f"Got {len(paired)} months."
        )

    rho = float(paired["raw"].corr(paired["residual"]))
    if not np.isfinite(rho):
        raise RuntimeError("Paired strategy correlation is not finite.")

    ci_lower, ci_upper = circular_block_bootstrap_correlation_ci(
        paired["raw"],
        paired["residual"],
        samples=cfg.bootstrap_samples,
        block_months=cfg.bootstrap_block_months,
        seed=cfg.bootstrap_seed,
    )

    # Required sample size is largest when |rho| is smallest. For the expected
    # positive-correlation case this is exactly the lower CI bound. If the CI
    # crosses zero, use zero rather than a negative lower bound so the gate
    # remains conservative.
    if ci_lower <= 0.0 <= ci_upper:
        rho_for_power = 0.0
    else:
        rho_for_power = min((ci_lower, ci_upper), key=abs)
    rho_for_power = float(np.clip(rho_for_power, -0.999999, 0.999999))
    power = spanning_alpha_power_requirement(
        rho=rho_for_power,
        incremental_return_hurdle=cfg.incremental_return_hurdle,
        target_vol=cfg.target_vol,
        desired_power=cfg.desired_power,
        family_significance=cfg.family_significance,
        registered_trials=cfg.registered_trials,
    )

    if available_clean_years is None:
        gate_status = "MEASUREMENT_ONLY"
    elif available_clean_years >= float(power["required_years"]):
        gate_status = "GREENLIGHT"
    else:
        gate_status = "UNRESOLVABLE"

    raw_signal_names = raw_signal.notna().sum(axis=1)
    residual_signal_names = residual_signal.notna().sum(axis=1)
    raw_portfolio_names = (raw_weights != 0.0).sum(axis=1)
    residual_portfolio_names = (residual_weights != 0.0).sum(axis=1)

    return GateSummary(
        rho=rho,
        rho_ci_lower=ci_lower,
        rho_ci_upper=ci_upper,
        rho_for_power=rho_for_power,
        required_years=float(power["required_years"]),
        required_months=int(power["required_months"]),
        appraisal_ratio_at_hurdle=float(power["appraisal_ratio"]),
        effective_test_significance=float(power["effective_test_significance"]),
        avg_monthly_turnover_raw=float(raw_path.loc[paired.index, "turnover"].mean()),
        avg_monthly_turnover_residual=float(residual_path.loc[paired.index, "turnover"].mean()),
        annualized_turnover_raw=float(raw_path.loc[paired.index, "turnover"].mean() * 12.0),
        annualized_turnover_residual=float(residual_path.loc[paired.index, "turnover"].mean() * 12.0),
        paired_months=int(len(paired)),
        first_paired_month=paired.index.min().date().isoformat() if len(paired) else None,
        last_paired_month=paired.index.max().date().isoformat() if len(paired) else None,
        median_raw_signal_names=float(raw_signal_names.loc[paired.index].median()),
        median_residual_signal_names=float(residual_signal_names.loc[paired.index].median()),
        median_raw_portfolio_names=float(raw_portfolio_names.loc[paired.index].median()),
        median_residual_portfolio_names=float(residual_portfolio_names.loc[paired.index].median()),
        downloaded_names=int(stock_daily_returns.shape[1]),
        gate_status=gate_status,
        available_clean_years=available_clean_years,
        config=asdict(cfg),
    )
