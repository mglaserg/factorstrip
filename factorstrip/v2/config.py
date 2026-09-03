from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class UniverseConfig:
    """Mechanical point-in-time universe rules.

    All price/liquidity fields are lagged before eligibility is evaluated.  A
    vendor/index membership flag is deliberately absent: the V2 hypothesis is
    tested on a reproducible mechanical universe, not S&P committee selection.
    """

    price_floor: float = 5.0
    adv_window_days: int = 63
    adv_min_periods: int = 42
    min_dollar_volume: float = 5_000_000.0
    top_n_by_dollar_volume: int | None = 3000
    min_history_days: int = 252
    information_lag_days: int = 1
    require_common_stock: bool = True
    require_major_exchange: bool = True

    def validate(self) -> None:
        if self.price_floor <= 0:
            raise ValueError("price_floor must be positive")
        if self.adv_window_days < 2:
            raise ValueError("adv_window_days must be >= 2")
        if not 1 <= self.adv_min_periods <= self.adv_window_days:
            raise ValueError("adv_min_periods must be within adv_window_days")
        if self.min_dollar_volume < 0:
            raise ValueError("min_dollar_volume must be non-negative")
        if self.top_n_by_dollar_volume is not None and self.top_n_by_dollar_volume < 2:
            raise ValueError("top_n_by_dollar_volume must be >= 2 or None")
        if self.min_history_days < 1:
            raise ValueError("min_history_days must be >= 1")
        if self.information_lag_days < 1:
            raise ValueError("information_lag_days must be >= 1")


@dataclass(frozen=True)
class ResearchDesign:
    """Frozen V2 economic/statistical choices.

    This is a design object, not an evaluator.  Keeping the design in code makes
    accidental post-result drift obvious in version control.
    """

    primary_signal: str = "12-1 residual momentum"
    secondary_signal: str = "6-1 residual momentum"
    registered_trials: int = 2
    target_vol: float = 0.10
    incremental_alpha_hurdle: float = 0.03
    desired_power: float = 0.80
    family_significance: float = 0.05
    primary_statistic: str = "net spanning-regression alpha: residual | raw"
    mechanism_statistic: str = "paired cross-sectional IC difference"
    secondary_statistic: str = "delta Sharpe"
    case_c_incremental_risk_budget: float = 0.25
    transaction_cost_bps: float = 5.0

    def validate(self) -> None:
        if self.registered_trials != 2:
            raise ValueError("V2 design is frozen at two registered momentum trials")
        if self.target_vol <= 0:
            raise ValueError("target_vol must be positive")
        if self.incremental_alpha_hurdle <= 0:
            raise ValueError("incremental_alpha_hurdle must be positive")
        if not 0 < self.desired_power < 1:
            raise ValueError("desired_power must be between 0 and 1")
        if not 0 < self.family_significance < 1:
            raise ValueError("family_significance must be between 0 and 1")
        if not 0 < self.case_c_incremental_risk_budget < 1:
            raise ValueError("case_c_incremental_risk_budget must be between 0 and 1")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
