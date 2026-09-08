from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .styles import StyleConfig, build_market_size_value_inputs


@dataclass(frozen=True)
class ToranikoEngineConfig:
    winsor_factor: float = 0.01
    residualize_styles: bool = True
    style: StyleConfig = StyleConfig()

    def validate(self) -> None:
        if not 0 <= self.winsor_factor < 0.5:
            raise ValueError("winsor_factor must be in [0, 0.5)")
        self.style.validate()


class ToranikoMarketSizeValueEngine:
    """Primary FactorStrip V2 factor engine.

    Toraniko owns the daily constrained cross-sectional regression. FactorStrip
    owns the source/PIT rules, style definitions, blinding, and downstream
    residual-signal construction.  The custom NumPy engine remains an
    independent correctness reference rather than the production path.
    """

    def __init__(self, config: ToranikoEngineConfig | None = None):
        self.config = config or ToranikoEngineConfig()
        self.config.validate()

    def fit(self, panel: Any):
        try:
            import polars as pl
            from toraniko.model import estimate_factor_returns
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "FactorStrip V2's primary engine requires Toraniko. Install the `rw` extra."
            ) from exc

        returns_df, mkt_cap_df, sector_df, style_df = build_market_size_value_inputs(
            panel, self.config.style
        )
        factor_returns, eps_wide = estimate_factor_returns(
            returns_df,
            mkt_cap_df,
            sector_df,
            style_df,
            winsor_factor=self.config.winsor_factor,
            residualize_styles=self.config.residualize_styles,
        )
        residuals = (
            eps_wide.unpivot(
                index="date", variable_name="symbol", value_name="residual"
            )
            .drop_nulls("residual")
            .sort(["date", "symbol"])
        )
        return {
            "factor_returns": factor_returns.sort("date"),
            "residuals": residuals,
            "returns_input": returns_df,
            "market_cap_input": mkt_cap_df,
            "sector_input": sector_df,
            "style_input": style_df,
        }
