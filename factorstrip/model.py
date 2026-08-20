from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskModelResult:
    """
    Outputs from the hierarchical cross-sectional model.

    factor_returns:
        Date x factor table. Columns are:
            MARKET
            SECTOR::<sector name>
            INDUSTRY::<industry name>

        Sector factors are returns relative to MARKET.
        Industry factors are returns relative to their parent sector.

    fitted:
        Model-predicted stock returns.

    residuals:
        Stock-specific returns after market + sector + industry.

    r2:
        Cross-sectional R^2 for each date.
    """
    factor_returns: pd.DataFrame
    fitted: pd.DataFrame
    residuals: pd.DataFrame
    r2: pd.Series


class HierarchicalRiskModel:
    """
    Transparent nested equity risk model:

        stock return
          = market
          + sector relative to market
          + industry relative to sector
          + stock-specific residual

    For a stock like NVDA:

        NVDA
          = MARKET
          + Information Technology
          + Semiconductors
          + NVDA residual

    NVDA never receives Healthcare exposure because its classifications
    are explicit and nested.

    The model is cross-sectional: every date uses that day's universe
    to estimate the common returns.

    weights:
        Optional date x ticker DataFrame or ticker Series.
        If omitted, every stock receives equal weight.
    """

    def __init__(
        self,
        sector_map: pd.Series,
        industry_map: pd.Series,
    ):
        self.sector_map = sector_map.astype("string")
        self.industry_map = industry_map.astype("string")

        # Each industry should have exactly one parent sector.
        mapping = pd.DataFrame(
            {
                "sector": self.sector_map,
                "industry": self.industry_map,
            }
        ).dropna()

        parent_counts = (
            mapping.groupby("industry")["sector"].nunique()
        )

        bad = parent_counts[parent_counts > 1]
        if not bad.empty:
            raise ValueError(
                "Industry labels must be nested inside exactly one sector. "
                f"Non-nested industries: {bad.index.tolist()}"
            )

        self.industry_to_sector = (
            mapping.drop_duplicates("industry")
            .set_index("industry")["sector"]
            .to_dict()
        )

    @staticmethod
    def _weighted_mean(x: pd.Series, w: pd.Series) -> float:
        mask = x.notna() & w.notna() & (w > 0)
        if mask.sum() == 0:
            return np.nan
        xx = x[mask].astype(float)
        ww = w[mask].astype(float)
        return float(np.average(xx, weights=ww))

    @staticmethod
    def _weighted_r2(
        y: pd.Series,
        fitted: pd.Series,
        w: pd.Series,
    ) -> float:
        df = pd.concat(
            [y.rename("y"), fitted.rename("fitted"), w.rename("w")],
            axis=1,
        ).dropna()

        df = df[df["w"] > 0]
        if len(df) < 2:
            return np.nan

        ybar = np.average(df["y"], weights=df["w"])
        sse = np.sum(df["w"] * (df["y"] - df["fitted"]) ** 2)
        sst = np.sum(df["w"] * (df["y"] - ybar) ** 2)

        if sst <= 0:
            return np.nan

        return float(1.0 - sse / sst)

    def fit_day(
        self,
        returns: pd.Series,
        weights: pd.Series | None = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series, float]:
        """
        Fit one cross-section.

        Returns:
            factor_returns, fitted_returns, residuals, r2
        """
        df = pd.DataFrame(
            {
                "return": returns,
                "sector": self.sector_map,
                "industry": self.industry_map,
            }
        ).dropna()

        if weights is None:
            df["weight"] = 1.0
        else:
            df["weight"] = weights.reindex(df.index)
            df = df.dropna(subset=["weight"])
            df = df[df["weight"] > 0]

        if df.empty:
            return (
                pd.Series(dtype=float),
                pd.Series(dtype=float),
                pd.Series(dtype=float),
                np.nan,
            )

        # 1) MARKET: weighted cross-sectional mean return.
        market = self._weighted_mean(df["return"], df["weight"])

        # 2) SECTORS: weighted mean sector return minus market.
        sector_mean = (
            df.groupby("sector", observed=True)
            .apply(
                lambda g: self._weighted_mean(g["return"], g["weight"]),
                include_groups=False,
            )
        )
        sector_effect = sector_mean - market

        # 3) INDUSTRIES:
        # weighted mean industry return minus its parent sector mean.
        industry_mean = (
            df.groupby("industry", observed=True)
            .apply(
                lambda g: self._weighted_mean(g["return"], g["weight"]),
                include_groups=False,
            )
        )

        industry_effect = pd.Series(index=industry_mean.index, dtype=float)

        for industry, mean_return in industry_mean.items():
            parent = self.industry_to_sector[industry]
            industry_effect.loc[industry] = (
                mean_return - sector_mean.loc[parent]
            )

        # 4) Fitted return for every stock.
        # MARKET + SECTOR + INDUSTRY = industry cross-sectional mean.
        fitted = pd.Series(index=df.index, dtype=float)

        for ticker, row in df.iterrows():
            fitted.loc[ticker] = (
                market
                + sector_effect.loc[row["sector"]]
                + industry_effect.loc[row["industry"]]
            )

        residual = df["return"] - fitted

        factor_returns = pd.concat(
            [
                pd.Series({"MARKET": market}),
                sector_effect.rename(
                    lambda x: f"SECTOR::{x}"
                ),
                industry_effect.rename(
                    lambda x: f"INDUSTRY::{x}"
                ),
            ]
        )

        r2 = self._weighted_r2(
            df["return"],
            fitted,
            df["weight"],
        )

        return factor_returns, fitted, residual, r2

    def fit(
        self,
        returns: pd.DataFrame,
        weights: pd.Series | pd.DataFrame | None = None,
    ) -> RiskModelResult:
        """
        Fit every date in a date x ticker return matrix.
        """
        factor_rows = []
        fitted = pd.DataFrame(
            index=returns.index,
            columns=returns.columns,
            dtype=float,
        )
        residuals = fitted.copy()
        r2 = pd.Series(index=returns.index, dtype=float, name="r2")

        for date, row in returns.iterrows():
            if weights is None:
                day_weights = None
            elif isinstance(weights, pd.Series):
                day_weights = weights
            else:
                day_weights = weights.loc[date]

            factors, day_fitted, day_resid, day_r2 = self.fit_day(
                row,
                weights=day_weights,
            )

            factors.name = date
            factor_rows.append(factors)

            fitted.loc[date, day_fitted.index] = day_fitted
            residuals.loc[date, day_resid.index] = day_resid
            r2.loc[date] = day_r2

        factor_returns = pd.DataFrame(factor_rows)
        factor_returns.index = returns.index

        return RiskModelResult(
            factor_returns=factor_returns,
            fitted=fitted,
            residuals=residuals,
            r2=r2,
        )
