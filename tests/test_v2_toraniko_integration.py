import numpy as np
import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("toraniko")

from factorstrip.v2.toraniko_engine import ToranikoMarketSizeValueEngine


def test_toraniko_primary_engine_runs_market_size_value_without_sector_history():
    dates = [pl.date(2020, 1, 2), pl.date(2020, 1, 3)]
    rows = []
    for di, dt_expr in enumerate(dates):
        # materialize Python dates through a tiny frame because pl.date is Expr
        dt = pl.select(dt_expr.alias("d"))["d"][0]
        for i in range(120):
            size = i + 1
            rows.append(
                {
                    "date": dt,
                    "ticker": f"S{i:03d}",
                    "asset_returns": 0.001 * di + 0.00001 * i + 0.0002 * np.sin(i),
                    "market_cap": float(1_000_000 + size * 50_000),
                    "book_price": float(0.2 + (i % 20) / 100),
                }
            )
    panel = pl.DataFrame(rows)
    result = ToranikoMarketSizeValueEngine().fit(panel)
    factors = result["factor_returns"]
    residuals = result["residuals"]

    assert {"market", "ALL", "size", "value", "date"}.issubset(factors.columns)
    assert "momentum" not in " ".join(factors.columns).lower()
    assert residuals.height == 240
    assert residuals["residual"].is_null().sum() == 0
    assert np.allclose(factors["ALL"].to_numpy(), 0.0, atol=1e-12)
