import numpy as np
import pandas as pd

from factorstrip.model import HierarchicalRiskModel


def test_hierarchy_reconstructs_returns():
    tickers = ["A", "B", "C", "D"]

    sector = pd.Series(
        {
            "A": "Tech",
            "B": "Tech",
            "C": "Health",
            "D": "Health",
        }
    )

    industry = pd.Series(
        {
            "A": "Semi",
            "B": "Semi",
            "C": "Biotech",
            "D": "Biotech",
        }
    )

    r = pd.DataFrame(
        {
            "A": [0.04],
            "B": [0.02],
            "C": [-0.01],
            "D": [-0.03],
        },
        index=pd.to_datetime(["2026-01-02"]),
    )

    model = HierarchicalRiskModel(sector, industry)
    result = model.fit(r)

    reconstructed = result.fitted + result.residuals

    assert np.allclose(
        reconstructed.values,
        r.values,
        equal_nan=True,
    )


def test_industry_residuals_average_to_zero():
    tickers = ["A", "B", "C", "D"]

    sector = pd.Series(
        {
            "A": "Tech",
            "B": "Tech",
            "C": "Health",
            "D": "Health",
        }
    )

    industry = pd.Series(
        {
            "A": "Semi",
            "B": "Semi",
            "C": "Biotech",
            "D": "Biotech",
        }
    )

    r = pd.Series(
        {
            "A": 0.04,
            "B": 0.02,
            "C": -0.01,
            "D": -0.03,
        }
    )

    model = HierarchicalRiskModel(sector, industry)
    _, _, residuals, _ = model.fit_day(r)

    assert np.isclose(residuals[["A", "B"]].mean(), 0.0)
    assert np.isclose(residuals[["C", "D"]].mean(), 0.0)
