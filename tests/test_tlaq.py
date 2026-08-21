import numpy as np
import pandas as pd

from factorstrip.tlaq.dependency import regression_stats
from factorstrip.tlaq.risk import risk_contribution_from_cov


def test_regression_recovers_beta():
    x = pd.Series(np.arange(1.0, 21.0), name="VTI")
    y = 0.01 + 2.5 * x
    stats = regression_stats(y, x)
    assert np.isclose(stats["alpha"], 0.01)
    assert np.isclose(stats["beta::VTI"], 2.5)
    assert np.isclose(stats["r2"], 1.0)


def test_risk_contributions_sum_to_one():
    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    w = pd.Series({"A": 0.5, "B": 0.5})
    rc = risk_contribution_from_cov(w, cov)
    assert np.isclose(rc[["A", "B"]].sum(), 1.0)
    assert rc["PORTFOLIO_VOL"] > 0
