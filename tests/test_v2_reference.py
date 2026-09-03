import numpy as np
import pandas as pd

from factorstrip.v2.reference import BlitzReferenceConfig, blitz_residual_momentum_reference


def _fixture():
    idx = pd.date_range("2018-01-31", periods=50, freq="ME")
    rng = np.random.default_rng(123)
    factors = pd.DataFrame(rng.normal(0, 0.02, size=(50, 3)), index=idx, columns=["MKT", "SMB", "HML"])
    eps = rng.normal(0, 0.01, size=(50, 2))
    stocks = pd.DataFrame(
        np.column_stack([
            0.8 * factors["MKT"] + 0.2 * factors["SMB"] + eps[:, 0],
            1.2 * factors["MKT"] - 0.3 * factors["HML"] + eps[:, 1],
        ]),
        index=idx,
        columns=["A", "B"],
    )
    return stocks, factors


def test_reference_signal_does_not_use_signal_month_or_future_return():
    stocks, factors = _fixture()
    cfg = BlitzReferenceConfig(estimation_months=36, formation_months=11, skip_months=1, min_estimation_months=24)
    base = blitz_residual_momentum_reference(stocks, factors, cfg)

    target_i = 40
    shocked = stocks.copy()
    shocked.iloc[target_i, 0] += 50.0
    shocked.iloc[target_i + 1 :, 0] -= 20.0
    changed = blitz_residual_momentum_reference(shocked, factors, cfg)

    assert np.isclose(base.iloc[target_i, 0], changed.iloc[target_i, 0], equal_nan=True)


def test_reference_intercept_does_not_force_signal_to_zero():
    stocks, factors = _fixture()
    signal = blitz_residual_momentum_reference(stocks, factors)
    finite = signal.to_numpy()[np.isfinite(signal.to_numpy())]
    assert finite.size > 0
    assert not np.allclose(finite, 0.0)
