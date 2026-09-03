import numpy as np

from factorstrip.v2.exposures import BetaConfig, rolling_beta_arrays


def test_rolling_beta_is_lagged_and_shrunk():
    market = np.array([0.01, -0.02, 0.03, 0.02, -0.01, 0.04])
    asset = 2.0 * market
    cfg = BetaConfig(window_days=4, min_periods=3, shrinkage_to_prior=0.5, prior_beta=1.0)
    raw, shrunk = rolling_beta_arrays(asset, market, cfg)

    assert np.isnan(raw[2])
    assert np.isclose(raw[3], 2.0)
    assert np.isclose(shrunk[3], 1.5)

    # A huge contemporaneous asset return must not affect today's beta.
    shocked = asset.copy()
    shocked[3] = 100.0
    shocked_raw, _ = rolling_beta_arrays(shocked, market, cfg)
    assert np.isclose(shocked_raw[3], raw[3])
