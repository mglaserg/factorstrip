import numpy as np

from factorstrip.v2.engine import solve_wls


def test_wls_residuals_are_orthogonal_to_exposures():
    x = np.array(
        [
            [0.8, 1.0, 0.0],
            [1.2, 1.0, 0.0],
            [0.7, 0.0, 1.0],
            [1.3, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.9, 0.0, 1.0],
        ]
    )
    true_f = np.array([0.02, 0.01, -0.005])
    noise = np.array([0.002, -0.003, 0.001, -0.001, 0.0015, -0.0005])
    y = x @ true_f + noise
    w = np.array([1.0, 2.0, 1.0, 1.5, 0.7, 1.2])

    result = solve_wls(y, x, w)
    assert result.max_abs_weighted_orthogonality < 1e-12
    assert np.max(np.abs(x.T @ (w * result.residuals))) < 1e-11


def test_wls_reconstructs_cross_section():
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([0.01, -0.02, 0.005])
    result = solve_wls(y, x)
    assert np.allclose(result.fitted + result.residuals, y)
