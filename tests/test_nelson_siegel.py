import numpy as np
import pytest

from yieldrep.features.nelson_siegel import fit_nelson_siegel, nelson_siegel_loadings


def test_nelson_siegel_loadings_shape_and_level_column() -> None:
    maturities = np.array([0.5, 1.0, 2.0, 5.0, 10.0])

    loadings = nelson_siegel_loadings(maturities, tau=1.5)

    assert loadings.shape == (5, 3)
    assert np.allclose(loadings[:, 0], 1.0)


def test_fit_nelson_siegel_recovers_known_betas() -> None:
    maturities = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
    tau = 1.7
    expected_betas = np.array([4.0, -1.2, 0.8])
    yields = nelson_siegel_loadings(maturities, tau=tau) @ expected_betas

    fit = fit_nelson_siegel(maturities, yields, tau=tau)

    assert fit.tau == tau
    assert np.allclose(
        [fit.beta_level, fit.beta_slope, fit.beta_curvature],
        expected_betas,
    )
    assert np.allclose(fit.fitted_yields, yields)
    assert np.allclose(fit.residuals, 0.0)


def test_nelson_siegel_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        nelson_siegel_loadings(np.array([0.0, 1.0]), tau=1.5)
    with pytest.raises(ValueError, match="tau"):
        nelson_siegel_loadings(np.array([1.0, 2.0]), tau=0.0)
    with pytest.raises(ValueError, match="same shape"):
        fit_nelson_siegel(np.array([1.0, 2.0]), np.array([4.0]), tau=1.5)
