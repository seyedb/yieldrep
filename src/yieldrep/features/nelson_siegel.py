"""Nelson-Siegel curve factors for classical term-structure baselines.

The Nelson-Siegel model represents a yield curve with three interpretable
loadings: level, slope, and curvature. Unlike PCA, the factor shapes are imposed
by the parametric form rather than learned from the covariance matrix. This makes
Nelson-Siegel a useful economic benchmark for later learned representations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NelsonSiegelFit:
    beta_level: float
    beta_slope: float
    beta_curvature: float
    tau: float
    fitted_yields: NDArray[np.float64]
    residuals: NDArray[np.float64]


def nelson_siegel_loadings(
    maturity_years: NDArray[np.float64],
    tau: float,
) -> NDArray[np.float64]:
    """Return level, slope, and curvature loadings for maturities and fixed tau."""
    maturities = _validate_maturities(maturity_years)
    if tau <= 0:
        raise ValueError("Nelson-Siegel tau must be positive")

    scaled = maturities / tau
    slope = (1.0 - np.exp(-scaled)) / scaled
    curvature = slope - np.exp(-scaled)
    level = np.ones_like(maturities)
    return np.column_stack([level, slope, curvature])


def fit_nelson_siegel(
    maturity_years: NDArray[np.float64],
    yields: NDArray[np.float64],
    tau: float,
) -> NelsonSiegelFit:
    """Fit Nelson-Siegel betas by least squares for a fixed tau."""
    maturities = _validate_maturities(maturity_years)
    curve_yields = np.asarray(yields, dtype=float)
    if curve_yields.shape != maturities.shape:
        raise ValueError("Maturities and yields must have the same shape")

    loadings = nelson_siegel_loadings(maturities, tau)
    betas, *_ = np.linalg.lstsq(loadings, curve_yields, rcond=None)
    fitted = loadings @ betas
    residuals = curve_yields - fitted
    return NelsonSiegelFit(
        beta_level=float(betas[0]),
        beta_slope=float(betas[1]),
        beta_curvature=float(betas[2]),
        tau=tau,
        fitted_yields=fitted,
        residuals=residuals,
    )


def _validate_maturities(maturity_years: NDArray[np.float64]) -> NDArray[np.float64]:
    maturities = np.asarray(maturity_years, dtype=float)
    if maturities.ndim != 1:
        raise ValueError("Maturities must be a one-dimensional array")
    if (maturities <= 0).any():
        raise ValueError("Maturities must be positive")
    return maturities
