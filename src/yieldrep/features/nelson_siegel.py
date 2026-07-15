"""Nelson-Siegel curve factors for classical term-structure baselines.

The Nelson-Siegel model represents a yield curve with three interpretable
loadings: level, slope, and curvature. Unlike PCA, the factor shapes are imposed
by the parametric form rather than learned from the covariance matrix. This makes
Nelson-Siegel a useful economic benchmark for later learned representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from yieldrep.config import ProjectConfig
from yieldrep.features.curve import curve_panel


@dataclass(frozen=True)
class NelsonSiegelFit:
    beta_level: float
    beta_slope: float
    beta_curvature: float
    tau: float
    fitted_yields: NDArray[np.float64]
    residuals: NDArray[np.float64]


def build_nelson_siegel(config: ProjectConfig) -> list[Path]:
    """Fit Nelson-Siegel factors by country/date and write parquet outputs."""
    curves = pd.read_parquet(config.curves_path)
    config.nelson_siegel_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for country in sorted(curves["country"].dropna().unique()):
        panel = curve_panel(curves, str(country)).ffill().dropna()
        if panel.shape[1] < config.nelson_siegel.min_maturities:
            continue

        output_paths.extend(_fit_country_nelson_siegel(config, str(country), panel))

    return output_paths


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


def _fit_country_nelson_siegel(
    config: ProjectConfig,
    country: str,
    panel: pd.DataFrame,
) -> list[Path]:
    tau = config.nelson_siegel.tau
    maturity_years = panel.columns.to_numpy(dtype=float)
    factor_rows: list[dict[str, object]] = []
    fitted_rows: list[dict[str, object]] = []

    for date, row in panel.iterrows():
        fit = fit_nelson_siegel(maturity_years, row.to_numpy(dtype=float), tau=tau)
        factor_rows.append(
            {
                "date": date,
                "country": country,
                "beta_level": fit.beta_level,
                "beta_slope": fit.beta_slope,
                "beta_curvature": fit.beta_curvature,
                "tau": fit.tau,
                "rmse": float(np.sqrt(np.mean(fit.residuals**2))),
            }
        )
        for maturity, fitted_yield, residual in zip(
            maturity_years,
            fit.fitted_yields,
            fit.residuals,
            strict=True,
        ):
            fitted_rows.append(
                {
                    "date": date,
                    "country": country,
                    "maturity_years": float(maturity),
                    "fitted_yield": float(fitted_yield),
                    "residual": float(residual),
                    "tau": fit.tau,
                }
            )

    country_key = country.lower()
    factors_path = config.nelson_siegel_dir / f"{country_key}_factors.parquet"
    fitted_path = config.nelson_siegel_dir / f"{country_key}_fitted.parquet"
    pd.DataFrame(factor_rows).to_parquet(factors_path, index=False)
    pd.DataFrame(fitted_rows).to_parquet(fitted_path, index=False)
    return [factors_path, fitted_path]


def _validate_maturities(maturity_years: NDArray[np.float64]) -> NDArray[np.float64]:
    maturities = np.asarray(maturity_years, dtype=float)
    if maturities.ndim != 1:
        raise ValueError("Maturities must be a one-dimensional array")
    if (maturities <= 0).any():
        raise ValueError("Maturities must be positive")
    return maturities
