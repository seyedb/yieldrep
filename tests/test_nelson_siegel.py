from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yieldrep.config import NelsonSiegelConfig, ProjectConfig, SourceConfig
from yieldrep.features.nelson_siegel import (
    build_nelson_siegel,
    fit_nelson_siegel,
    nelson_siegel_loadings,
)


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


def test_build_nelson_siegel_writes_expected_outputs(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    _sample_curves(tau=1.5).to_parquet(processed_dir / "curves.parquet", index=False)
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
        nelson_siegel=NelsonSiegelConfig(tau=1.5, min_maturities=3),
    )

    output_paths = build_nelson_siegel(config)

    assert output_paths == [
        processed_dir / "nelson_siegel" / "us_factors.parquet",
        processed_dir / "nelson_siegel" / "us_fitted.parquet",
    ]
    factors = pd.read_parquet(processed_dir / "nelson_siegel" / "us_factors.parquet")
    fitted = pd.read_parquet(processed_dir / "nelson_siegel" / "us_fitted.parquet")

    assert factors.shape == (3, 7)
    assert fitted.shape == (12, 6)
    assert factors["country"].unique().tolist() == ["US"]
    assert np.allclose(factors["beta_level"], 4.0)
    assert np.allclose(factors["beta_slope"], -1.2)
    assert np.allclose(factors["beta_curvature"], 0.8)
    assert np.allclose(factors["rmse"], 0.0)


def _sample_curves(tau: float) -> pd.DataFrame:
    maturities = np.array([0.5, 1.0, 2.0, 5.0])
    yields = nelson_siegel_loadings(maturities, tau=tau) @ np.array([4.0, -1.2, 0.8])
    dates = pd.date_range("2024-01-01", periods=3)
    return pd.DataFrame(
        [
            {
                "date": date,
                "country": "US",
                "maturity_years": maturity,
                "yield": curve_yield,
                "source": "test",
            }
            for date in dates
            for maturity, curve_yield in zip(maturities, yields, strict=True)
        ]
    )
