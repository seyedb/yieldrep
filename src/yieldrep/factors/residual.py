from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from yieldrep.config import ProjectConfig

RESIDUAL_FEATURE_COLUMNS = [
    "residual",
    "residual_z_60",
    "residual_z_252",
    "residual_change_1",
    "residual_change_5",
    "residual_vol_20",
    "residual_local_slope",
    "residual_local_curvature",
    "residual_butterfly",
]


def build_residual_features(config: ProjectConfig) -> Path:
    """Build dynamic features from Nelson-Siegel fitted residuals."""
    fitted = _read_nelson_siegel_fitted(config)
    features = make_residual_features(fitted)

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(config.residual_features_path, index=False)
    return config.residual_features_path


def make_residual_features(fitted_curves: pd.DataFrame) -> pd.DataFrame:
    """Create dynamic and local cross-sectional Nelson-Siegel residual features."""
    features = fitted_curves.loc[:, ["date", "country", "maturity_years", "residual"]].copy()
    features["date"] = pd.to_datetime(features["date"])
    features = features.sort_values(["country", "maturity_years", "date"]).reset_index(drop=True)
    grouped = features.groupby(["country", "maturity_years"], sort=False)["residual"]

    features["residual_change_1"] = grouped.diff(1)
    features["residual_change_5"] = grouped.diff(5)
    features["residual_vol_20"] = grouped.transform(lambda series: series.diff().rolling(20).std())
    features["residual_z_60"] = grouped.transform(lambda series: _rolling_z_score(series, 60))
    features["residual_z_252"] = grouped.transform(lambda series: _rolling_z_score(series, 252))
    features = _add_local_residual_shape_features(features)

    features = features.sort_values(["country", "maturity_years", "date"]).reset_index(drop=True)

    return features.dropna(subset=RESIDUAL_FEATURE_COLUMNS).loc[
        :,
        ["date", "country", "maturity_years", *RESIDUAL_FEATURE_COLUMNS],
    ]


def _add_local_residual_shape_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add local slope, curvature, and butterfly-style residual dislocation features."""
    frames = []
    for _, group in features.groupby(["country", "date"], sort=False):
        local = group.sort_values("maturity_years").copy()
        maturities = local["maturity_years"].to_numpy(dtype=float)
        residuals = local["residual"].to_numpy(dtype=float)

        if len(local) >= 2:
            slope = _safe_gradient(residuals, maturities)
        else:
            slope = np.zeros(len(local), dtype=float)

        if len(local) >= 3:
            curvature = _safe_gradient(slope, maturities)
            previous_residual = pd.Series(residuals).shift(1).to_numpy(dtype=float)
            next_residual = pd.Series(residuals).shift(-1).to_numpy(dtype=float)
            previous_maturity = pd.Series(maturities).shift(1).to_numpy(dtype=float)
            next_maturity = pd.Series(maturities).shift(-1).to_numpy(dtype=float)
            maturity_span = next_maturity - previous_maturity
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = (maturities - previous_maturity) / maturity_span
                interpolated_residual = previous_residual + weight * (
                    next_residual - previous_residual
                )
            butterfly = np.where(
                np.isfinite(interpolated_residual),
                residuals - interpolated_residual,
                0.0,
            )
        else:
            curvature = np.zeros(len(local), dtype=float)
            butterfly = np.zeros(len(local), dtype=float)

        local["residual_local_slope"] = slope
        local["residual_local_curvature"] = curvature
        local["residual_butterfly"] = butterfly
        frames.append(local)

    if not frames:
        return features.assign(
            residual_local_slope=pd.Series(dtype=float),
            residual_local_curvature=pd.Series(dtype=float),
            residual_butterfly=pd.Series(dtype=float),
        )
    return pd.concat(frames, ignore_index=True)


def _safe_gradient(values: NDArray[np.float64], grid: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(values) < 2 or len(set(grid)) < 2:
        return np.zeros(len(values), dtype=float)
    return np.asarray(np.gradient(values, grid), dtype=float)


def _rolling_z_score(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def _read_nelson_siegel_fitted(config: ProjectConfig) -> pd.DataFrame:
    frames = [
        pd.read_parquet(fitted_path)
        for fitted_path in sorted(config.nelson_siegel_dir.glob("*_fitted.parquet"))
    ]
    if not frames:
        raise FileNotFoundError(
            f"No Nelson-Siegel fitted curve files found in {config.nelson_siegel_dir}"
        )
    return pd.concat(frames, ignore_index=True)
