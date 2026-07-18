from __future__ import annotations

from pathlib import Path

import pandas as pd

from yieldrep.config import ProjectConfig

RESIDUAL_FEATURE_COLUMNS = [
    "residual",
    "residual_z_60",
    "residual_z_252",
    "residual_change_1",
    "residual_change_5",
    "residual_vol_20",
]


def build_residual_features(config: ProjectConfig) -> Path:
    """Build dynamic features from Nelson-Siegel fitted residuals."""
    fitted = _read_nelson_siegel_fitted(config)
    features = make_residual_features(fitted)

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(config.residual_features_path, index=False)
    return config.residual_features_path


def make_residual_features(fitted_curves: pd.DataFrame) -> pd.DataFrame:
    """Create residual z-score, momentum, and volatility features."""
    features = fitted_curves.loc[:, ["date", "country", "maturity_years", "residual"]].copy()
    features["date"] = pd.to_datetime(features["date"])
    features = features.sort_values(["country", "maturity_years", "date"]).reset_index(drop=True)
    grouped = features.groupby(["country", "maturity_years"], sort=False)["residual"]

    features["residual_change_1"] = grouped.diff(1)
    features["residual_change_5"] = grouped.diff(5)
    features["residual_vol_20"] = grouped.transform(lambda series: series.diff().rolling(20).std())
    features["residual_z_60"] = grouped.transform(lambda series: _rolling_z_score(series, 60))
    features["residual_z_252"] = grouped.transform(lambda series: _rolling_z_score(series, 252))

    return features.dropna(subset=RESIDUAL_FEATURE_COLUMNS).loc[
        :,
        ["date", "country", "maturity_years", *RESIDUAL_FEATURE_COLUMNS],
    ]


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
