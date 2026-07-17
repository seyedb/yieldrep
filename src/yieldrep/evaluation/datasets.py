from __future__ import annotations

from pathlib import Path

import pandas as pd

from yieldrep.config import ProjectConfig


def build_modeling_datasets(config: ProjectConfig) -> list[Path]:
    """Join baseline representations to forward yield-change targets."""
    targets = pd.read_parquet(config.targets_path)
    curves = pd.read_parquet(config.curves_path)
    config.modeling_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    pca_targets = _join_pca_targets(config, targets)
    if not pca_targets.empty:
        pca_path = config.modeling_dir / "pca_targets.parquet"
        pca_targets.to_parquet(pca_path, index=False)
        output_paths.append(pca_path)

    nelson_siegel_targets = _join_nelson_siegel_targets(config, targets)
    if not nelson_siegel_targets.empty:
        ns_path = config.modeling_dir / "nelson_siegel_targets.parquet"
        nelson_siegel_targets.to_parquet(ns_path, index=False)
        output_paths.append(ns_path)

    lagged_targets = _join_lagged_targets(curves, targets, config.evaluation.lag_days)
    if not lagged_targets.empty:
        lagged_path = config.modeling_dir / "lagged_targets.parquet"
        lagged_targets.to_parquet(lagged_path, index=False)
        output_paths.append(lagged_path)

    return output_paths


def _join_pca_targets(config: ProjectConfig, targets: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for scores_path in sorted(config.pca_dir.glob("*_scores.parquet")):
        country = scores_path.name.removesuffix("_scores.parquet").upper()
        scores = pd.read_parquet(scores_path)
        scores["country"] = country
        frames.append(scores)
    if not frames:
        return pd.DataFrame()

    features = pd.concat(frames, ignore_index=True)
    return targets.merge(features, on=["date", "country"], how="inner")


def _join_nelson_siegel_targets(config: ProjectConfig, targets: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for factors_path in sorted(config.nelson_siegel_dir.glob("*_factors.parquet")):
        frames.append(pd.read_parquet(factors_path))
    if not frames:
        return pd.DataFrame()

    features = pd.concat(frames, ignore_index=True)
    return targets.merge(features, on=["date", "country"], how="inner")


def _join_lagged_targets(
    curves: pd.DataFrame,
    targets: pd.DataFrame,
    lag_days: list[int],
) -> pd.DataFrame:
    features = make_lagged_yield_change_features(curves, lag_days=lag_days)
    return targets.merge(features, on=["date", "country", "maturity_years"], how="inner")


def make_lagged_yield_change_features(
    curves: pd.DataFrame,
    lag_days: list[int],
) -> pd.DataFrame:
    """Create lagged yield-change features by country and maturity."""
    if not lag_days:
        raise ValueError("At least one lag is required")
    if any(lag <= 0 for lag in lag_days):
        raise ValueError("Lag days must be positive")

    features = curves.loc[:, ["date", "country", "maturity_years", "yield"]].copy()
    features["date"] = pd.to_datetime(features["date"])
    features = features.sort_values(["country", "maturity_years", "date"]).reset_index(drop=True)
    grouped = features.groupby(["country", "maturity_years"], sort=False)["yield"]

    lag_columns: list[str] = []
    for lag in lag_days:
        column = f"lag_{lag}_change"
        features[column] = features["yield"] - grouped.shift(lag)
        lag_columns.append(column)

    return features.dropna(subset=lag_columns).loc[
        :,
        ["date", "country", "maturity_years", *lag_columns],
    ]
