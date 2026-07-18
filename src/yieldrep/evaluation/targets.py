from __future__ import annotations

from pathlib import Path

import pandas as pd

from yieldrep.config import ProjectConfig

TARGET_COLUMNS = (
    "date",
    "country",
    "maturity_years",
    "horizon_days",
    "yield",
    "future_yield",
    "target_yield_change",
)
RESIDUAL_TARGET_COLUMNS = (
    "date",
    "country",
    "maturity_years",
    "horizon_days",
    "residual",
    "future_residual",
    "target_residual_change",
    "fitted_yield",
)


def build_targets(config: ProjectConfig) -> Path:
    """Build forward yield-change targets from normalized curves."""
    curves = pd.read_parquet(config.curves_path)
    targets = make_forward_yield_change_targets(curves, config.targets.horizons_days)

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(config.targets_path, index=False)
    return config.targets_path


def build_residual_targets(config: ProjectConfig) -> Path:
    """Build forward Nelson-Siegel residual-change targets."""
    fitted = _read_nelson_siegel_fitted(config)
    targets = make_forward_residual_change_targets(fitted, config.targets.horizons_days)

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(config.residual_targets_path, index=False)
    return config.residual_targets_path


def make_forward_yield_change_targets(
    curves: pd.DataFrame,
    horizons_days: list[int],
) -> pd.DataFrame:
    """Create yield(t+h) - yield(t) targets by country and maturity."""
    if not horizons_days:
        raise ValueError("At least one target horizon is required")
    if any(horizon <= 0 for horizon in horizons_days):
        raise ValueError("Target horizons must be positive")

    base = curves.loc[:, ["date", "country", "maturity_years", "yield"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    base = base.sort_values(["country", "maturity_years", "date"]).reset_index(drop=True)

    frames = [_make_horizon_targets(base, horizon) for horizon in horizons_days]
    return pd.concat(frames, ignore_index=True).loc[:, TARGET_COLUMNS]


def make_forward_residual_change_targets(
    fitted_curves: pd.DataFrame,
    horizons_days: list[int],
) -> pd.DataFrame:
    """Create residual(t+h) - residual(t) targets by country and maturity."""
    if not horizons_days:
        raise ValueError("At least one target horizon is required")
    if any(horizon <= 0 for horizon in horizons_days):
        raise ValueError("Target horizons must be positive")

    base = fitted_curves.loc[
        :,
        ["date", "country", "maturity_years", "residual", "fitted_yield"],
    ].copy()
    base["date"] = pd.to_datetime(base["date"])
    base = base.sort_values(["country", "maturity_years", "date"]).reset_index(drop=True)

    frames = [_make_horizon_residual_targets(base, horizon) for horizon in horizons_days]
    return pd.concat(frames, ignore_index=True).loc[:, RESIDUAL_TARGET_COLUMNS]


def _make_horizon_targets(curves: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    target = curves.copy()
    grouped = target.groupby(["country", "maturity_years"], sort=False)["yield"]
    target["future_yield"] = grouped.shift(-horizon_days)
    target["target_yield_change"] = target["future_yield"] - target["yield"]
    target["horizon_days"] = horizon_days
    return target.dropna(subset=["future_yield", "target_yield_change"])


def _make_horizon_residual_targets(
    fitted_curves: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    target = fitted_curves.copy()
    grouped = target.groupby(["country", "maturity_years"], sort=False)["residual"]
    target["future_residual"] = grouped.shift(-horizon_days)
    target["target_residual_change"] = target["future_residual"] - target["residual"]
    target["horizon_days"] = horizon_days
    return target.dropna(subset=["future_residual", "target_residual_change"])


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
