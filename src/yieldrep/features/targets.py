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


def build_targets(config: ProjectConfig) -> Path:
    """Build forward yield-change targets from normalized curves."""
    curves = pd.read_parquet(config.curves_path)
    targets = make_forward_yield_change_targets(curves, config.targets.horizons_days)

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(config.targets_path, index=False)
    return config.targets_path


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


def _make_horizon_targets(curves: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    target = curves.copy()
    grouped = target.groupby(["country", "maturity_years"], sort=False)["yield"]
    target["future_yield"] = grouped.shift(-horizon_days)
    target["target_yield_change"] = target["future_yield"] - target["yield"]
    target["horizon_days"] = horizon_days
    return target.dropna(subset=["future_yield", "target_yield_change"])
