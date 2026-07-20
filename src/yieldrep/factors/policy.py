from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from yieldrep.config import ProjectConfig
from yieldrep.factors.curve import curve_panel

POLICY_FEATURE_COLUMNS = [
    "policy_rate",
    "policy_change_21d",
    "policy_change_63d",
    "policy_change_252d",
    "policy_2y_spread",
]


def build_policy_features(config: ProjectConfig) -> Path:
    """Build policy-rate features aligned to country curve dates."""
    policy_rates = pd.read_parquet(config.policy_rates_path)
    curves = pd.read_parquet(config.curves_path)
    features = make_policy_features(policy_rates, curves)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(config.policy_features_path, index=False)
    return config.policy_features_path


def make_policy_features(policy_rates: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """Create policy level, recent changes, and curve-policy spread features."""
    frames: list[pd.DataFrame] = []
    for country in sorted(curves["country"].dropna().unique()):
        country_key = str(country)
        country_policy = policy_rates.loc[policy_rates["country"] == country_key].copy()
        if country_policy.empty:
            continue

        dates = (
            curves.loc[curves["country"] == country_key, ["date", "country"]]
            .drop_duplicates()
            .sort_values("date")
            .reset_index(drop=True)
        )
        aligned = pd.merge_asof(
            dates,
            country_policy.sort_values("date").loc[:, ["date", "policy_rate"]],
            on="date",
            direction="backward",
        )
        aligned = aligned.dropna(subset=["policy_rate"]).reset_index(drop=True)
        if aligned.empty:
            continue

        aligned["policy_change_21d"] = aligned["policy_rate"] - aligned["policy_rate"].shift(21)
        aligned["policy_change_63d"] = aligned["policy_rate"] - aligned["policy_rate"].shift(63)
        aligned["policy_change_252d"] = aligned["policy_rate"] - aligned["policy_rate"].shift(252)
        aligned["policy_regime_63d"] = _policy_regime(aligned["policy_change_63d"])
        aligned["policy_2y_spread"] = _nearest_2y_yield(curves, country_key).reindex(
            pd.to_datetime(aligned["date"])
        ).to_numpy() - aligned["policy_rate"].to_numpy()
        frames.append(aligned)

    if not frames:
        return pd.DataFrame(columns=["date", "country", *POLICY_FEATURE_COLUMNS, "policy_regime_63d"])

    return pd.concat(frames, ignore_index=True).sort_values(["country", "date"]).reset_index(
        drop=True
    )


def _nearest_2y_yield(curves: pd.DataFrame, country: str) -> pd.Series:
    panel = curve_panel(curves, country)
    if panel.empty:
        return pd.Series(dtype=float)
    nearest_maturity = min(panel.columns, key=lambda maturity: abs(float(maturity) - 2.0))
    series = panel[nearest_maturity].copy()
    series.index = pd.to_datetime(series.index)
    return series


def _policy_regime(policy_change: pd.Series) -> pd.Series:
    labels = np.select(
        [policy_change > 0.05, policy_change < -0.05],
        ["hiking", "easing"],
        default="stable",
    )
    return pd.Series(labels, index=policy_change.index, dtype="string")
