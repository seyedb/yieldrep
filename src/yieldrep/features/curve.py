from __future__ import annotations

from pathlib import Path

import pandas as pd

from yieldrep.config import ProjectConfig


CURVE_FEATURE_COLUMNS = [
    "level",
    "slope_10y_2y",
    "curvature_2s5s10s",
    "front_slope_2y_1y",
    "long_slope_30y_10y",
]


def curve_panel(curves: pd.DataFrame, country: str) -> pd.DataFrame:
    """Pivot long-format curves into a date by maturity panel for one country."""
    panel = curves.loc[curves["country"] == country].pivot_table(
        index="date",
        columns="maturity_years",
        values="yield",
        aggfunc="mean",
    )
    panel = panel.sort_index().sort_index(axis=1)
    panel.columns = [float(column) for column in panel.columns]
    return panel


def build_curve_features(config: ProjectConfig) -> Path:
    """Build engineered level, slope, and curvature features from curves."""
    curves = pd.read_parquet(config.curves_path)
    features = make_curve_features(curves)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(config.curve_features_path, index=False)
    return config.curve_features_path


def make_curve_features(curves: pd.DataFrame) -> pd.DataFrame:
    """Create standard curve-shape features by country and date.

    The features are intentionally classical rates descriptors: level, selected
    slope spreads, and a 2s5s10s butterfly-style curvature measure.
    """
    frames: list[pd.DataFrame] = []
    for country in sorted(curves["country"].dropna().unique()):
        panel = curve_panel(curves, country=str(country))
        if panel.empty:
            continue
        country_features = pd.DataFrame(index=panel.index)
        country_features["country"] = str(country)
        country_features["level"] = panel.mean(axis=1)
        country_features["slope_10y_2y"] = _at(panel, 10.0) - _at(panel, 2.0)
        country_features["curvature_2s5s10s"] = (
            2.0 * _at(panel, 5.0) - _at(panel, 2.0) - _at(panel, 10.0)
        )
        country_features["front_slope_2y_1y"] = _at(panel, 2.0) - _at(panel, 1.0)
        country_features["long_slope_30y_10y"] = _at(panel, 30.0) - _at(panel, 10.0)
        frames.append(country_features.reset_index().rename(columns={"index": "date"}))

    if not frames:
        return pd.DataFrame(columns=["date", "country", *CURVE_FEATURE_COLUMNS])

    return pd.concat(frames, ignore_index=True).loc[:, ["date", "country", *CURVE_FEATURE_COLUMNS]]


def _at(panel: pd.DataFrame, maturity: float) -> pd.Series:
    """Return the nearest available maturity column to the requested anchor."""
    nearest_maturity = min(panel.columns, key=lambda column: abs(float(column) - maturity))
    return panel[nearest_maturity]
