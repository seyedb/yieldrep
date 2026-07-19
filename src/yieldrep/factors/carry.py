from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from yieldrep.config import ProjectConfig
from yieldrep.factors.curve import curve_panel

ROLL_HORIZONS_YEARS = {
    "1m": 1.0 / 12.0,
    "3m": 0.25,
    "12m": 1.0,
}

CARRY_ROLL_FEATURE_COLUMNS = [
    "carry_1m",
    "roll_down_1m",
    "carry_3m",
    "roll_down_3m",
    "carry_12m",
    "roll_down_12m",
]


def build_carry_roll_features(config: ProjectConfig) -> Path:
    """Build simple carry and roll-down proxy features from normalized curves."""
    curves = pd.read_parquet(config.curves_path)
    features = make_carry_roll_features(curves)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(config.carry_roll_features_path, index=False)
    return config.carry_roll_features_path


def make_carry_roll_features(curves: pd.DataFrame) -> pd.DataFrame:
    """Create maturity-specific carry and roll-down proxies.

    With only public zero-coupon curves, these are deliberately simple proxies:
    carry is the yield scaled to the roll horizon, and roll-down is the same-day
    interpolated yield at the rolled maturity minus the current yield.
    """
    frames: list[pd.DataFrame] = []
    for country in sorted(curves["country"].dropna().unique()):
        panel = curve_panel(curves, country=str(country))
        if panel.empty:
            continue
        frames.append(_country_carry_roll_features(panel, country=str(country)))

    if not frames:
        return pd.DataFrame(
            columns=["date", "country", "maturity_years", *CARRY_ROLL_FEATURE_COLUMNS]
        )

    return pd.concat(frames, ignore_index=True).loc[
        :, ["date", "country", "maturity_years", *CARRY_ROLL_FEATURE_COLUMNS]
    ]


def _country_carry_roll_features(panel: pd.DataFrame, country: str) -> pd.DataFrame:
    maturities = np.asarray(panel.columns, dtype=float)
    rows: list[pd.DataFrame] = []
    for date, yields in panel.iterrows():
        curve = pd.Series(yields.to_numpy(dtype=float), index=maturities).dropna()
        if curve.empty:
            continue

        frame = pd.DataFrame(
            {
                "date": date,
                "country": country,
                "maturity_years": curve.index.to_numpy(dtype=float),
                "yield": curve.to_numpy(dtype=float),
            }
        )
        for label, horizon_years in ROLL_HORIZONS_YEARS.items():
            frame[f"carry_{label}"] = frame["yield"] * horizon_years
            frame[f"roll_down_{label}"] = _rolled_yields(curve, horizon_years) - frame["yield"]
        rows.append(frame.drop(columns=["yield"]))

    if not rows:
        return pd.DataFrame(
            columns=["date", "country", "maturity_years", *CARRY_ROLL_FEATURE_COLUMNS]
        )
    return pd.concat(rows, ignore_index=True)


def _rolled_yields(curve: pd.Series, horizon_years: float) -> NDArray[np.float64]:
    maturities = curve.index.to_numpy(dtype=float)
    yields = curve.to_numpy(dtype=float)
    rolled_maturities = np.maximum(maturities - horizon_years, maturities.min())
    return np.interp(rolled_maturities, maturities, yields)
