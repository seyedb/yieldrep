from __future__ import annotations

from typing import Final

import pandas as pd

CURVE_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "country",
    "maturity_years",
    "yield",
    "source",
)


def validate_curve_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize the common long-format yield curve schema."""
    missing = [column for column in CURVE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required curve columns: {missing}")

    curve = frame.loc[:, CURVE_COLUMNS].copy()
    curve["date"] = pd.to_datetime(curve["date"], errors="raise").dt.normalize()
    curve["country"] = curve["country"].astype("string")
    curve["maturity_years"] = pd.to_numeric(curve["maturity_years"], errors="raise")
    curve["yield"] = pd.to_numeric(curve["yield"], errors="raise")
    curve["source"] = curve["source"].astype("string")

    if curve.isna().any().any():
        raise ValueError("Curve data contains null values in required columns")
    if (curve["maturity_years"] <= 0).any():
        raise ValueError("Curve maturities must be positive")

    return curve.sort_values(["country", "date", "maturity_years"]).reset_index(drop=True)
