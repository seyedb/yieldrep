from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_SR_PATTERN = re.compile(r"^SR_(?:(?P<years>\d+)Y)?(?:(?P<months>\d+)M)?$")


def load_ecb_yield_curve_raw(path: Path) -> pd.DataFrame:
    """Read a local ECB yield-curve CSV file."""
    frame = pd.read_csv(path, na_values=["", "NA", "NaN"])
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def ecb_maturity_from_code(code: str) -> float | None:
    """Parse ECB spot-rate maturity codes such as SR_3M, SR_1Y, or SR_10Y6M."""
    match = _SR_PATTERN.match(code)
    if match is None:
        return None

    years = int(match.group("years") or 0)
    months = int(match.group("months") or 0)
    maturity = years + months / 12.0
    return maturity if maturity > 0 else None


def ecb_spot_rate_columns(frame: pd.DataFrame) -> dict[str, float]:
    """Return wide ECB spot-rate columns as column name to years."""
    maturities: dict[str, float] = {}
    for column in frame.columns:
        maturity = ecb_maturity_from_code(str(column))
        if maturity is not None:
            maturities[str(column)] = maturity
    return maturities


def ecb_date_column(frame: pd.DataFrame) -> str:
    if "TIME_PERIOD" in frame.columns:
        return "TIME_PERIOD"
    if "Date" in frame.columns:
        return "Date"
    raise ValueError("ECB raw data must contain a TIME_PERIOD or Date column")
