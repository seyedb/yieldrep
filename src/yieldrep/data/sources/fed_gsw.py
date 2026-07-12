from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_SVENY_PATTERN = re.compile(r"^SVENY(?P<years>\d{2})$")


def load_fed_gsw_raw(path: Path) -> pd.DataFrame:
    """Read a local Federal Reserve GSW CSV file."""
    return pd.read_csv(path, skiprows=_header_offset(path), na_values=["NA"])


def fed_gsw_maturity_columns(frame: pd.DataFrame) -> dict[str, float]:
    """Return zero-coupon maturity columns as column name to years."""
    maturities: dict[str, float] = {}
    for column in frame.columns:
        match = _SVENY_PATTERN.match(str(column))
        if match is not None:
            maturities[str(column)] = float(match.group("years"))
    return maturities


def fed_gsw_date_column(frame: pd.DataFrame) -> str:
    if "Date" in frame.columns:
        return "Date"
    raise ValueError("Fed GSW raw data must contain a Date column")


def _header_offset(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line.startswith("Date,"):
                return line_number
    return 0
