from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_ZC_PATTERN = re.compile(r"^ZC(?P<quarters>\d{3,4})YR$")


def load_bank_of_canada_raw(path: Path) -> pd.DataFrame:
    """Read a local Bank of Canada zero-coupon CSV file."""
    frame = pd.read_csv(path, na_values=["na", "NA"])
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame.loc[:, ~frame.columns.str.startswith("Unnamed")]


def bank_of_canada_maturity_columns(frame: pd.DataFrame) -> dict[str, float]:
    """Return zero-coupon maturity columns as column name to years."""
    maturities: dict[str, float] = {}
    for column in frame.columns:
        match = _ZC_PATTERN.match(str(column))
        if match is not None:
            maturities[str(column)] = int(match.group("quarters")) / 100.0
    return maturities


def bank_of_canada_date_column(frame: pd.DataFrame) -> str:
    if "Date" in frame.columns:
        return "Date"
    raise ValueError("Bank of Canada raw data must contain a Date column")
