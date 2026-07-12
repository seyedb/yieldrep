from __future__ import annotations

import pandas as pd

from yieldrep.data.schema import validate_curve_frame
from yieldrep.data.sources.bank_of_canada import (
    bank_of_canada_date_column,
    bank_of_canada_maturity_columns,
)
from yieldrep.data.sources.fed_gsw import fed_gsw_date_column, fed_gsw_maturity_columns


def normalize_fed_gsw(frame: pd.DataFrame, country: str = "US", source: str = "fed_gsw") -> pd.DataFrame:
    """Normalize Fed GSW zero-coupon yields to the common long curve schema."""
    date_column = fed_gsw_date_column(frame)
    maturity_columns = fed_gsw_maturity_columns(frame)
    if not maturity_columns:
        raise ValueError("Fed GSW raw data does not contain SVENY maturity columns")

    return _normalize_wide_curve(
        frame=frame,
        date_column=date_column,
        maturity_columns=maturity_columns,
        country=country,
        source=source,
        yield_scale=1.0,
    )


def normalize_bank_of_canada(
    frame: pd.DataFrame,
    country: str = "CA",
    source: str = "bank_of_canada",
) -> pd.DataFrame:
    """Normalize Bank of Canada zero-coupon yields to the common long curve schema."""
    date_column = bank_of_canada_date_column(frame)
    maturity_columns = bank_of_canada_maturity_columns(frame)
    if not maturity_columns:
        raise ValueError("Bank of Canada raw data does not contain ZC maturity columns")

    return _normalize_wide_curve(
        frame=frame,
        date_column=date_column,
        maturity_columns=maturity_columns,
        country=country,
        source=source,
        yield_scale=100.0,
    )


def _normalize_wide_curve(
    frame: pd.DataFrame,
    date_column: str,
    maturity_columns: dict[str, float],
    country: str,
    source: str,
    yield_scale: float,
) -> pd.DataFrame:
    long = frame.melt(
        id_vars=[date_column],
        value_vars=list(maturity_columns),
        var_name="maturity_code",
        value_name="yield",
    )
    long["date"] = long[date_column]
    long["country"] = country
    long["maturity_years"] = long["maturity_code"].map(maturity_columns)
    long["yield"] = pd.to_numeric(long["yield"], errors="coerce") * yield_scale
    long["source"] = source

    return validate_curve_frame(long.dropna(subset=["yield"]))
