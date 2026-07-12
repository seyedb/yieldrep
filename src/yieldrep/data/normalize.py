from __future__ import annotations

from pathlib import Path

import pandas as pd

from yieldrep.config import ProjectConfig, SourceConfig
from yieldrep.data.schema import validate_curve_frame
from yieldrep.data.sources.bank_of_canada import (
    bank_of_canada_date_column,
    bank_of_canada_maturity_columns,
    load_bank_of_canada_raw,
)
from yieldrep.data.sources.fed_gsw import (
    fed_gsw_date_column,
    fed_gsw_maturity_columns,
    load_fed_gsw_raw,
)


def build_curves_parquet(config: ProjectConfig) -> Path:
    """Build the normalized curve parquet file from configured local raw files."""
    frames = [
        _normalize_source(name, source_config)
        for name, source_config in config.sources.items()
    ]
    curves = validate_curve_frame(pd.concat(frames, ignore_index=True))

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    curves.to_parquet(config.curves_path, index=False)
    return config.curves_path


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


def _normalize_source(name: str, source_config: SourceConfig) -> pd.DataFrame:
    if name == "fed_gsw":
        raw = load_fed_gsw_raw(source_config.raw_file)
        return normalize_fed_gsw(raw, country=source_config.country, source=source_config.source)
    if name == "bank_of_canada":
        raw = load_bank_of_canada_raw(source_config.raw_file)
        return normalize_bank_of_canada(
            raw,
            country=source_config.country,
            source=source_config.source,
        )
    raise ValueError(f"Unsupported source: {name}")
