from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from yieldrep.config import ProjectConfig, SourceConfig

MACRO_INDICATOR_COLUMNS = ("date", "country", "indicator", "value", "source")
MACRO_REGIME_COLUMNS = ("date", "country", "indicator", "value", "macro_regime", "source")


def build_macro_indicators(config: ProjectConfig) -> Path:
    """Build normalized macro-indicator parquet from configured raw files."""
    frames = [
        _normalize_macro_source(name, source_config)
        for name, source_config in config.macro_indicators.items()
    ]
    if not frames:
        raise ValueError("No macro-indicator sources configured")

    indicators = validate_macro_indicator_frame(pd.concat(frames, ignore_index=True))
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    indicators.to_parquet(config.macro_indicators_path, index=False)
    return config.macro_indicators_path


def build_macro_regimes(config: ProjectConfig) -> Path:
    """Build expanding-tercile macro regimes from normalized macro indicators."""
    indicators = pd.read_parquet(config.macro_indicators_path)
    regimes = make_macro_regimes(indicators)

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    regimes.to_parquet(config.macro_regimes_path, index=False)
    return config.macro_regimes_path


def normalize_macro_indicator_source(
    frame: pd.DataFrame,
    source_config: SourceConfig,
    indicator: str,
) -> pd.DataFrame:
    """Normalize one FRED macro series to date, country, indicator, value, source."""
    date_column = _first_available_column(frame, ["DATE", "observation_date", "date"])
    value_column = _first_value_column(frame, date_column)
    data = frame.loc[:, [date_column, value_column]].copy()
    data["date"] = data[date_column]
    data["country"] = source_config.country
    data["indicator"] = indicator
    data["value"] = data[value_column]
    data["source"] = source_config.source
    normalized = validate_macro_indicator_frame(data)

    if source_config.source in {"fred_cpi_index_yoy", "fred_hicp_index_yoy"}:
        normalized["value"] = (
            normalized.sort_values("date")["value"].pct_change(12) * 100.0
        )
        normalized = normalized.dropna(subset=["value"]).reset_index(drop=True)
        normalized["indicator"] = "inflation"
    elif source_config.source == "fred_inflation_yoy":
        normalized["indicator"] = "inflation"
    elif source_config.source == "fred_unemployment_rate":
        normalized["indicator"] = "unemployment"
    else:
        raise ValueError(f"Unsupported macro-indicator source: {source_config.source}")

    return validate_macro_indicator_frame(normalized)


def validate_macro_indicator_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize the common macro-indicator schema."""
    missing = [column for column in MACRO_INDICATOR_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required macro-indicator columns: {missing}")

    indicators = frame.loc[:, MACRO_INDICATOR_COLUMNS].copy()
    indicators["date"] = pd.to_datetime(indicators["date"], errors="raise").dt.normalize()
    indicators["country"] = indicators["country"].astype("string")
    indicators["indicator"] = indicators["indicator"].astype("string")
    indicators["value"] = pd.to_numeric(indicators["value"], errors="coerce")
    indicators["source"] = indicators["source"].astype("string")
    indicators = indicators.dropna(subset=["value"])
    if indicators.isna().any().any():
        raise ValueError("Macro-indicator data contains null values in required columns")
    return indicators.sort_values(["country", "indicator", "date"]).reset_index(drop=True)


def make_macro_regimes(indicators: pd.DataFrame, min_history: int = 60) -> pd.DataFrame:
    """Label low/medium/high macro regimes from expanding historical terciles."""
    if min_history <= 1:
        raise ValueError("min_history must be greater than 1")

    base = validate_macro_indicator_frame(indicators)
    frames = [
        _macro_regimes(group, min_history=min_history)
        for _, group in base.groupby(["country", "indicator"], sort=True)
    ]
    if not frames:
        return pd.DataFrame(columns=MACRO_REGIME_COLUMNS)
    regimes = pd.concat(frames, ignore_index=True)
    return regimes.dropna(subset=["macro_regime"]).loc[:, MACRO_REGIME_COLUMNS]


def _normalize_macro_source(name: str, source_config: SourceConfig) -> pd.DataFrame:
    frame = pd.read_csv(source_config.raw_file, na_values=["", ".", "NA", "NaN"])
    frame.columns = [str(column).strip() for column in frame.columns]
    return normalize_macro_indicator_source(frame, source_config, indicator=name)


def _macro_regimes(group: pd.DataFrame, min_history: int) -> pd.DataFrame:
    frame = group.sort_values("date").copy()
    values = frame["value"]
    lower = values.expanding(min_periods=min_history).quantile(1 / 3).shift(1)
    upper = values.expanding(min_periods=min_history).quantile(2 / 3).shift(1)
    frame["macro_regime"] = np.select(
        [values <= lower, values <= upper],
        ["low", "medium"],
        default="high",
    )
    frame.loc[lower.isna() | upper.isna(), "macro_regime"] = pd.NA
    return frame


def _first_available_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Missing macro-indicator date column; expected one of {candidates}")


def _first_value_column(frame: pd.DataFrame, date_column: str) -> str:
    candidates = [str(column) for column in frame.columns if str(column) != date_column]
    if not candidates:
        raise ValueError("Missing macro-indicator value column")
    return candidates[0]
