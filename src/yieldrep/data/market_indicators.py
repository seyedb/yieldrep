from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from yieldrep.config import ProjectConfig, SourceConfig

MARKET_INDICATOR_COLUMNS = ("date", "indicator", "value", "source")
MARKET_REGIME_COLUMNS = ("date", "indicator", "value", "market_vol_regime", "source")


def build_market_indicators(config: ProjectConfig) -> Path:
    """Build normalized market-indicator parquet from configured raw files."""
    frames = [
        _normalize_market_indicator_source(name, source_config)
        for name, source_config in config.market_indicators.items()
    ]
    if not frames:
        raise ValueError("No market-indicator sources configured")

    indicators = validate_market_indicator_frame(pd.concat(frames, ignore_index=True))
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    indicators.to_parquet(config.market_indicators_path, index=False)
    return config.market_indicators_path


def build_market_regimes(config: ProjectConfig) -> Path:
    """Build expanding-tercile low/medium/high market-volatility regimes."""
    indicators = pd.read_parquet(config.market_indicators_path)
    regimes = make_market_regimes(indicators)

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    regimes.to_parquet(config.market_regimes_path, index=False)
    return config.market_regimes_path


def normalize_market_indicator_source(
    frame: pd.DataFrame,
    source_config: SourceConfig,
    indicator: str,
) -> pd.DataFrame:
    """Normalize one market indicator to date, indicator, value, source."""
    if source_config.source == "yahoo_move":
        return _normalize_yahoo_chart(frame, source_config, indicator)
    if source_config.source != "fred_vixcls":
        raise ValueError(f"Unsupported market-indicator source: {source_config.source}")

    date_column = _first_available_column(frame, ["DATE", "observation_date", "date"])
    value_column = _first_available_column(
        frame,
        ["VIXCLS", "MOVE", source_config.source.removeprefix("fred_").upper()],
    )
    data = frame.loc[:, [date_column, value_column]].copy()
    data["date"] = data[date_column]
    data["indicator"] = indicator.upper()
    data["value"] = data[value_column]
    data["source"] = source_config.source
    return validate_market_indicator_frame(data)


def validate_market_indicator_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize the common market-indicator schema."""
    missing = [column for column in MARKET_INDICATOR_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required market-indicator columns: {missing}")

    indicators = frame.loc[:, MARKET_INDICATOR_COLUMNS].copy()
    indicators["date"] = pd.to_datetime(indicators["date"], errors="raise").dt.normalize()
    indicators["indicator"] = indicators["indicator"].astype("string")
    indicators["value"] = pd.to_numeric(indicators["value"], errors="coerce")
    indicators["source"] = indicators["source"].astype("string")
    indicators = indicators.dropna(subset=["value"])
    if indicators.isna().any().any():
        raise ValueError("Market-indicator data contains null values in required columns")
    return indicators.sort_values(["indicator", "date"]).reset_index(drop=True)


def make_market_regimes(
    indicators: pd.DataFrame,
    min_history: int = 252,
) -> pd.DataFrame:
    """Label market-volatility regimes from expanding historical terciles."""
    if min_history <= 1:
        raise ValueError("min_history must be greater than 1")

    base = validate_market_indicator_frame(indicators)
    frames = [
        _indicator_regimes(group, min_history=min_history)
        for _, group in base.groupby("indicator", sort=True)
    ]
    if not frames:
        return pd.DataFrame(columns=MARKET_REGIME_COLUMNS)
    regimes = pd.concat(frames, ignore_index=True)
    return regimes.dropna(subset=["market_vol_regime"]).loc[:, MARKET_REGIME_COLUMNS]


def _normalize_market_indicator_source(name: str, source_config: SourceConfig) -> pd.DataFrame:
    if source_config.source == "yahoo_move":
        with source_config.raw_file.open("r", encoding="utf-8") as handle:
            return _normalize_yahoo_chart_json(json.load(handle), source_config, indicator=name)

    frame = pd.read_csv(source_config.raw_file, na_values=["", ".", "NA", "NaN"])
    frame.columns = [str(column).strip() for column in frame.columns]
    return normalize_market_indicator_source(frame, source_config, indicator=name)


def _normalize_yahoo_chart(
    frame: pd.DataFrame,
    source_config: SourceConfig,
    indicator: str,
) -> pd.DataFrame:
    if not {"timestamp", "close"}.issubset(frame.columns):
        raise ValueError("Yahoo market indicator frame requires timestamp and close columns")
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["timestamp"], unit="s").dt.normalize(),
            "indicator": indicator.upper(),
            "value": frame["close"],
            "source": source_config.source,
        }
    )
    return validate_market_indicator_frame(data)


def _normalize_yahoo_chart_json(
    payload: dict[str, object],
    source_config: SourceConfig,
    indicator: str,
) -> pd.DataFrame:
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        raise ValueError("Yahoo chart payload is missing chart data")
    result = chart.get("result")
    if not isinstance(result, list) or not result:
        raise ValueError("Yahoo chart payload has no result data")
    first = result[0]
    if not isinstance(first, dict):
        raise ValueError("Yahoo chart result is invalid")

    timestamps = first.get("timestamp")
    indicators = first.get("indicators")
    if not isinstance(timestamps, list) or not isinstance(indicators, dict):
        raise ValueError("Yahoo chart payload is missing timestamps or indicators")
    quote = indicators.get("quote")
    if not isinstance(quote, list) or not quote or not isinstance(quote[0], dict):
        raise ValueError("Yahoo chart payload is missing quote data")
    closes = quote[0].get("close")
    if not isinstance(closes, list):
        raise ValueError("Yahoo chart payload is missing close data")

    frame = pd.DataFrame({"timestamp": timestamps, "close": closes})
    return _normalize_yahoo_chart(frame, source_config, indicator)


def _indicator_regimes(group: pd.DataFrame, min_history: int) -> pd.DataFrame:
    frame = group.sort_values("date").copy()
    values = frame["value"]
    lower = values.expanding(min_periods=min_history).quantile(1 / 3).shift(1)
    upper = values.expanding(min_periods=min_history).quantile(2 / 3).shift(1)
    frame["market_vol_regime"] = np.select(
        [values <= lower, values <= upper],
        ["low", "medium"],
        default="high",
    )
    frame.loc[lower.isna() | upper.isna(), "market_vol_regime"] = pd.NA
    return frame


def _first_available_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Missing market-indicator columns; expected one of {candidates}")
