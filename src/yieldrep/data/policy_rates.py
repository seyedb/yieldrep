from __future__ import annotations

from pathlib import Path

import pandas as pd

from yieldrep.config import ProjectConfig, SourceConfig

POLICY_RATE_COLUMNS = ("date", "country", "policy_rate", "source")


def build_policy_rates(config: ProjectConfig) -> Path:
    """Build normalized policy-rate parquet from configured local raw files."""
    frames = [
        _normalize_policy_source(source_config)
        for source_config in config.policy_rates.values()
    ]
    if not frames:
        raise ValueError("No policy-rate sources configured")

    policy_rates = validate_policy_rate_frame(pd.concat(frames, ignore_index=True))
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    policy_rates.to_parquet(config.policy_rates_path, index=False)
    return config.policy_rates_path


def normalize_policy_rate_source(frame: pd.DataFrame, source_config: SourceConfig) -> pd.DataFrame:
    """Normalize one policy-rate raw frame to date, country, policy_rate, source."""
    if source_config.source in {"fred_fedfunds", "fred_dff"}:
        date_column = _first_available_column(frame, ["DATE", "observation_date", "date"])
        value_column = _first_available_column(frame, ["FEDFUNDS", "DFF"])
        return _normalize_two_column_policy_rate(
            frame,
            date_column=date_column,
            value_column=value_column,
            source_config=source_config,
        )
    if source_config.source == "bank_of_canada_policy_rate":
        value_column = _first_available_column(frame, ["V39079", "OBS_VALUE"])
        date_column = _first_available_column(frame, ["date", "Date", "TIME_PERIOD"])
        return _normalize_two_column_policy_rate(
            frame,
            date_column=date_column,
            value_column=value_column,
            source_config=source_config,
        )
    if source_config.source == "ecb_deposit_facility":
        return _normalize_two_column_policy_rate(
            frame,
            date_column="TIME_PERIOD",
            value_column="OBS_VALUE",
            source_config=source_config,
        )
    raise ValueError(f"Unsupported policy-rate source: {source_config.source}")


def validate_policy_rate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize the common policy-rate schema."""
    missing = [column for column in POLICY_RATE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required policy-rate columns: {missing}")

    policy_rates = frame.loc[:, POLICY_RATE_COLUMNS].copy()
    policy_rates["date"] = pd.to_datetime(policy_rates["date"], errors="raise").dt.normalize()
    policy_rates["country"] = policy_rates["country"].astype("string")
    policy_rates["policy_rate"] = pd.to_numeric(policy_rates["policy_rate"], errors="coerce")
    policy_rates["source"] = policy_rates["source"].astype("string")
    policy_rates = policy_rates.dropna(subset=["policy_rate"])
    if policy_rates.isna().any().any():
        raise ValueError("Policy-rate data contains null values in required columns")
    return policy_rates.sort_values(["country", "date"]).reset_index(drop=True)


def _normalize_policy_source(source_config: SourceConfig) -> pd.DataFrame:
    frame = pd.read_csv(
        source_config.raw_file,
        na_values=["", ".", "NA", "NaN"],
        skiprows=_policy_header_offset(source_config),
    )
    frame.columns = [str(column).strip() for column in frame.columns]
    return normalize_policy_rate_source(frame, source_config)


def _policy_header_offset(source_config: SourceConfig) -> int:
    if source_config.source != "bank_of_canada_policy_rate":
        return 0

    with source_config.raw_file.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle):
            if line.strip().strip('"') == "OBSERVATIONS":
                return line_number + 1
    return 0


def _normalize_two_column_policy_rate(
    frame: pd.DataFrame,
    date_column: str,
    value_column: str,
    source_config: SourceConfig,
) -> pd.DataFrame:
    missing = [column for column in [date_column, value_column] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing policy-rate columns: {missing}")

    data = frame.loc[:, [date_column, value_column]].copy()
    data["date"] = data[date_column]
    data["country"] = source_config.country
    data["policy_rate"] = data[value_column]
    data["source"] = source_config.source
    return validate_policy_rate_frame(data)


def _first_available_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Missing policy-rate columns; expected one of {candidates}")
