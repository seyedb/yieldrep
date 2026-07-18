from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from yieldrep.config import ProjectConfig


@dataclass(frozen=True)
class DiagnosticTarget:
    name: str
    path: Path
    target_column: str


def diagnose_lagged_baseline(config: ProjectConfig) -> Path:
    """Measure whether lagged baselines mainly reflect target autocorrelation."""
    frames: list[pd.DataFrame] = []
    for target in _diagnostic_targets(config):
        if not target.path.exists():
            continue

        target_data = pd.read_parquet(
            target.path,
            columns=["date", "country", "maturity_years", "horizon_days", target.target_column],
        )
        frames.append(_diagnose_target_autocorrelation(target_data, target, config.evaluation.lag_days))
        frames.append(_diagnose_lag_feature_correlation(config, target))

    diagnostics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    config.evaluation_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.to_parquet(config.lagged_diagnostics_path, index=False)
    return config.lagged_diagnostics_path


def _diagnose_target_autocorrelation(
    data: pd.DataFrame,
    target: DiagnosticTarget,
    lag_days: list[int],
) -> pd.DataFrame:
    sorted_data = data.sort_values(["country", "maturity_years", "horizon_days", "date"]).copy()
    group_columns = ["country", "maturity_years", "horizon_days"]
    grouped = sorted_data.groupby(group_columns, sort=False)[target.target_column]

    rows: list[pd.DataFrame] = []
    for lag in lag_days:
        lag_column = f"target_lag_{lag}"
        sorted_data[lag_column] = grouped.shift(lag)
        rows.append(
            _correlation_rows(
                sorted_data,
                target.name,
                "target_autocorrelation",
                lag,
                lag_column,
                target.target_column,
            )
        )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _diagnose_lag_feature_correlation(config: ProjectConfig, target: DiagnosticTarget) -> pd.DataFrame:
    lagged_path = config.modeling_dir / _lagged_dataset_name(target.name)
    if not lagged_path.exists():
        return pd.DataFrame()

    lag_columns = [f"lag_{lag}_change" for lag in config.evaluation.lag_days]
    data = pd.read_parquet(
        lagged_path,
        columns=[
            "date",
            "country",
            "maturity_years",
            "horizon_days",
            target.target_column,
            *lag_columns,
        ],
    )

    rows: list[pd.DataFrame] = []
    for lag_column in lag_columns:
        lag = int(lag_column.removeprefix("lag_").removesuffix("_change"))
        rows.append(
            _correlation_rows(
                data,
                target.name,
                "lag_feature_correlation",
                lag,
                lag_column,
                target.target_column,
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _correlation_rows(
    data: pd.DataFrame,
    target_name: str,
    diagnostic: str,
    lag: int,
    lag_column: str,
    target_column: str,
) -> pd.DataFrame:
    complete = data.dropna(subset=[lag_column, target_column])
    if complete.empty:
        return pd.DataFrame()

    rows = (
        complete.groupby(["country", "maturity_years", "horizon_days"], sort=True)
        .apply(_correlation_summary, lag_column=lag_column, target_column=target_column)
        .reset_index()
    )
    rows["target"] = target_name
    rows["diagnostic"] = diagnostic
    rows["lag_days"] = lag
    return rows.loc[
        :,
        [
            "target",
            "diagnostic",
            "country",
            "maturity_years",
            "horizon_days",
            "lag_days",
            "observations",
            "correlation",
            "sign_agreement",
        ],
    ]


def _correlation_summary(
    group: pd.DataFrame,
    lag_column: str,
    target_column: str,
) -> pd.Series:
    lagged = group[lag_column].to_numpy(dtype=float)
    target = group[target_column].to_numpy(dtype=float)
    return pd.Series(
        {
            "observations": len(group),
            "correlation": _safe_corr(lagged, target),
            "sign_agreement": float(np.mean(np.sign(lagged) == np.sign(target))),
        }
    )


def _safe_corr(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    if len(left) < 2 or np.isclose(np.std(left), 0.0) or np.isclose(np.std(right), 0.0):
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _diagnostic_targets(config: ProjectConfig) -> list[DiagnosticTarget]:
    return [
        DiagnosticTarget(
            name="yield_change",
            path=config.targets_path,
            target_column="target_yield_change",
        ),
        DiagnosticTarget(
            name="residual_change",
            path=config.residual_targets_path,
            target_column="target_residual_change",
        ),
        DiagnosticTarget(
            name="vol_change",
            path=config.vol_targets_path,
            target_column="target_vol_change",
        ),
    ]


def _lagged_dataset_name(target_name: str) -> str:
    if target_name == "yield_change":
        return "lagged_targets.parquet"
    if target_name == "residual_change":
        return "lagged_residual_targets.parquet"
    if target_name == "vol_change":
        return "lagged_vol_targets.parquet"
    raise ValueError(f"Unsupported target for lagged diagnostics: {target_name}")
