from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from yieldrep.config import ProjectConfig
from yieldrep.evaluation.reconstruction import _out_of_sample_reconstruction_errors


SUMMARY_COLUMNS = [
    "evidence_type",
    "representation",
    "country",
    "horizon_days",
    "regime_type",
    "indicator",
    "regime",
    "observations",
    "dates",
    "metric",
    "value",
]


def build_macro_conditioned_representation_summary(config: ProjectConfig) -> Path:
    """Write regime-conditioned diagnostics for existing representation evidence."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    summary = macro_conditioned_representation_summary(config)
    summary.to_csv(config.macro_conditioned_representation_summary_table_path, index=False)
    return config.macro_conditioned_representation_summary_table_path


def macro_conditioned_representation_summary(config: ProjectConfig) -> pd.DataFrame:
    rows = [
        _reconstruction_by_regime(config),
        _residual_rv_by_regime(config),
        _volatility_classification_summary(config),
        _learned_state_separation_summary(config),
    ]
    non_empty = [row for row in rows if not row.empty]
    if not non_empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.concat(non_empty, ignore_index=True).loc[:, SUMMARY_COLUMNS].sort_values(
        [
            "evidence_type",
            "country",
            "representation",
            "regime_type",
            "indicator",
            "horizon_days",
            "regime",
            "metric",
        ],
        na_position="last",
    ).reset_index(drop=True)


def _reconstruction_by_regime(config: ProjectConfig) -> pd.DataFrame:
    if not config.curves_path.exists():
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    curves = pd.read_parquet(config.curves_path)
    errors = _out_of_sample_reconstruction_errors(curves, config)
    if errors.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    errors = _keep_best_reconstruction_specs(errors, config)
    regimes = _joined_regimes(
        errors.loc[
            :,
            [
                "date",
                "country",
                "reconstruction_task",
                "representation",
                "n_components",
                "error",
                "squared_error",
                "absolute_error",
            ],
        ],
        config,
    )
    if regimes.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    grouped = (
        regimes.groupby(
            [
                "reconstruction_task",
                "representation",
                "country",
                "regime_type",
                "indicator",
                "regime",
            ],
            sort=True,
            observed=True,
        )
        .agg(
            observations=("error", "size"),
            dates=("date", "nunique"),
            mse=("squared_error", "mean"),
            mae=("absolute_error", "mean"),
        )
        .reset_index()
    )
    grouped["rmse"] = np.sqrt(grouped["mse"])
    grouped["evidence_type"] = grouped["reconstruction_task"]
    grouped["horizon_days"] = np.nan

    rmse = _metric_rows(grouped, metric_column="rmse", metric_name="rmse")
    mae = _metric_rows(grouped, metric_column="mae", metric_name="mae")
    return pd.concat([rmse, mae], ignore_index=True)


def _keep_best_reconstruction_specs(errors: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    if not config.reconstruction_oos_comparison_table_path.exists():
        return errors

    comparison = pd.read_csv(config.reconstruction_oos_comparison_table_path)
    best = comparison.sort_values(
        ["reconstruction_task", "country", "representation", "rmse", "n_components"]
    ).groupby(["reconstruction_task", "country", "representation"], as_index=False).first()
    keys = best.loc[
        :,
        ["reconstruction_task", "country", "representation", "n_components"],
    ]
    return errors.merge(keys, on=["reconstruction_task", "country", "representation", "n_components"])


def _residual_rv_by_regime(config: ProjectConfig) -> pd.DataFrame:
    rows = [
        _residual_rv_rows(
            path=config.residual_rv_by_macro_regime_table_path,
            regime_type="macro",
            regime_column="macro_regime",
        ),
        _residual_rv_rows(
            path=config.residual_rv_by_market_regime_table_path,
            regime_type="market",
            regime_column="market_vol_regime",
        ),
    ]
    return pd.concat([row for row in rows if not row.empty], ignore_index=True)


def _residual_rv_rows(path: Path, regime_type: str, regime_column: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    frame = frame.rename(columns={regime_column: "regime"})
    frame["evidence_type"] = "residual_relative_value"
    frame["representation"] = "nelson_siegel_residual"
    frame["regime_type"] = regime_type
    metrics = [
        ("convergence_hit_rate", "convergence_hit_rate"),
        ("mean_rank_ic", "rank_ic"),
    ]
    rows = [
        _metric_rows(frame, metric_column=column, metric_name=name)
        for column, name in metrics
        if column in frame.columns
    ]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=SUMMARY_COLUMNS)


def _volatility_classification_summary(config: ProjectConfig) -> pd.DataFrame:
    if not config.volatility_regime_benchmark_table_path.exists():
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    frame = pd.read_csv(config.volatility_regime_benchmark_table_path)
    if frame.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    rows = []
    metric_map = {
        "curve_vol": "curve_vol_balanced_accuracy",
        "policy": "policy_balanced_accuracy",
        "curve": "curve_balanced_accuracy",
        "autoencoder": "autoencoder_balanced_accuracy",
        "transformer": "transformer_balanced_accuracy",
    }
    for representation, column in metric_map.items():
        if column not in frame.columns:
            continue
        subset = frame.loc[:, ["country", "horizon_days", column]].copy()
        subset["evidence_type"] = "volatility_regime_classification"
        subset["representation"] = representation
        subset["regime_type"] = "target"
        subset["indicator"] = "curve_volatility"
        subset["regime"] = "all"
        subset["observations"] = np.nan
        subset["dates"] = np.nan
        rows.append(_metric_rows(subset, metric_column=column, metric_name="balanced_accuracy"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=SUMMARY_COLUMNS)


def _learned_state_separation_summary(config: ProjectConfig) -> pd.DataFrame:
    if not config.learned_state_regime_summary_table_path.exists():
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    frame = pd.read_csv(config.learned_state_regime_summary_table_path)
    if frame.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    frame["evidence_type"] = "learned_state_separation"
    frame["regime"] = "high_vs_low"
    frame["horizon_days"] = np.nan
    rows = [
        _metric_rows(
            frame,
            metric_column="separation_ratio",
            metric_name="separation_ratio",
        ),
        _metric_rows(
            frame,
            metric_column="high_low_distance",
            metric_name="high_low_distance",
        ),
    ]
    return pd.concat(rows, ignore_index=True)


def _joined_regimes(frame: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    rows = [
        _join_macro_regimes(frame, config),
        _join_market_regimes(frame, config),
    ]
    non_empty = [row for row in rows if not row.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


def _join_macro_regimes(frame: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    if not config.macro_regimes_path.exists():
        return pd.DataFrame()

    macro = pd.read_parquet(config.macro_regimes_path)
    if macro.empty:
        return pd.DataFrame()

    rows = []
    left = frame.copy()
    left["date"] = pd.to_datetime(left["date"])
    for (country, indicator), regime in macro.groupby(["country", "indicator"], sort=True):
        country_frame = left.loc[left["country"] == country].sort_values("date")
        if country_frame.empty:
            continue

        regime_dates = regime.sort_values("date").copy()
        regime_dates["date"] = pd.to_datetime(regime_dates["date"])
        joined = pd.merge_asof(
            country_frame,
            regime_dates.loc[:, ["date", "indicator", "macro_regime"]],
            on="date",
            direction="backward",
        ).dropna(subset=["macro_regime"])
        if joined.empty:
            continue

        joined["regime_type"] = "macro"
        joined["indicator"] = str(indicator)
        joined["regime"] = joined["macro_regime"]
        rows.append(joined.drop(columns=["macro_regime"]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _join_market_regimes(frame: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    if not config.market_regimes_path.exists():
        return pd.DataFrame()

    market = pd.read_parquet(config.market_regimes_path)
    if market.empty:
        return pd.DataFrame()

    rows = []
    left = frame.copy()
    left["date"] = pd.to_datetime(left["date"])
    for indicator, regime in market.groupby("indicator", sort=True):
        regime_dates = regime.sort_values("date").copy()
        regime_dates["date"] = pd.to_datetime(regime_dates["date"])
        joined = pd.merge_asof(
            left.sort_values("date"),
            regime_dates.loc[:, ["date", "indicator", "market_vol_regime"]],
            on="date",
            direction="backward",
        ).dropna(subset=["market_vol_regime"])
        if joined.empty:
            continue

        joined["regime_type"] = "market"
        joined["indicator"] = str(indicator)
        joined["regime"] = joined["market_vol_regime"]
        rows.append(joined.drop(columns=["market_vol_regime"]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _metric_rows(frame: pd.DataFrame, metric_column: str, metric_name: str) -> pd.DataFrame:
    rows = frame.copy()
    rows["metric"] = metric_name
    rows["value"] = rows[metric_column]
    for column in SUMMARY_COLUMNS:
        if column not in rows.columns:
            rows[column] = np.nan
    return rows.loc[:, SUMMARY_COLUMNS]
