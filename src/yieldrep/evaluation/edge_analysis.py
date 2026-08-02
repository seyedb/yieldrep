from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from yieldrep.config import ProjectConfig


EDGE_COLUMNS = [
    "scenario",
    "country",
    "horizon_days",
    "maturity_bucket",
    "regime_type",
    "indicator",
    "regime",
    "ae_metric_value",
    "benchmark_representation",
    "benchmark_model",
    "benchmark_metric_value",
    "ae_gap_to_benchmark",
    "ae_pct_gap_to_benchmark",
    "coverage_dates",
    "edge_label",
    "edge_breadth_label",
]

MATERIAL_RMSE_EDGE = 0.005


def build_autoencoder_edge_analysis(config: ProjectConfig) -> Path:
    """Write granular diagnostics for the current autoencoder-positive scenarios."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    table = autoencoder_edge_analysis(config)
    table.to_csv(config.autoencoder_edge_analysis_table_path, index=False)
    return config.autoencoder_edge_analysis_table_path


def autoencoder_edge_analysis(config: ProjectConfig) -> pd.DataFrame:
    rows = [
        _standardized_forecast_edges(config),
        _masked_reconstruction_edges(config),
        _masked_regime_edges(config),
    ]
    non_empty = [row for row in rows if not row.empty]
    if not non_empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    table = pd.concat(non_empty, ignore_index=True)
    table = _add_edge_breadth_labels(table)
    return table.loc[:, EDGE_COLUMNS].sort_values(
        [
            "scenario",
            "country",
            "horizon_days",
            "maturity_bucket",
            "regime_type",
            "indicator",
            "regime",
        ],
        na_position="last",
    ).reset_index(drop=True)


def _standardized_forecast_edges(config: ProjectConfig) -> pd.DataFrame:
    path = config.baseline_by_maturity_bucket_table_path
    if not path.exists():
        return pd.DataFrame(columns=EDGE_COLUMNS)

    metrics = pd.read_csv(path)
    metrics = metrics.loc[
        (metrics["target"] == "standardized_yield_change")
        & (metrics["model"] != "train_mean")
    ].copy()
    if metrics.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    ae = metrics.loc[
        (metrics["representation"] == "autoencoder") & (metrics["model"] == "ridge")
    ].copy()
    classical = metrics.loc[~metrics["representation"].isin(["autoencoder", "transformer"])].copy()
    if ae.empty or classical.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    group_columns = ["country", "horizon_days", "maturity_bucket"]
    best_classical = (
        classical.sort_values([*group_columns, "mean_rmse", "mean_mae", "representation", "model"])
        .groupby(group_columns, as_index=False)
        .first()
    )
    joined = ae.merge(best_classical, on=group_columns, suffixes=("_ae", "_benchmark"))
    rows = pd.DataFrame(
        {
            "scenario": "standardized_yield_change_forecasting",
            "country": joined["country"],
            "horizon_days": joined["horizon_days"],
            "maturity_bucket": joined["maturity_bucket"],
            "regime_type": np.nan,
            "indicator": np.nan,
            "regime": np.nan,
            "ae_metric_value": joined["mean_rmse_ae"],
            "benchmark_representation": joined["representation_benchmark"],
            "benchmark_model": joined["model_benchmark"],
            "benchmark_metric_value": joined["mean_rmse_benchmark"],
            "coverage_dates": joined["rank_ic_dates_ae"].fillna(0),
        }
    )
    return _add_lower_is_better_edge_labels(rows)


def _masked_reconstruction_edges(config: ProjectConfig) -> pd.DataFrame:
    path = config.masked_reconstruction_by_maturity_bucket_table_path
    if not path.exists():
        return pd.DataFrame(columns=EDGE_COLUMNS)

    metrics = pd.read_csv(path)
    if metrics.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    ae = metrics.loc[metrics["representation"] == "masked_autoencoder"].copy()
    transformer = metrics.loc[metrics["representation"] == "masked_transformer"].copy()
    if ae.empty or transformer.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    group_columns = ["country", "maturity_bucket"]
    joined = ae.merge(transformer, on=group_columns, suffixes=("_ae", "_benchmark"))
    rows = pd.DataFrame(
        {
            "scenario": "masked_maturity_reconstruction",
            "country": joined["country"],
            "horizon_days": np.nan,
            "maturity_bucket": joined["maturity_bucket"],
            "regime_type": np.nan,
            "indicator": np.nan,
            "regime": np.nan,
            "ae_metric_value": joined["rmse_ae"],
            "benchmark_representation": "masked_transformer",
            "benchmark_model": "n_components=5",
            "benchmark_metric_value": joined["rmse_benchmark"],
            "coverage_dates": joined["dates_ae"],
        }
    )
    return _add_lower_is_better_edge_labels(rows)


def _masked_regime_edges(config: ProjectConfig) -> pd.DataFrame:
    path = config.macro_conditioned_representation_summary_table_path
    if not path.exists():
        return pd.DataFrame(columns=EDGE_COLUMNS)

    metrics = pd.read_csv(path)
    metrics = metrics.loc[
        (metrics["evidence_type"] == "masked_maturity_reconstruction")
        & (metrics["metric"] == "rmse")
        & (metrics["representation"].isin(["masked_autoencoder", "masked_transformer"]))
    ].copy()
    if metrics.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    group_columns = ["country", "regime_type", "indicator", "regime"]
    ae = metrics.loc[metrics["representation"] == "masked_autoencoder"].copy()
    transformer = metrics.loc[metrics["representation"] == "masked_transformer"].copy()
    joined = ae.merge(transformer, on=group_columns, suffixes=("_ae", "_benchmark"))
    if joined.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    rows = pd.DataFrame(
        {
            "scenario": "masked_maturity_reconstruction_by_regime",
            "country": joined["country"],
            "horizon_days": np.nan,
            "maturity_bucket": np.nan,
            "regime_type": joined["regime_type"],
            "indicator": joined["indicator"],
            "regime": joined["regime"],
            "ae_metric_value": joined["value_ae"],
            "benchmark_representation": "masked_transformer",
            "benchmark_model": "masked reconstruction",
            "benchmark_metric_value": joined["value_benchmark"],
            "coverage_dates": joined["dates_ae"],
        }
    )
    return _add_lower_is_better_edge_labels(rows)


def _add_lower_is_better_edge_labels(rows: pd.DataFrame) -> pd.DataFrame:
    frame = rows.copy()
    frame["ae_gap_to_benchmark"] = frame["ae_metric_value"] - frame["benchmark_metric_value"]
    frame["ae_pct_gap_to_benchmark"] = frame["ae_gap_to_benchmark"] / frame[
        "benchmark_metric_value"
    ].abs()
    frame["edge_label"] = np.select(
        [
            frame["ae_pct_gap_to_benchmark"] <= -MATERIAL_RMSE_EDGE,
            frame["ae_pct_gap_to_benchmark"].abs() <= MATERIAL_RMSE_EDGE,
        ],
        ["material_ae_edge", "competitive_tie"],
        default="no_ae_edge",
    )
    return frame


def _add_edge_breadth_labels(table: pd.DataFrame) -> pd.DataFrame:
    frame = table.copy()
    edge_rates = (
        frame["edge_label"].eq("material_ae_edge").groupby(frame["scenario"]).mean()
    )
    frame["edge_breadth_label"] = frame["scenario"].map(
        lambda scenario: _edge_breadth_label(float(edge_rates.loc[scenario]))
    )
    return frame


def _edge_breadth_label(material_edge_rate: float) -> str:
    if material_edge_rate >= 0.60:
        return "broad"
    if material_edge_rate >= 0.20:
        return "localized"
    return "fragile"
