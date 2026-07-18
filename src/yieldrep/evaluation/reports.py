from __future__ import annotations

from pathlib import Path

import pandas as pd

from yieldrep.config import ProjectConfig

SUMMARY_GROUP_COLUMNS = ["target", "representation", "model"]
BUCKET_GROUP_COLUMNS = ["target", "representation", "model", "maturity_bucket"]
METRIC_COLUMNS = ["rmse", "mae", "directional_accuracy"]


def summarize_baselines(config: ProjectConfig, top_n: int = 100) -> list[Path]:
    """Write human-readable CSV summaries from baseline metric parquets."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_metrics(pd.read_parquet(config.baseline_metrics_path))
    summary.to_csv(config.baseline_summary_table_path, index=False)

    bucket_summary = summarize_metrics(
        pd.read_parquet(config.baseline_metrics_by_maturity_path),
        group_columns=BUCKET_GROUP_COLUMNS,
    )
    bucket_summary.to_csv(config.baseline_by_maturity_bucket_table_path, index=False)

    maturity_point_top = top_maturity_point_metrics(
        pd.read_parquet(config.baseline_metrics_by_maturity_point_path),
        top_n=top_n,
    )
    maturity_point_top.to_csv(config.baseline_by_maturity_point_top_table_path, index=False)

    return [
        config.baseline_summary_table_path,
        config.baseline_by_maturity_bucket_table_path,
        config.baseline_by_maturity_point_top_table_path,
    ]


def summarize_metrics(
    metrics: pd.DataFrame,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate metric rows into compact mean-performance tables."""
    groups = group_columns or SUMMARY_GROUP_COLUMNS
    summary = (
        metrics.groupby(groups, sort=True)
        .agg(
            rows=("rmse", "size"),
            countries=("country", "nunique"),
            horizons=("horizon_days", "nunique"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_directional_accuracy=("directional_accuracy", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values([*groups, "mean_rmse"]).reset_index(drop=True)


def top_maturity_point_metrics(metrics: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """Return the best exact-maturity metric rows ranked by RMSE."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    columns = [
        "target",
        "representation",
        "model",
        "split_method",
        "window_id",
        "country",
        "horizon_days",
        "maturity_years",
        *METRIC_COLUMNS,
        "train_rows",
        "test_rows",
        "train_dates",
        "test_dates",
    ]
    available_columns = [column for column in columns if column in metrics.columns]
    return (
        metrics.sort_values(["rmse", "mae", "target", "representation", "model"])
        .loc[:, available_columns]
        .head(top_n)
        .reset_index(drop=True)
    )
