from __future__ import annotations

from pathlib import Path

import pandas as pd

from yieldrep.config import ProjectConfig

SUMMARY_GROUP_COLUMNS = ["target", "representation", "model"]
BUCKET_GROUP_COLUMNS = ["target", "representation", "model", "maturity_bucket"]
RANK_GROUP_COLUMNS = ["target", "country", "horizon_days"]
METRIC_COLUMNS = ["rmse", "mae", "directional_accuracy"]


def summarize_baselines(config: ProjectConfig, top_n: int = 100) -> list[Path]:
    """Write human-readable CSV summaries from baseline metric parquets."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_parquet(config.baseline_metrics_path)

    summary = summarize_metrics(metrics)
    summary.to_csv(config.baseline_summary_table_path, index=False)

    rank_table = rank_baselines(metrics)
    rank_table.to_csv(config.baseline_rank_table_path, index=False)

    winners = baseline_winners(rank_table)
    winners.to_csv(config.baseline_winners_table_path, index=False)

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
        config.baseline_rank_table_path,
        config.baseline_winners_table_path,
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


def rank_baselines(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank baseline representations within each target/country/horizon task."""
    summary = (
        metrics.groupby([*RANK_GROUP_COLUMNS, "representation", "model"], sort=True)
        .agg(
            rows=("rmse", "size"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_directional_accuracy=("directional_accuracy", "mean"),
            mean_test_dates=("test_dates", "mean"),
        )
        .reset_index()
    )
    summary["rank"] = summary.groupby(RANK_GROUP_COLUMNS)["mean_rmse"].rank(
        method="min",
        ascending=True,
    )
    best_rmse = summary.groupby(RANK_GROUP_COLUMNS)["mean_rmse"].transform("min")
    summary["rmse_gap_to_best"] = summary["mean_rmse"] - best_rmse
    summary["pct_gap_to_best"] = summary["rmse_gap_to_best"] / best_rmse
    return summary.sort_values([*RANK_GROUP_COLUMNS, "rank", "mean_mae"]).reset_index(drop=True)


def baseline_winners(rank_table: pd.DataFrame) -> pd.DataFrame:
    """Create a compact winner table with PCA and lagged gaps to best."""
    rows: list[dict[str, object]] = []
    for group_values, group in rank_table.groupby(RANK_GROUP_COLUMNS, sort=True):
        keys = dict(zip(RANK_GROUP_COLUMNS, group_values, strict=True))
        best = group.sort_values(["rank", "mean_mae", "representation", "model"]).iloc[0]
        pca = _best_representation_row(group, "pca")
        lagged = _best_representation_row(group, "lagged")
        rows.append(
            {
                **keys,
                "best_representation": best["representation"],
                "best_model": best["model"],
                "best_rmse": best["mean_rmse"],
                "best_mae": best["mean_mae"],
                "pca_rank": _rank_value(pca),
                "pca_rmse_gap_to_best": _gap_value(pca),
                "pca_pct_gap_to_best": _pct_gap_value(pca),
                "lagged_rank": _rank_value(lagged),
                "lagged_rmse_gap_to_best": _gap_value(lagged),
                "lagged_pct_gap_to_best": _pct_gap_value(lagged),
            }
        )
    return pd.DataFrame(rows)


def _best_representation_row(group: pd.DataFrame, representation: str) -> pd.Series | None:
    rows = group.loc[group["representation"] == representation]
    if rows.empty:
        return None
    return rows.sort_values(["rank", "mean_mae", "model"]).iloc[0]


def _rank_value(row: pd.Series | None) -> float | None:
    return None if row is None else float(row["rank"])


def _gap_value(row: pd.Series | None) -> float | None:
    return None if row is None else float(row["rmse_gap_to_best"])


def _pct_gap_value(row: pd.Series | None) -> float | None:
    return None if row is None else float(row["pct_gap_to_best"])
