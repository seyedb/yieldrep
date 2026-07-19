from __future__ import annotations

from pathlib import Path

import pandas as pd

from yieldrep.config import ProjectConfig
from yieldrep.models.baselines import evaluate_baseline_frames

SUMMARY_GROUP_COLUMNS = ["target", "representation", "model"]
BUCKET_GROUP_COLUMNS = ["target", "representation", "model", "maturity_bucket"]
RANK_GROUP_COLUMNS = ["target", "country", "horizon_days"]
METRIC_COLUMNS = ["rmse", "mae", "directional_accuracy", "mean_rank_ic", "rank_ic_dates"]


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


def build_overlap_sensitivity_report(config: ProjectConfig) -> Path:
    """Compare baseline ranks with overlapping and non-overlapping target windows."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)

    overlapping = _evaluate_with_target_window(config, non_overlapping_targets=False)
    non_overlapping = _evaluate_with_target_window(config, non_overlapping_targets=True)
    report = overlap_sensitivity_table(overlapping, non_overlapping)
    report.to_csv(config.overlap_sensitivity_table_path, index=False)
    return config.overlap_sensitivity_table_path


def overlap_sensitivity_table(
    overlapping_metrics: pd.DataFrame,
    non_overlapping_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact side-by-side comparison of two evaluation protocols."""
    overlapping = _rank_for_target_window(overlapping_metrics, target_window="overlapping")
    non_overlapping = _rank_for_target_window(
        non_overlapping_metrics,
        target_window="non_overlapping",
    )
    join_columns = [*RANK_GROUP_COLUMNS, "representation", "model"]
    report = overlapping.merge(non_overlapping, on=join_columns, how="outer")
    report["rmse_change_non_overlapping_minus_overlapping"] = (
        report["non_overlapping_mean_rmse"] - report["overlapping_mean_rmse"]
    )
    report["rank_change_non_overlapping_minus_overlapping"] = (
        report["non_overlapping_rank"] - report["overlapping_rank"]
    )
    return report.sort_values(
        [*RANK_GROUP_COLUMNS, "non_overlapping_rank", "overlapping_rank", "representation", "model"],
        na_position="last",
    ).reset_index(drop=True)


def summarize_metrics(
    metrics: pd.DataFrame,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate metric rows into compact mean-performance tables."""
    groups = group_columns or SUMMARY_GROUP_COLUMNS
    aggregations = {
        "rows": ("rmse", "size"),
        "countries": ("country", "nunique"),
        "horizons": ("horizon_days", "nunique"),
        "mean_rmse": ("rmse", "mean"),
        "mean_mae": ("mae", "mean"),
        "mean_directional_accuracy": ("directional_accuracy", "mean"),
    }
    if "mean_rank_ic" in metrics.columns:
        aggregations["mean_rank_ic"] = ("mean_rank_ic", "mean")
    if "rank_ic_dates" in metrics.columns:
        aggregations["rank_ic_dates"] = ("rank_ic_dates", "sum")

    summary = (
        metrics.groupby(groups, sort=True)
        .agg(**aggregations)
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
        .agg(**_rank_aggregations(metrics))
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


def _rank_aggregations(metrics: pd.DataFrame) -> dict[str, tuple[str, str]]:
    aggregations = {
        "rows": ("rmse", "size"),
        "mean_rmse": ("rmse", "mean"),
        "mean_mae": ("mae", "mean"),
        "mean_directional_accuracy": ("directional_accuracy", "mean"),
        "mean_test_dates": ("test_dates", "mean"),
    }
    if "mean_rank_ic" in metrics.columns:
        aggregations["mean_rank_ic"] = ("mean_rank_ic", "mean")
    if "rank_ic_dates" in metrics.columns:
        aggregations["rank_ic_dates"] = ("rank_ic_dates", "sum")
    return aggregations


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


def _evaluate_with_target_window(
    config: ProjectConfig,
    non_overlapping_targets: bool,
) -> pd.DataFrame:
    evaluation = config.evaluation.model_copy(
        update={"non_overlapping_targets": non_overlapping_targets}
    )
    evaluation_config = config.model_copy(update={"evaluation": evaluation})
    return evaluate_baseline_frames(evaluation_config).metrics


def _rank_for_target_window(metrics: pd.DataFrame, target_window: str) -> pd.DataFrame:
    rank_table = rank_baselines(metrics)
    columns = [
        *RANK_GROUP_COLUMNS,
        "representation",
        "model",
        "mean_rmse",
        "mean_mae",
        "mean_directional_accuracy",
        "rank",
        "rmse_gap_to_best",
        "pct_gap_to_best",
    ]
    if "mean_rank_ic" in rank_table.columns:
        columns.append("mean_rank_ic")
    if "rank_ic_dates" in rank_table.columns:
        columns.append("rank_ic_dates")

    renamed = {
        column: f"{target_window}_{column}"
        for column in columns
        if column not in [*RANK_GROUP_COLUMNS, "representation", "model"]
    }
    return rank_table.loc[:, columns].rename(columns=renamed)
