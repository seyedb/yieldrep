from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px

from yieldrep.config import ProjectConfig
from yieldrep.visualization.plotly_residual_rv import plot_residual_zscores


def plot_baseline_metrics(config: ProjectConfig) -> list[Path]:
    """Write Plotly HTML figures for baseline evaluation metrics."""
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_parquet(config.baseline_metrics_path)
    bucket_metrics = pd.read_parquet(config.baseline_metrics_by_maturity_path)
    point_metrics = pd.read_parquet(config.baseline_metrics_by_maturity_point_path)

    summary_path = config.figures_dir / "baseline_rmse_summary.html"
    bucket_path = config.figures_dir / "baseline_rmse_by_maturity_bucket.html"
    point_path = config.figures_dir / "baseline_rmse_by_maturity_point.html"

    _plot_summary(metrics).write_html(summary_path)
    _plot_maturity_buckets(bucket_metrics).write_html(bucket_path)
    _plot_maturity_points(point_metrics).write_html(point_path)

    output_paths = [summary_path, bucket_path, point_path]
    output_paths.extend(_plot_supervised_metrics(config))
    if config.residual_features_path.exists():
        output_paths.extend(plot_residual_zscores(config))
    return output_paths


def _plot_summary(metrics: pd.DataFrame) -> Any:
    summary = (
        metrics.groupby(["target", "representation", "model"], sort=True)["rmse"]
        .mean()
        .reset_index()
    )
    return px.bar(
        summary,
        x="representation",
        y="rmse",
        color="model",
        facet_col="target",
        barmode="group",
        title="Baseline RMSE by representation",
        labels={"rmse": "Mean RMSE", "representation": "Representation"},
    )


def _plot_maturity_buckets(metrics: pd.DataFrame) -> Any:
    bucket_summary = (
        metrics.groupby(["target", "maturity_bucket", "representation", "model"], sort=True)[
            "rmse"
        ]
        .mean()
        .reset_index()
    )
    return px.bar(
        bucket_summary,
        x="maturity_bucket",
        y="rmse",
        color="representation",
        facet_col="target",
        facet_row="model",
        barmode="group",
        title="Baseline RMSE by maturity bucket",
        labels={"rmse": "Mean RMSE", "maturity_bucket": "Maturity bucket"},
    )


def _plot_maturity_points(metrics: pd.DataFrame) -> Any:
    point_summary = (
        metrics.loc[metrics["model"] == "ridge"]
        .groupby(["target", "country", "maturity_years", "representation"], sort=True)["rmse"]
        .mean()
        .reset_index()
    )
    return px.line(
        point_summary,
        x="maturity_years",
        y="rmse",
        color="representation",
        facet_col="target",
        line_dash="country",
        markers=True,
        title="Ridge RMSE by maturity point",
        labels={"rmse": "Mean RMSE", "maturity_years": "Maturity years"},
    )


def _plot_supervised_metrics(config: ProjectConfig) -> list[Path]:
    output_paths: list[Path] = []

    if config.supervised_forecast_metrics_path.exists():
        metrics = pd.read_parquet(config.supervised_forecast_metrics_path)
        rmse_path = config.figures_dir / "supervised_rmse_by_target.html"
        improvement_path = config.figures_dir / "supervised_improvement_vs_mean.html"
        _plot_supervised_rmse(metrics).write_html(rmse_path)
        _plot_supervised_improvement(metrics).write_html(improvement_path)
        output_paths.extend([rmse_path, improvement_path])

    if config.supervised_walk_forward_comparison_table_path.exists():
        comparison = pd.read_csv(config.supervised_walk_forward_comparison_table_path)
        walk_forward_path = config.figures_dir / "supervised_walk_forward_rank_changes.html"
        _plot_walk_forward_rank_changes(comparison).write_html(walk_forward_path)
        output_paths.append(walk_forward_path)

    if config.supervised_forecast_coefficients_path.exists():
        coefficients = pd.read_parquet(config.supervised_forecast_coefficients_path)
        coefficient_path = config.figures_dir / "supervised_coefficient_importance.html"
        _plot_coefficient_importance(coefficients).write_html(coefficient_path)
        output_paths.append(coefficient_path)

    return output_paths


def _plot_supervised_rmse(metrics: pd.DataFrame) -> Any:
    summary = (
        metrics.groupby(["target", "representation", "model"], sort=True)["rmse"]
        .mean()
        .reset_index()
    )
    return px.bar(
        summary,
        x="representation",
        y="rmse",
        color="model",
        facet_col="target",
        barmode="group",
        title="Supervised forecast RMSE by target",
        labels={"rmse": "Mean RMSE", "representation": "Representation"},
    )


def _plot_supervised_improvement(metrics: pd.DataFrame) -> Any:
    summary = (
        metrics.loc[metrics["model"] != "train_mean"]
        .groupby(["target", "representation", "model"], sort=True)[
            "pct_improvement_vs_train_mean"
        ]
        .mean()
        .reset_index()
    )
    return px.bar(
        summary,
        x="representation",
        y="pct_improvement_vs_train_mean",
        color="model",
        facet_col="target",
        barmode="group",
        title="Supervised RMSE improvement versus train mean",
        labels={
            "pct_improvement_vs_train_mean": "Mean fractional RMSE improvement",
            "representation": "Representation",
        },
    )


def _plot_walk_forward_rank_changes(comparison: pd.DataFrame) -> Any:
    frame = comparison.dropna(
        subset=["rank_change_walk_forward_minus_date_ordered", "walk_forward_rank"]
    )
    return px.bar(
        frame,
        x="representation",
        y="rank_change_walk_forward_minus_date_ordered",
        color="model",
        facet_col="target",
        facet_row="horizon_days",
        barmode="group",
        title="Walk-forward rank change versus date-ordered split",
        labels={
            "rank_change_walk_forward_minus_date_ordered": "Rank change",
            "representation": "Representation",
        },
    )


def _plot_coefficient_importance(coefficients: pd.DataFrame, top_n: int = 10) -> Any:
    summary = (
        coefficients.loc[coefficients["model"].isin(["ridge", "elastic_net"])]
        .groupby(["target", "representation", "model", "feature"], sort=True)[
            "abs_coefficient"
        ]
        .mean()
        .reset_index()
        .sort_values(["target", "model", "abs_coefficient"], ascending=[True, True, False])
        .groupby(["target", "model"], sort=False)
        .head(top_n)
    )
    return px.bar(
        summary,
        x="abs_coefficient",
        y="feature",
        color="representation",
        facet_col="target",
        facet_row="model",
        orientation="h",
        title="Top standardized coefficient magnitudes",
        labels={"abs_coefficient": "Mean absolute coefficient", "feature": "Feature"},
    )
