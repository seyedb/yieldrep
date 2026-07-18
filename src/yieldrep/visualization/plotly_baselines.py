from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px

from yieldrep.config import ProjectConfig


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

    return [summary_path, bucket_path, point_path]


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
