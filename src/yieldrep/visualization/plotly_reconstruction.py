from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px

from yieldrep.config import ProjectConfig


def plot_reconstruction(config: ProjectConfig) -> list[Path]:
    """Write Plotly HTML figures for reconstruction quality metrics."""
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(config.reconstruction_summary_table_path)
    oos_summary = pd.read_csv(config.reconstruction_oos_summary_table_path)
    oos_by_maturity = pd.read_csv(config.reconstruction_oos_by_maturity_table_path)
    oos_by_bucket = pd.read_csv(config.reconstruction_oos_by_maturity_bucket_table_path)

    component_path = config.figures_dir / "reconstruction_rmse_by_component.html"
    comparison_path = config.figures_dir / "reconstruction_oos_rmse_comparison.html"
    bucket_path = config.figures_dir / "reconstruction_oos_rmse_by_maturity_bucket.html"
    maturity_path = config.figures_dir / "reconstruction_oos_rmse_by_maturity.html"

    _plot_pca_components(summary).write_html(component_path)
    _plot_representation_comparison(oos_summary).write_html(comparison_path)
    _plot_maturity_bucket_profile(oos_by_bucket).write_html(bucket_path)
    _plot_maturity_profile(oos_by_maturity).write_html(maturity_path)

    return [component_path, comparison_path, bucket_path, maturity_path]


def _plot_pca_components(summary: pd.DataFrame) -> Any:
    pca = summary.loc[summary["representation"] == "pca"].copy()
    return px.line(
        pca,
        x="n_components",
        y="rmse",
        color="country",
        markers=True,
        title="PCA reconstruction RMSE by component count",
        labels={"n_components": "PCA components", "rmse": "Reconstruction RMSE"},
    )


def _plot_representation_comparison(summary: pd.DataFrame) -> Any:
    comparison = _clean_comparison_rows(summary)
    return px.bar(
        comparison,
        x="country",
        y="rmse",
        color="representation_label",
        barmode="group",
        title="Out-of-sample curve reconstruction RMSE",
        labels={"rmse": "Reconstruction RMSE", "representation_label": "Representation"},
    )


def _plot_maturity_bucket_profile(by_bucket: pd.DataFrame) -> Any:
    comparison = _clean_comparison_rows(by_bucket)
    return px.bar(
        comparison,
        x="maturity_bucket",
        y="rmse",
        color="representation_label",
        facet_col="country",
        barmode="group",
        title="Out-of-sample reconstruction RMSE by maturity bucket",
        labels={
            "maturity_bucket": "Maturity bucket",
            "rmse": "Reconstruction RMSE",
            "representation_label": "Representation",
        },
    )


def _plot_maturity_profile(by_maturity: pd.DataFrame) -> Any:
    comparison = _clean_comparison_rows(by_maturity)
    return px.line(
        comparison,
        x="maturity_years",
        y="rmse",
        color="representation_label",
        facet_col="country",
        markers=True,
        title="Out-of-sample reconstruction RMSE by maturity",
        labels={"maturity_years": "Maturity years", "rmse": "Reconstruction RMSE"},
    )


def _clean_comparison_rows(data: pd.DataFrame) -> pd.DataFrame:
    clean = data.loc[data["reconstruction_task"] == "clean_reconstruction"].copy()
    pca = clean.loc[clean["representation"] == "pca"].copy()
    if not pca.empty:
        pca = pca.loc[pca["n_components"] == pca["n_components"].max()]

    nelson_siegel = clean.loc[clean["representation"] == "nelson_siegel"].copy()
    autoencoder = clean.loc[clean["representation"] == "autoencoder"].copy()
    comparison = pd.concat([pca, nelson_siegel, autoencoder], ignore_index=True)
    comparison["representation_label"] = comparison.apply(_representation_label, axis=1)
    return comparison


def _representation_label(row: pd.Series) -> str:
    if row["representation"] == "pca":
        return f"PCA {int(row['n_components'])} components"
    if row["representation"] == "autoencoder":
        return f"Autoencoder {int(row['n_components'])} latent dims"
    return "Nelson-Siegel"
