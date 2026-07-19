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
    by_maturity = pd.read_csv(config.reconstruction_by_maturity_table_path)

    component_path = config.figures_dir / "reconstruction_rmse_by_component.html"
    comparison_path = config.figures_dir / "reconstruction_rmse_comparison.html"
    maturity_path = config.figures_dir / "reconstruction_rmse_by_maturity.html"

    _plot_pca_components(summary).write_html(component_path)
    _plot_representation_comparison(summary).write_html(comparison_path)
    _plot_maturity_profile(by_maturity).write_html(maturity_path)

    return [component_path, comparison_path, maturity_path]


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
    comparison = _comparison_rows(summary)
    return px.bar(
        comparison,
        x="country",
        y="rmse",
        color="representation_label",
        barmode="group",
        title="Classical reconstruction RMSE comparison",
        labels={"rmse": "Reconstruction RMSE", "representation_label": "Representation"},
    )


def _plot_maturity_profile(by_maturity: pd.DataFrame) -> Any:
    comparison = _comparison_rows(by_maturity)
    return px.line(
        comparison,
        x="maturity_years",
        y="rmse",
        color="representation_label",
        facet_col="country",
        markers=True,
        title="Reconstruction RMSE by maturity",
        labels={"maturity_years": "Maturity years", "rmse": "Reconstruction RMSE"},
    )


def _comparison_rows(data: pd.DataFrame) -> pd.DataFrame:
    pca = data.loc[data["representation"] == "pca"].copy()
    if not pca.empty:
        pca = pca.loc[pca["n_components"] == pca["n_components"].max()]

    nelson_siegel = data.loc[data["representation"] == "nelson_siegel"].copy()
    comparison = pd.concat([pca, nelson_siegel], ignore_index=True)
    comparison["representation_label"] = comparison.apply(_representation_label, axis=1)
    return comparison


def _representation_label(row: pd.Series) -> str:
    if row["representation"] == "pca":
        return f"PCA {int(row['n_components'])} components"
    return "Nelson-Siegel"
