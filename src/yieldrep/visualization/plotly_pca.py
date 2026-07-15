from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px

from yieldrep.config import ProjectConfig


def plot_pca(config: ProjectConfig) -> list[Path]:
    """Write Plotly HTML figures for PCA variance and score time series."""
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for variance_path in sorted(config.pca_dir.glob("*_variance.parquet")):
        country = variance_path.name.removesuffix("_variance.parquet")
        output_paths.extend(_plot_country_pca(config, country, variance_path))

    return output_paths


def _plot_country_pca(config: ProjectConfig, country: str, variance_path: Path) -> list[Path]:
    scores_path = config.pca_dir / f"{country}_scores.parquet"
    variance = pd.read_parquet(variance_path)
    scores = pd.read_parquet(scores_path)

    variance_html = config.figures_dir / f"{country}_pca_explained_variance.html"
    scores_html = config.figures_dir / f"{country}_pca_scores.html"

    variance_fig = px.bar(
        variance,
        x="component",
        y="explained_variance_ratio",
        title=f"{country.upper()} PCA explained variance",
        labels={"explained_variance_ratio": "Explained variance ratio"},
    )
    variance_fig.write_html(variance_html)

    score_columns = [column for column in ["PC1", "PC2", "PC3"] if column in scores.columns]
    scores_long = scores.melt(id_vars=["date"], value_vars=score_columns, var_name="component")
    scores_fig = px.line(
        scores_long,
        x="date",
        y="value",
        color="component",
        title=f"{country.upper()} PCA scores",
        labels={"value": "Score", "date": "Date"},
    )
    scores_fig.write_html(scores_html)

    return [variance_html, scores_html]
