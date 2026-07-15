from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px

from yieldrep.config import ProjectConfig


def plot_nelson_siegel(config: ProjectConfig) -> list[Path]:
    """Write Plotly HTML figures for Nelson-Siegel factors and fit error."""
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for factors_path in sorted(config.nelson_siegel_dir.glob("*_factors.parquet")):
        country = factors_path.name.removesuffix("_factors.parquet")
        output_paths.extend(_plot_country_nelson_siegel(config, country, factors_path))

    return output_paths


def _plot_country_nelson_siegel(
    config: ProjectConfig,
    country: str,
    factors_path: Path,
) -> list[Path]:
    factors = pd.read_parquet(factors_path)
    factor_columns = ["beta_level", "beta_slope", "beta_curvature"]
    factors_long = factors.melt(
        id_vars=["date"],
        value_vars=factor_columns,
        var_name="factor",
        value_name="beta",
    )

    factors_html = config.figures_dir / f"{country}_nelson_siegel_factors.html"
    rmse_html = config.figures_dir / f"{country}_nelson_siegel_rmse.html"

    factors_fig = px.line(
        factors_long,
        x="date",
        y="beta",
        color="factor",
        title=f"{country.upper()} Nelson-Siegel factors",
        labels={"date": "Date", "beta": "Beta", "factor": "Factor"},
    )
    factors_fig.write_html(factors_html)

    rmse_fig = px.line(
        factors,
        x="date",
        y="rmse",
        title=f"{country.upper()} Nelson-Siegel fit RMSE",
        labels={"date": "Date", "rmse": "RMSE"},
    )
    rmse_fig.write_html(rmse_html)

    return [factors_html, rmse_html]
