from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px

from yieldrep.config import ProjectConfig


def plot_residual_zscores(config: ProjectConfig) -> list[Path]:
    """Plot recent Nelson-Siegel residual z-scores for selected maturities."""
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(config.residual_features_path)
    figure = _plot_residual_zscores(features, config.plots.selected_maturities)
    figure.write_html(config.residual_zscores_figure_path)
    return [config.residual_zscores_figure_path]


def _plot_residual_zscores(features: pd.DataFrame, selected_maturities: list[float]) -> Any:
    frame = features.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    max_date = frame["date"].max()
    frame = frame.loc[frame["date"] >= max_date - pd.DateOffset(years=5)]
    frame = frame.loc[frame["maturity_years"].isin(selected_maturities)]
    frame["maturity"] = frame["maturity_years"].map(lambda value: f"{value:g}Y")

    return px.line(
        frame,
        x="date",
        y="residual_z_252",
        color="maturity",
        facet_col="country",
        title="Nelson-Siegel residual z-scores",
        labels={
            "date": "Date",
            "residual_z_252": "252-day residual z-score",
            "maturity": "Maturity",
        },
    )
