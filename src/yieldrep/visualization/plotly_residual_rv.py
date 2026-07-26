from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from yieldrep.config import ProjectConfig


def plot_residual_zscores(config: ProjectConfig) -> list[Path]:
    """Plot recent Nelson-Siegel residual z-scores for selected maturities."""
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(config.residual_features_path)
    figure = _plot_residual_zscores(features, config.plots.selected_maturities)
    figure.write_html(config.residual_zscores_figure_path)
    return [config.residual_zscores_figure_path]


def plot_residual_rv_regime_scorecard(config: ProjectConfig) -> list[Path]:
    """Plot RV regime scorecard high-minus-low hit-rate as a heatmap."""
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    scorecard = pd.read_csv(config.residual_rv_regime_scorecard_table_path)
    figure = _plot_residual_rv_regime_heatmap(scorecard)
    figure.write_html(config.residual_rv_regime_heatmap_figure_path)
    return [config.residual_rv_regime_heatmap_figure_path]


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


def _plot_residual_rv_regime_heatmap(scorecard: pd.DataFrame) -> Any:
    if scorecard.empty:
        return go.Figure()

    frame = scorecard.copy()
    frame["row"] = frame["country"].astype(str) + " " + frame["horizon_days"].astype(str) + "d"
    frame["column"] = frame["regime_type"].astype(str) + ": " + frame["indicator"].astype(str)
    frame = frame.sort_values(["country", "horizon_days", "regime_type", "indicator"])

    rows = frame["row"].drop_duplicates().tolist()
    columns = frame["column"].drop_duplicates().tolist()
    z = frame.pivot(index="row", columns="column", values="high_minus_low_hit_rate").reindex(
        index=rows,
        columns=columns,
    )
    hover = _regime_heatmap_hover(frame, rows, columns)

    figure = go.Figure(
        data=go.Heatmap(
            z=z.to_numpy(),
            x=columns,
            y=rows,
            customdata=hover,
            colorscale="RdBu",
            zmid=0.0,
            colorbar={"title": "High minus low hit rate"},
            hovertemplate=(
                "Scenario=%{y}<br>"
                "Regime=%{x}<br>"
                "High-low hit rate=%{z:.3f}<br>"
                "%{customdata}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Residual RV regime scorecard",
        xaxis_title="Regime indicator",
        yaxis_title="Country / horizon",
    )
    return figure


def _regime_heatmap_hover(
    frame: pd.DataFrame,
    rows: list[str],
    columns: list[str],
) -> list[list[str]]:
    lookup = {
        (row["row"], row["column"]): (
            f"Best regime={row['best_regime']}<br>"
            f"Best hit rate={row['best_hit_rate']:.3f}<br>"
            f"Best rank IC={row['best_rank_ic']:.3f}<br>"
            f"Interpretation={row['interpretation']}"
        )
        for _, row in frame.iterrows()
    }
    return [[lookup.get((row, column), "") for column in columns] for row in rows]
