from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px

from yieldrep.config import ProjectConfig


def plot_learned_state_regimes(config: ProjectConfig) -> list[Path]:
    """Write Plotly diagnostics for learned states across macro/market regimes."""
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    if config.learned_state_regime_summary_table_path.exists():
        summary = pd.read_csv(config.learned_state_regime_summary_table_path)
        _plot_regime_separation_heatmap(summary).write_html(
            config.learned_state_regime_heatmap_figure_path
        )
        output_paths.append(config.learned_state_regime_heatmap_figure_path)

    if config.learned_state_regimes_path.exists():
        regimes = pd.read_parquet(config.learned_state_regimes_path)
        _plot_state_space(regimes).write_html(config.learned_state_space_figure_path)
        output_paths.append(config.learned_state_space_figure_path)

    return output_paths


def _plot_regime_separation_heatmap(summary: pd.DataFrame) -> Any:
    if summary.empty:
        return px.imshow([[0.0]], title="Learned state regime separation")

    frame = summary.copy()
    frame["row"] = frame["representation"] + " / " + frame["country"]
    frame["column"] = frame["regime_type"] + ": " + frame["indicator"]
    pivot = frame.pivot_table(
        index="row",
        columns="column",
        values="separation_ratio",
        aggfunc="mean",
    ).sort_index()
    return px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Learned state regime separation",
        labels={
            "x": "Regime indicator",
            "y": "Representation / country",
            "color": "Between / within variance",
        },
    )


def _plot_state_space(regimes: pd.DataFrame) -> Any:
    frame = _state_space_frame(regimes)
    if frame.empty:
        return px.scatter(title="Learned state space by regime")

    return px.scatter(
        frame,
        x="x",
        y="y",
        color="regime",
        facet_col="country",
        facet_row="representation_indicator",
        opacity=0.45,
        title="Learned state space by selected regimes",
        labels={
            "x": "First latent dimension",
            "y": "Second latent dimension",
            "regime": "Regime",
            "representation_indicator": "State / regime",
        },
    )


def _state_space_frame(regimes: pd.DataFrame) -> pd.DataFrame:
    if regimes.empty:
        return pd.DataFrame()

    selected = regimes.loc[
        ((regimes["regime_type"] == "macro") & (regimes["indicator"] == "inflation"))
        | ((regimes["regime_type"] == "market") & (regimes["indicator"] == "MOVE"))
    ].copy()
    if selected.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for representation, group in selected.groupby("representation", sort=True):
        features = [
            column
            for column in group.columns
            if column.startswith("AE") or column.startswith("TE") or column.startswith("GE")
        ]
        if len(features) < 2:
            continue
        frame = group.loc[
            :,
            [
                "date",
                "country",
                "split",
                "representation",
                "regime_type",
                "indicator",
                "regime",
                features[0],
                features[1],
            ],
        ].copy()
        frame = frame.rename(columns={features[0]: "x", features[1]: "y"})
        frame["representation_indicator"] = representation + " / " + frame["indicator"]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
