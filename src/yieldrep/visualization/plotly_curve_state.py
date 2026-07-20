from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from yieldrep.config import ProjectConfig

STATE_LABELS = ["low", "medium", "high"]
STATE_CODES = {"low": 0, "medium": 1, "high": 2}


def plot_curve_state(config: ProjectConfig) -> list[Path]:
    """Write Plotly HTML figures for PCA curve-state timelines and transitions."""
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    scores = _read_pca_scores(config)
    components = [column for column in ["PC1", "PC2", "PC3"] if column in scores.columns]
    if not components:
        return []

    state_scores = _add_state_labels(scores, components)
    output_paths: list[Path] = []
    for country, country_scores in state_scores.groupby("country", sort=True):
        country_code = str(country).lower()
        timeline_path = config.figures_dir / f"{country_code}_curve_state_timeline.html"
        _state_timeline(country_scores, str(country), components).write_html(timeline_path)
        output_paths.append(timeline_path)

    if config.curve_state_targets_path.exists():
        targets = pd.read_parquet(config.curve_state_targets_path)
        output_paths.extend(_plot_transition_matrices(config, state_scores, targets, components))

    return output_paths


def _read_pca_scores(config: ProjectConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for scores_path in sorted(config.pca_dir.glob("*_scores.parquet")):
        country = scores_path.name.removesuffix("_scores.parquet").upper()
        scores = pd.read_parquet(scores_path)
        scores["country"] = country
        frames.append(scores)
    if not frames:
        raise FileNotFoundError(f"No PCA score files found in {config.pca_dir}")
    scores = pd.concat(frames, ignore_index=True)
    scores["date"] = pd.to_datetime(scores["date"])
    return scores.sort_values(["country", "date"]).reset_index(drop=True)


def _add_state_labels(scores: pd.DataFrame, components: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in scores.groupby("country", sort=True):
        frame = group.copy()
        for component in components:
            frame[f"{component}_state"] = _tercile_labels(frame[component])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _tercile_labels(values: pd.Series) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=3, labels=STATE_LABELS, duplicates="drop").astype("string")


def _state_timeline(country_scores: pd.DataFrame, country: str, components: list[str]) -> Any:
    z = np.vstack(
        [
            country_scores[f"{component}_state"]
            .map(STATE_CODES)
            .to_numpy(dtype=float, na_value=np.nan)
            for component in components
        ]
    )
    labels = np.vstack(
        [country_scores[f"{component}_state"].astype(str).to_numpy() for component in components]
    )
    fig = go.Figure(
        data=go.Heatmap(
            x=country_scores["date"],
            y=components,
            z=z,
            text=labels,
            hovertemplate="Date=%{x}<br>Component=%{y}<br>State=%{text}<extra></extra>",
            colorscale=[
                [0.0, "#4c78a8"],
                [0.5, "#f2cf5b"],
                [1.0, "#d95f59"],
            ],
            zmin=0,
            zmax=2,
            colorbar={
                "tickmode": "array",
                "tickvals": [0, 1, 2],
                "ticktext": STATE_LABELS,
                "title": "State",
            },
        )
    )
    fig.update_layout(
        title=f"{country} PCA curve-state timeline",
        xaxis_title="Date",
        yaxis_title="Component",
    )
    return fig


def _plot_transition_matrices(
    config: ProjectConfig,
    state_scores: pd.DataFrame,
    targets: pd.DataFrame,
    components: list[str],
) -> list[Path]:
    targets = targets.copy()
    targets["date"] = pd.to_datetime(targets["date"])
    outputs: list[Path] = []
    for country, country_scores in state_scores.groupby("country", sort=True):
        country_targets = targets.loc[targets["country"] == country]
        if country_targets.empty:
            continue
        thresholds = _country_thresholds(country_scores, components)
        for horizon_days, horizon_targets in country_targets.groupby("horizon_days", sort=True):
            merged = horizon_targets.merge(
                country_scores.loc[:, ["date", "country", *[f"{c}_state" for c in components]]],
                on=["date", "country"],
                how="inner",
            )
            for component in components:
                future_column = f"future_{component}"
                if future_column not in merged.columns:
                    continue
                current_state = merged[f"{component}_state"].astype("string")
                future_state = _apply_thresholds(merged[future_column], thresholds[component])
                matrix = _transition_matrix(current_state, future_state)
                output_path = (
                    config.figures_dir
                    / f"{str(country).lower()}_{component.lower()}_state_transitions_{int(horizon_days)}d.html"
                )
                _transition_heatmap(
                    matrix,
                    country=str(country),
                    component=component,
                    horizon_days=int(horizon_days),
                ).write_html(output_path)
                outputs.append(output_path)
    return outputs


def _country_thresholds(
    country_scores: pd.DataFrame,
    components: list[str],
) -> dict[str, tuple[float, float]]:
    return {
        component: tuple(country_scores[component].quantile([1 / 3, 2 / 3]).to_numpy(dtype=float))
        for component in components
    }


def _apply_thresholds(values: pd.Series, thresholds: tuple[float, float]) -> pd.Series:
    lower, upper = thresholds
    labels = np.select(
        [values <= lower, values <= upper],
        ["low", "medium"],
        default="high",
    )
    return pd.Series(labels, index=values.index, dtype="string")


def _transition_matrix(current_state: pd.Series, future_state: pd.Series) -> pd.DataFrame:
    counts = pd.crosstab(
        pd.Categorical(current_state, categories=STATE_LABELS, ordered=True),
        pd.Categorical(future_state, categories=STATE_LABELS, ordered=True),
        dropna=False,
    )
    counts.index = STATE_LABELS
    counts.columns = STATE_LABELS
    row_totals = counts.sum(axis=1)
    return counts.div(row_totals.replace(0, np.nan), axis=0).fillna(0.0)


def _transition_heatmap(
    matrix: pd.DataFrame,
    country: str,
    component: str,
    horizon_days: int,
) -> Any:
    fig = go.Figure(
        data=go.Heatmap(
            x=STATE_LABELS,
            y=STATE_LABELS,
            z=matrix.to_numpy(dtype=float),
            text=np.round(matrix.to_numpy(dtype=float), 3),
            hovertemplate=(
                "Current=%{y}<br>Future=%{x}<br>Probability=%{z:.3f}<extra></extra>"
            ),
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar={"title": "Probability"},
        )
    )
    fig.update_layout(
        title=f"{country} {component} state transitions, {horizon_days}d horizon",
        xaxis_title="Future state",
        yaxis_title="Current state",
    )
    return fig
