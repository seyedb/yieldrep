from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from yieldrep.config import ProjectConfig
from yieldrep.factors.carry import CARRY_ROLL_FEATURE_COLUMNS

GRAPH_NODE_COLUMNS = [
    "date",
    "country",
    "node_id",
    "maturity_years",
    "yield",
    "yield_change_1d",
    "realized_vol",
    *CARRY_ROLL_FEATURE_COLUMNS,
]

GRAPH_EDGE_COLUMNS = [
    "country",
    "source_node_id",
    "target_node_id",
    "source_maturity_years",
    "target_maturity_years",
    "edge_type",
    "maturity_distance_years",
    "distance_weight",
    "correlation",
    "edge_weight",
]


def build_maturity_graph_dataset(config: ProjectConfig) -> list[Path]:
    """Build graph-ready node and edge parquet datasets from observed curves."""
    curves = pd.read_parquet(config.curves_path)
    carry_roll = (
        pd.read_parquet(config.carry_roll_features_path)
        if config.carry_roll_features_path.exists()
        else None
    )

    nodes = make_maturity_graph_nodes(
        curves,
        carry_roll=carry_roll,
        realized_vol_window=config.graph.realized_vol_window,
    )
    edges = make_maturity_graph_edges(
        curves,
        min_observations=config.graph.correlation_min_observations,
        correlation_top_k=config.graph.correlation_top_k,
    )

    config.graph_dir.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(config.graph_nodes_path, index=False)
    edges.to_parquet(config.graph_edges_path, index=False)
    return [config.graph_nodes_path, config.graph_edges_path]


def make_maturity_graph_nodes(
    curves: pd.DataFrame,
    carry_roll: pd.DataFrame | None,
    realized_vol_window: int,
) -> pd.DataFrame:
    """Create one graph node per country/date/maturity observation."""
    if realized_vol_window <= 1:
        raise ValueError("realized_vol_window must be greater than 1")

    nodes = curves.loc[:, ["date", "country", "maturity_years", "yield"]].copy()
    nodes["date"] = pd.to_datetime(nodes["date"]).dt.normalize()
    nodes = nodes.sort_values(["country", "maturity_years", "date"]).reset_index(drop=True)
    grouped = nodes.groupby(["country", "maturity_years"], sort=False)["yield"]
    nodes["yield_change_1d"] = grouped.diff()
    nodes["realized_vol"] = grouped.transform(
        lambda series: series.diff().rolling(realized_vol_window).std()
    )

    if carry_roll is not None and not carry_roll.empty:
        nodes = nodes.merge(
            carry_roll,
            on=["date", "country", "maturity_years"],
            how="left",
        )

    for column in CARRY_ROLL_FEATURE_COLUMNS:
        if column not in nodes.columns:
            nodes[column] = np.nan

    nodes["node_id"] = _node_ids(nodes["country"], nodes["maturity_years"])
    return (
        nodes.loc[:, GRAPH_NODE_COLUMNS]
        .sort_values(["country", "date", "maturity_years"])
        .reset_index(drop=True)
    )


def make_maturity_graph_edges(
    curves: pd.DataFrame,
    min_observations: int,
    correlation_top_k: int,
) -> pd.DataFrame:
    """Create static maturity edges per country.

    Adjacent edges encode the curve's natural maturity ordering. Correlation
    edges connect each maturity to its strongest historical yield-change peers
    within the same country.
    """
    if min_observations <= 1:
        raise ValueError("min_observations must be greater than 1")
    if correlation_top_k < 0:
        raise ValueError("correlation_top_k must be non-negative")

    edge_frames: list[pd.DataFrame] = []
    for country in sorted(curves["country"].dropna().unique()):
        country_curves = curves.loc[curves["country"] == country].copy()
        maturities = sorted(country_curves["maturity_years"].dropna().unique())
        if len(maturities) < 2:
            continue

        edge_frames.append(_adjacent_edges(str(country), maturities))
        if correlation_top_k:
            edge_frames.append(
                _correlation_edges(
                    str(country),
                    country_curves,
                    min_observations=min_observations,
                    top_k=correlation_top_k,
                )
            )

    non_empty = [frame for frame in edge_frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=GRAPH_EDGE_COLUMNS)

    return (
        pd.concat(non_empty, ignore_index=True)
        .loc[:, GRAPH_EDGE_COLUMNS]
        .sort_values(["country", "edge_type", "source_maturity_years", "target_maturity_years"])
        .reset_index(drop=True)
    )


def _adjacent_edges(country: str, maturities: list[float]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for source, target in zip(maturities[:-1], maturities[1:], strict=False):
        rows.extend(
            [
                _edge_row(country, source, target, "adjacent", correlation=np.nan),
                _edge_row(country, target, source, "adjacent", correlation=np.nan),
            ]
        )
    return pd.DataFrame(rows, columns=GRAPH_EDGE_COLUMNS)


def _correlation_edges(
    country: str,
    curves: pd.DataFrame,
    min_observations: int,
    top_k: int,
) -> pd.DataFrame:
    panel = curves.pivot(index="date", columns="maturity_years", values="yield").sort_index()
    changes = panel.diff()
    correlations = changes.corr(min_periods=min_observations)

    rows: list[dict[str, float | str]] = []
    for source in correlations.columns:
        peers = correlations.loc[source].drop(index=source).dropna()
        if peers.empty:
            continue
        strongest = peers.reindex(peers.abs().sort_values(ascending=False).head(top_k).index)
        for target, correlation in strongest.items():
            rows.append(
                _edge_row(country, float(source), float(target), "correlation", correlation)
            )
    return pd.DataFrame(rows, columns=GRAPH_EDGE_COLUMNS)


def _edge_row(
    country: str,
    source: float,
    target: float,
    edge_type: str,
    correlation: float,
) -> dict[str, float | str]:
    distance = abs(target - source)
    distance_weight = 1.0 / (1.0 + distance)
    edge_weight = abs(correlation) if np.isfinite(correlation) else distance_weight
    return {
        "country": country,
        "source_node_id": _node_id(country, source),
        "target_node_id": _node_id(country, target),
        "source_maturity_years": source,
        "target_maturity_years": target,
        "edge_type": edge_type,
        "maturity_distance_years": distance,
        "distance_weight": distance_weight,
        "correlation": correlation,
        "edge_weight": edge_weight,
    }


def _node_ids(countries: pd.Series, maturities: pd.Series) -> pd.Series:
    return countries.astype(str) + ":" + maturities.map(_format_maturity)


def _node_id(country: str, maturity: float) -> str:
    return f"{country}:{_format_maturity(maturity)}"


def _format_maturity(maturity: float) -> str:
    return f"{maturity:g}Y"
