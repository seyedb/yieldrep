from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px

from yieldrep.config import ProjectConfig

COMPONENTS = ["PC1", "PC2", "PC3"]


def plot_cross_market_pca(config: ProjectConfig) -> list[Path]:
    """Write Plotly HTML figures for cross-market PCA diagnostics."""
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    loadings = _read_pca_loadings(config)
    if loadings.empty:
        return []

    output_path = config.figures_dir / "cross_market_pca_loadings.html"
    _plot_pca_loadings(loadings).write_html(output_path)
    return [output_path]


def _read_pca_loadings(config: ProjectConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(config.pca_dir.glob("*_loadings.parquet")):
        country = path.name.removesuffix("_loadings.parquet").upper()
        frame = pd.read_parquet(path)
        frame["country"] = country
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _plot_pca_loadings(loadings: pd.DataFrame) -> Any:
    components = [component for component in COMPONENTS if component in loadings.columns]
    long = loadings.melt(
        id_vars=["country", "maturity_years"],
        value_vars=components,
        var_name="component",
        value_name="loading",
    )
    return px.line(
        long,
        x="maturity_years",
        y="loading",
        color="country",
        facet_col="component",
        markers=True,
        title="Cross-market PCA loading comparison",
        labels={"maturity_years": "Maturity years", "loading": "Loading"},
    )
