from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px

from yieldrep.config import ProjectConfig
from yieldrep.factors.curve import curve_panel


def plot_curves(config: ProjectConfig) -> list[Path]:
    """Write Plotly HTML figures for curve history and selected maturities."""
    curves = pd.read_parquet(config.curves_path)
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for country in sorted(curves["country"].dropna().unique()):
        country_code = str(country)
        output_paths.extend(_plot_country_curves(config, curves, country_code))

    return output_paths


def _plot_country_curves(
    config: ProjectConfig,
    curves: pd.DataFrame,
    country: str,
) -> list[Path]:
    country_key = country.lower()
    country_curves = curves.loc[curves["country"] == country].copy()
    selected = country_curves[
        country_curves["maturity_years"].isin(config.plots.selected_maturities)
    ]
    panel = curve_panel(curves, country)

    selected_html = config.figures_dir / f"{country_key}_selected_maturities.html"
    heatmap_html = config.figures_dir / f"{country_key}_curve_heatmap.html"

    selected_fig = px.line(
        selected,
        x="date",
        y="yield",
        color="maturity_years",
        title=f"{country} selected maturities",
        labels={"date": "Date", "yield": "Yield", "maturity_years": "Maturity"},
    )
    selected_fig.write_html(selected_html)

    heatmap_fig = px.imshow(
        panel.T,
        aspect="auto",
        origin="lower",
        title=f"{country} yield curve through time",
        labels={"x": "Date", "y": "Maturity", "color": "Yield"},
    )
    heatmap_fig.write_html(heatmap_html)

    return [selected_html, heatmap_html]
