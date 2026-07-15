"""Principal-component baseline for yield curve representations.

PCA treats each daily curve as a vector of yields across maturities, such as
[1Y, 2Y, 5Y, 10Y, 30Y], and finds the orthogonal linear directions that explain
the most historical variation in those curve vectors. It is a classical
low-dimensional representation of the curve, not a forecasting model by itself.

In rates applications, the first components often resemble level, slope, and
curvature factors, although signs are arbitrary and the exact shapes depend on
the sample, maturities, and scaling. Scores are the date-indexed factor time
series, loadings are the maturity-wise factor shapes, and explained variance
measures how much historical curve variation each component captures.

This baseline is useful because later learned representations should be compared
against a strong linear benchmark before adding nonlinear autoencoders,
Transformers, or graph models.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from yieldrep.config import ProjectConfig
from yieldrep.features.curve import curve_panel


def build_pca(config: ProjectConfig) -> list[Path]:
    """Fit PCA by country and write scores, loadings, and explained variance."""
    curves = pd.read_parquet(config.curves_path)
    config.pca_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for country in sorted(curves["country"].dropna().unique()):
        panel = curve_panel(curves, str(country)).ffill().dropna()
        if panel.shape[1] < config.pca.min_maturities:
            continue

        n_components = min(config.pca.n_components, panel.shape[0], panel.shape[1])
        output_paths.extend(_fit_country_pca(config, str(country), panel, n_components))

    return output_paths


def _fit_country_pca(
    config: ProjectConfig,
    country: str,
    panel: pd.DataFrame,
    n_components: int,
) -> list[Path]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(panel)

    model = PCA(n_components=n_components)
    scores = model.fit_transform(x_scaled)
    components = [f"PC{i}" for i in range(1, n_components + 1)]
    country_key = country.lower()

    scores_path = config.pca_dir / f"{country_key}_scores.parquet"
    loadings_path = config.pca_dir / f"{country_key}_loadings.parquet"
    variance_path = config.pca_dir / f"{country_key}_variance.parquet"

    pd.DataFrame(scores, index=panel.index, columns=components).reset_index().to_parquet(
        scores_path,
        index=False,
    )
    pd.DataFrame(
        model.components_.T,
        index=pd.Index(panel.columns, name="maturity_years"),
        columns=components,
    ).reset_index().to_parquet(loadings_path, index=False)
    pd.DataFrame(
        {
            "component": components,
            "explained_variance_ratio": model.explained_variance_ratio_,
        }
    ).to_parquet(variance_path, index=False)

    return [scores_path, loadings_path, variance_path]
