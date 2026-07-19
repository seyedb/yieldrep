from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from yieldrep.config import ProjectConfig
from yieldrep.factors.curve import curve_panel


GROUP_COLUMNS = ["country", "representation", "n_components"]
MATURITY_GROUP_COLUMNS = [*GROUP_COLUMNS, "maturity_years", "maturity_bucket"]


def evaluate_reconstruction(config: ProjectConfig) -> list[Path]:
    """Evaluate how well classical representations reconstruct observed curves."""
    curves = pd.read_parquet(config.curves_path)

    rows = [_pca_reconstruction_errors(curves, config), _nelson_siegel_reconstruction_errors(config)]
    errors = pd.concat([row for row in rows if not row.empty], ignore_index=True)

    config.tables_dir.mkdir(parents=True, exist_ok=True)
    summary = _summarize_reconstruction(errors, GROUP_COLUMNS)
    by_maturity = _summarize_reconstruction(errors, MATURITY_GROUP_COLUMNS)
    summary.to_csv(config.reconstruction_summary_table_path, index=False)
    by_maturity.to_csv(config.reconstruction_by_maturity_table_path, index=False)
    return [
        config.reconstruction_summary_table_path,
        config.reconstruction_by_maturity_table_path,
    ]


def _pca_reconstruction_errors(curves: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for country in sorted(curves["country"].dropna().unique()):
        panel = curve_panel(curves, str(country)).ffill().dropna()
        if panel.shape[1] < config.pca.min_maturities:
            continue

        max_components = min(config.pca.n_components, panel.shape[0], panel.shape[1])
        rows.extend(_fit_pca_reconstructions(str(country), panel, max_components))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fit_pca_reconstructions(
    country: str,
    panel: pd.DataFrame,
    max_components: int,
) -> list[pd.DataFrame]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(panel)
    model = PCA(n_components=max_components)
    scores = model.fit_transform(x_scaled)

    rows: list[pd.DataFrame] = []
    for n_components in range(1, max_components + 1):
        reconstructed_scaled = scores[:, :n_components] @ model.components_[:n_components]
        reconstructed = scaler.inverse_transform(reconstructed_scaled)
        rows.append(
            _panel_errors(
                country=country,
                representation="pca",
                n_components=n_components,
                actual=panel,
                fitted=pd.DataFrame(reconstructed, index=panel.index, columns=panel.columns),
            )
        )
    return rows


def _nelson_siegel_reconstruction_errors(config: ProjectConfig) -> pd.DataFrame:
    if not config.nelson_siegel_dir.exists():
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    curves = pd.read_parquet(config.curves_path)
    for fitted_path in sorted(config.nelson_siegel_dir.glob("*_fitted.parquet")):
        fitted = pd.read_parquet(fitted_path)
        merged = curves.merge(
            fitted,
            on=["date", "country", "maturity_years"],
            how="inner",
        )
        if merged.empty:
            continue

        frame = merged.loc[:, ["date", "country", "maturity_years", "yield", "fitted_yield"]].copy()
        frame["representation"] = "nelson_siegel"
        frame["n_components"] = 3
        frame["error"] = frame["yield"] - frame["fitted_yield"]
        rows.append(_format_errors(frame))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _panel_errors(
    country: str,
    representation: str,
    n_components: int,
    actual: pd.DataFrame,
    fitted: pd.DataFrame,
) -> pd.DataFrame:
    actual_long = _stack_panel(actual, value_name="yield")
    fitted_long = _stack_panel(fitted, value_name="fitted_yield")
    frame = actual_long.merge(fitted_long, on=["date", "maturity_years"], how="inner")
    frame["country"] = country
    frame["representation"] = representation
    frame["n_components"] = n_components
    frame["error"] = frame["yield"] - frame["fitted_yield"]
    return _format_errors(frame)


def _stack_panel(panel: pd.DataFrame, value_name: str) -> pd.DataFrame:
    long = panel.stack().rename(value_name).reset_index()
    return long.rename(columns={long.columns[0]: "date", long.columns[1]: "maturity_years"})


def _format_errors(errors: pd.DataFrame) -> pd.DataFrame:
    frame = errors.copy()
    frame["maturity_years"] = frame["maturity_years"].astype(float)
    frame["maturity_bucket"] = _maturity_bucket(frame["maturity_years"])
    frame["squared_error"] = np.square(frame["error"])
    frame["absolute_error"] = np.abs(frame["error"])
    return frame.loc[
        :,
        [
            "date",
            "country",
            "representation",
            "n_components",
            "maturity_years",
            "maturity_bucket",
            "error",
            "squared_error",
            "absolute_error",
        ],
    ]


def _summarize_reconstruction(errors: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if errors.empty:
        return pd.DataFrame(
            columns=[
                *group_columns,
                "observations",
                "dates",
                "rmse",
                "mae",
                "mean_error",
            ]
        )

    summary = (
        errors.groupby(group_columns, sort=True, observed=True)
        .agg(
            observations=("error", "size"),
            dates=("date", "nunique"),
            mse=("squared_error", "mean"),
            mae=("absolute_error", "mean"),
            mean_error=("error", "mean"),
        )
        .reset_index()
    )
    summary["rmse"] = np.sqrt(summary["mse"])
    return summary.drop(columns=["mse"]).sort_values([*group_columns, "rmse"]).reset_index(drop=True)


def _maturity_bucket(maturity_years: pd.Series) -> pd.Series:
    return pd.cut(
        maturity_years,
        bins=[0.0, 2.0, 10.0, float("inf")],
        labels=["front_end", "belly", "long_end"],
        right=True,
    ).astype("string")
