from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

from yieldrep.config import ProjectConfig
from yieldrep.evaluation.reports import ae_classical_factor_correlation_summary


def plot_reconstruction(config: ProjectConfig) -> list[Path]:
    """Write Plotly HTML figures for reconstruction quality metrics."""
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(config.reconstruction_summary_table_path)
    oos_summary = pd.read_csv(config.reconstruction_oos_summary_table_path)
    oos_by_maturity = pd.read_csv(config.reconstruction_oos_by_maturity_table_path)
    oos_by_bucket = pd.read_csv(config.reconstruction_oos_by_maturity_bucket_table_path)

    component_path = config.figures_dir / "reconstruction_rmse_by_component.html"
    comparison_path = config.figures_dir / "reconstruction_oos_rmse_comparison.html"
    bucket_path = config.figures_dir / "reconstruction_oos_rmse_by_maturity_bucket.html"
    maturity_path = config.figures_dir / "reconstruction_oos_rmse_by_maturity.html"
    training_path = config.figures_dir / "reconstruction_autoencoder_training_history.html"
    latent_path = config.figures_dir / "autoencoder_latent_factor_correlations.html"
    latent_time_path = config.figures_dir / "autoencoder_latent_time_series.html"
    latent_scatter_path = config.figures_dir / "autoencoder_latent_state_space.html"

    _plot_pca_components(summary).write_html(component_path)
    _plot_representation_comparison(oos_summary).write_html(comparison_path)
    _plot_maturity_bucket_profile(oos_by_bucket).write_html(bucket_path)
    _plot_maturity_profile(oos_by_maturity).write_html(maturity_path)
    _plot_training_history(config).write_html(training_path)
    _plot_latent_factor_correlations(config).write_html(latent_path)
    _plot_latent_time_series(config).write_html(latent_time_path)
    _plot_latent_state_space(config).write_html(latent_scatter_path)

    return [
        component_path,
        comparison_path,
        bucket_path,
        maturity_path,
        training_path,
        latent_path,
        latent_time_path,
        latent_scatter_path,
    ]


def _plot_pca_components(summary: pd.DataFrame) -> Any:
    pca = summary.loc[summary["representation"] == "pca"].copy()
    return px.line(
        pca,
        x="n_components",
        y="rmse",
        color="country",
        markers=True,
        title="PCA reconstruction RMSE by component count",
        labels={"n_components": "PCA components", "rmse": "Reconstruction RMSE"},
    )


def _plot_representation_comparison(summary: pd.DataFrame) -> Any:
    comparison = _clean_comparison_rows(summary)
    return px.bar(
        comparison,
        x="country",
        y="rmse",
        color="representation_label",
        barmode="group",
        title="Out-of-sample curve reconstruction RMSE",
        labels={"rmse": "Reconstruction RMSE", "representation_label": "Representation"},
    )


def _plot_maturity_bucket_profile(by_bucket: pd.DataFrame) -> Any:
    comparison = _clean_comparison_rows(by_bucket)
    return px.bar(
        comparison,
        x="maturity_bucket",
        y="rmse",
        color="representation_label",
        facet_col="country",
        barmode="group",
        title="Out-of-sample reconstruction RMSE by maturity bucket",
        labels={
            "maturity_bucket": "Maturity bucket",
            "rmse": "Reconstruction RMSE",
            "representation_label": "Representation",
        },
    )


def _plot_maturity_profile(by_maturity: pd.DataFrame) -> Any:
    comparison = _clean_comparison_rows(by_maturity)
    return px.line(
        comparison,
        x="maturity_years",
        y="rmse",
        color="representation_label",
        facet_col="country",
        markers=True,
        title="Out-of-sample reconstruction RMSE by maturity",
        labels={"maturity_years": "Maturity years", "rmse": "Reconstruction RMSE"},
    )


def _clean_comparison_rows(data: pd.DataFrame) -> pd.DataFrame:
    clean = data.loc[data["reconstruction_task"] == "clean_reconstruction"].copy()
    pca = clean.loc[clean["representation"] == "pca"].copy()
    if not pca.empty:
        pca = pca.loc[pca["n_components"] == pca["n_components"].max()]

    nelson_siegel = clean.loc[clean["representation"] == "nelson_siegel"].copy()
    autoencoder = clean.loc[clean["representation"] == "autoencoder"].copy()
    comparison = pd.concat([pca, nelson_siegel, autoencoder], ignore_index=True)
    comparison["representation_label"] = comparison.apply(_representation_label, axis=1)
    return comparison


def _representation_label(row: pd.Series) -> str:
    if row["representation"] == "pca":
        return f"PCA {int(row['n_components'])} components"
    if row["representation"] == "autoencoder":
        return f"Autoencoder {int(row['n_components'])} latent dims"
    return "Nelson-Siegel"


def _plot_training_history(config: ProjectConfig) -> Figure:
    histories = [
        pd.read_parquet(path)
        for path in sorted(config.autoencoder_dir.glob("*_training_history.parquet"))
    ]
    if not histories:
        return Figure()

    history = pd.concat(histories, ignore_index=True)
    long = history.melt(
        id_vars=["country", "epoch"],
        value_vars=["train_loss", "validation_loss"],
        var_name="loss_type",
        value_name="loss",
    )
    return px.line(
        long,
        x="epoch",
        y="loss",
        color="loss_type",
        facet_col="country",
        title="Autoencoder training history",
        labels={"epoch": "Epoch", "loss": "Scaled reconstruction loss", "loss_type": "Loss"},
    )


def _plot_latent_factor_correlations(config: ProjectConfig) -> Figure:
    if config.ae_classical_factor_correlations_table_path.exists():
        correlations = pd.read_csv(config.ae_classical_factor_correlations_table_path)
    else:
        correlations = ae_classical_factor_correlation_summary(config)
        correlations.to_csv(config.ae_classical_factor_correlations_table_path, index=False)
    if correlations.empty:
        return Figure()

    correlations = correlations.copy()
    correlations["factor"] = (
        correlations["classical_family"].astype(str)
        + ": "
        + correlations["classical_feature"].astype(str)
    )
    return px.imshow(
        correlations.pivot_table(
            index=["country", "factor"],
            columns="ae_feature",
            values="abs_correlation",
            aggfunc="max",
        ),
        color_continuous_scale="Viridis",
        aspect="auto",
        title="Autoencoder latent correlations with curve and classical factors",
        labels={"x": "AE latent dimension", "y": "Factor", "color": "|Correlation|"},
    )


def _plot_latent_time_series(config: ProjectConfig) -> Figure:
    embeddings = _read_autoencoder_embeddings(config)
    if embeddings.empty:
        return Figure()

    latent_columns = _latent_columns(embeddings)[:3]
    if not latent_columns:
        return Figure()

    long = embeddings.melt(
        id_vars=["date", "country", "split"],
        value_vars=latent_columns,
        var_name="latent_dimension",
        value_name="latent_value",
    )
    return px.line(
        long,
        x="date",
        y="latent_value",
        color="latent_dimension",
        facet_col="country",
        title="Autoencoder latent dimensions through time",
        labels={
            "date": "Date",
            "latent_value": "Latent value",
            "latent_dimension": "Latent dimension",
        },
    )


def _plot_latent_state_space(config: ProjectConfig) -> Figure:
    embeddings = _read_autoencoder_embeddings(config)
    if embeddings.empty or not {"AE1", "AE2"}.issubset(embeddings.columns):
        return Figure()

    frame = _join_curve_features(config, embeddings)
    color_column = "level" if "level" in frame.columns else "split"
    return px.scatter(
        frame,
        x="AE1",
        y="AE2",
        color=color_column,
        facet_col="country",
        hover_data=["date", "split"],
        title="Autoencoder latent state space",
        labels={"AE1": "AE1", "AE2": "AE2", "level": "Curve level", "split": "Split"},
    )


def _read_autoencoder_embeddings(config: ProjectConfig) -> pd.DataFrame:
    frames = [
        pd.read_parquet(path)
        for path in sorted(config.autoencoder_dir.glob("*_embeddings.parquet"))
    ]
    if not frames:
        return pd.DataFrame()
    embeddings = pd.concat(frames, ignore_index=True)
    embeddings["date"] = pd.to_datetime(embeddings["date"])
    return embeddings.sort_values(["country", "date"]).reset_index(drop=True)


def _latent_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith("AE")],
        key=lambda column: int(column.removeprefix("AE")),
    )


def _join_curve_features(config: ProjectConfig, embeddings: pd.DataFrame) -> pd.DataFrame:
    if not config.curve_features_path.exists():
        return embeddings
    features = pd.read_parquet(config.curve_features_path)
    if features.empty:
        return embeddings
    features = features.copy()
    features["date"] = pd.to_datetime(features["date"])
    feature_columns = [
        "date",
        "country",
        "level",
        "slope_10y_2y",
        "curvature_2s5s10s",
    ]
    available_columns = [column for column in feature_columns if column in features.columns]
    return embeddings.merge(features.loc[:, available_columns], on=["date", "country"], how="left")
