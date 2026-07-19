from pathlib import Path

import typer

from yieldrep.config import ProjectConfig, load_config
from yieldrep.data.ingest import ingest_sources
from yieldrep.data.normalize import build_curves_parquet
from yieldrep.evaluation.datasets import build_modeling_datasets
from yieldrep.evaluation.diagnostics import diagnose_lagged_baseline
from yieldrep.evaluation.reports import summarize_baselines
from yieldrep.evaluation.targets import (
    build_residual_targets,
    build_standardized_targets,
    build_targets,
    build_vol_targets,
)
from yieldrep.factors.curve import build_curve_features
from yieldrep.factors.nelson_siegel import build_nelson_siegel
from yieldrep.factors.pca import build_pca
from yieldrep.factors.residual import build_residual_features
from yieldrep.models.baselines import evaluate_baselines
from yieldrep.visualization.plotly_baselines import plot_baseline_metrics
from yieldrep.visualization.plotly_curves import plot_curves
from yieldrep.visualization.plotly_nelson_siegel import plot_nelson_siegel
from yieldrep.visualization.plotly_pca import plot_pca

app = typer.Typer(help="Yield curve research pipelines.")


@app.callback()
def main() -> None:
    """Yield curve research pipelines."""


@app.command()
def ingest(config: Path = Path("configs/default.yaml"), overwrite: bool = False) -> None:
    """Download configured raw source files."""
    project_config = load_config(config)
    for raw_path in ingest_sources(project_config, overwrite=overwrite):
        typer.echo(raw_path)


@app.command()
def normalize(config: Path = Path("configs/default.yaml")) -> None:
    """Build normalized yield curve parquet from local raw files."""
    project_config = load_config(config)
    output_path = build_curves_parquet(project_config)
    typer.echo(output_path)


@app.command("build-pca")
def build_pca_command(config: Path = Path("configs/default.yaml")) -> None:
    """Build PCA baseline outputs from normalized curves."""
    project_config = load_config(config)
    for output_path in build_pca(project_config):
        typer.echo(output_path)


@app.command("build-nelson-siegel")
def build_nelson_siegel_command(config: Path = Path("configs/default.yaml")) -> None:
    """Build Nelson-Siegel baseline outputs from normalized curves."""
    project_config = load_config(config)
    for output_path in build_nelson_siegel(project_config):
        typer.echo(output_path)


@app.command("build-targets")
def build_targets_command(config: Path = Path("configs/default.yaml")) -> None:
    """Build forward yield-change prediction targets."""
    project_config = load_config(config)
    typer.echo(build_targets(project_config))


@app.command("build-standardized-targets")
def build_standardized_targets_command(config: Path = Path("configs/default.yaml")) -> None:
    """Build volatility-scaled yield-change prediction targets."""
    project_config = load_config(config)
    typer.echo(build_standardized_targets(project_config))


@app.command("build-residual-targets")
def build_residual_targets_command(config: Path = Path("configs/default.yaml")) -> None:
    """Build Nelson-Siegel residual-change prediction targets."""
    project_config = load_config(config)
    typer.echo(build_residual_targets(project_config))


@app.command("build-vol-targets")
def build_vol_targets_command(config: Path = Path("configs/default.yaml")) -> None:
    """Build realized-volatility-change prediction targets."""
    project_config = load_config(config)
    typer.echo(build_vol_targets(project_config))


@app.command("build-curve-features")
def build_curve_features_command(config: Path = Path("configs/default.yaml")) -> None:
    """Build engineered curve-shape baseline features."""
    project_config = load_config(config)
    typer.echo(build_curve_features(project_config))


@app.command("build-residual-features")
def build_residual_features_command(config: Path = Path("configs/default.yaml")) -> None:
    """Build dynamic Nelson-Siegel residual baseline features."""
    project_config = load_config(config)
    typer.echo(build_residual_features(project_config))


@app.command("run-baselines")
def run_baselines_command(config: Path = Path("configs/default.yaml")) -> None:
    """Run the full classical baseline research pipeline."""
    project_config = load_config(config)
    for output_path in run_baseline_pipeline(project_config):
        typer.echo(output_path)


@app.command("build-modeling-datasets")
def build_modeling_datasets_command(config: Path = Path("configs/default.yaml")) -> None:
    """Build supervised datasets by joining baseline features to targets."""
    project_config = load_config(config)
    for output_path in build_modeling_datasets(project_config):
        typer.echo(output_path)


@app.command("evaluate-baselines")
def evaluate_baselines_command(config: Path = Path("configs/default.yaml")) -> None:
    """Evaluate simple forecasting baselines."""
    project_config = load_config(config)
    typer.echo(evaluate_baselines(project_config))


@app.command("diagnose-lagged-baseline")
def diagnose_lagged_baseline_command(config: Path = Path("configs/default.yaml")) -> None:
    """Measure autocorrelation behind lagged baseline performance."""
    project_config = load_config(config)
    typer.echo(diagnose_lagged_baseline(project_config))


@app.command("summarize-baselines")
def summarize_baselines_command(config: Path = Path("configs/default.yaml")) -> None:
    """Write CSV summary tables from baseline evaluation metrics."""
    project_config = load_config(config)
    for output_path in summarize_baselines(project_config):
        typer.echo(output_path)


@app.command("plot-pca")
def plot_pca_command(config: Path = Path("configs/default.yaml")) -> None:
    """Generate Plotly HTML figures from PCA outputs."""
    project_config = load_config(config)
    for output_path in plot_pca(project_config):
        typer.echo(output_path)


@app.command("plot-baseline-metrics")
def plot_baseline_metrics_command(config: Path = Path("configs/default.yaml")) -> None:
    """Generate Plotly HTML figures from baseline evaluation metrics."""
    project_config = load_config(config)
    for output_path in plot_baseline_metrics(project_config):
        typer.echo(output_path)


@app.command("plot-nelson-siegel")
def plot_nelson_siegel_command(config: Path = Path("configs/default.yaml")) -> None:
    """Generate Plotly HTML figures from Nelson-Siegel outputs."""
    project_config = load_config(config)
    for output_path in plot_nelson_siegel(project_config):
        typer.echo(output_path)


@app.command("plot-curves")
def plot_curves_command(config: Path = Path("configs/default.yaml")) -> None:
    """Generate Plotly HTML figures from normalized curves."""
    project_config = load_config(config)
    for output_path in plot_curves(project_config):
        typer.echo(output_path)


def run_baseline_pipeline(project_config: ProjectConfig) -> list[Path]:
    """Run the standard baseline pipeline from normalized curves to reports."""
    output_paths: list[Path] = []
    output_paths.append(build_curves_parquet(project_config))
    output_paths.extend(build_pca(project_config))
    output_paths.extend(build_nelson_siegel(project_config))
    output_paths.append(build_curve_features(project_config))
    output_paths.append(build_residual_features(project_config))
    output_paths.append(build_targets(project_config))
    output_paths.append(build_standardized_targets(project_config))
    output_paths.append(build_residual_targets(project_config))
    output_paths.append(build_vol_targets(project_config))
    output_paths.extend(build_modeling_datasets(project_config))
    output_paths.append(evaluate_baselines(project_config))
    output_paths.extend(summarize_baselines(project_config))
    output_paths.extend(plot_baseline_metrics(project_config))
    return output_paths


if __name__ == "__main__":
    app()
