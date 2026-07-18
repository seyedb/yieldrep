from pathlib import Path

import typer

from yieldrep.config import load_config
from yieldrep.data.ingest import ingest_sources
from yieldrep.data.normalize import build_curves_parquet
from yieldrep.evaluation.datasets import build_modeling_datasets
from yieldrep.evaluation.reports import summarize_baselines
from yieldrep.factors.curve import build_curve_features
from yieldrep.factors.nelson_siegel import build_nelson_siegel
from yieldrep.factors.pca import build_pca
from yieldrep.evaluation.targets import build_residual_targets, build_targets, build_vol_targets
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


if __name__ == "__main__":
    app()
