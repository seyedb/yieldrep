from pathlib import Path

import typer

from yieldrep.config import load_config
from yieldrep.data.ingest import ingest_sources
from yieldrep.data.normalize import build_curves_parquet
from yieldrep.evaluation.datasets import build_modeling_datasets
from yieldrep.features.nelson_siegel import build_nelson_siegel
from yieldrep.features.pca import build_pca
from yieldrep.features.targets import build_targets
from yieldrep.models.baselines import evaluate_baselines
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


@app.command("plot-pca")
def plot_pca_command(config: Path = Path("configs/default.yaml")) -> None:
    """Generate Plotly HTML figures from PCA outputs."""
    project_config = load_config(config)
    for output_path in plot_pca(project_config):
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
