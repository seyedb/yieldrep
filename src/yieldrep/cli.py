from pathlib import Path

import typer

from yieldrep.config import load_config
from yieldrep.data.normalize import build_curves_parquet

app = typer.Typer(help="Yield curve research pipelines.")


@app.callback()
def main() -> None:
    """Yield curve research pipelines."""


@app.command()
def normalize(config: Path = Path("configs/default.yaml")) -> None:
    """Build normalized yield curve parquet from local raw files."""
    project_config = load_config(config)
    output_path = build_curves_parquet(project_config)
    typer.echo(output_path)


if __name__ == "__main__":
    app()
