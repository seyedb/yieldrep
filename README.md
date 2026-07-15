# yieldrep
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

`yieldrep` is a Python research project for studying whether latent representations
of sovereign yield curves contain information beyond classical term-structure
features.

The project uses public fixed-income data to compare learned curve representations
against baselines such as PCA, Nelson-Siegel factors, slope and curvature measures,
and carry/roll-down style features across forecasting, relative-value, volatility,
and curve-state tasks.

## Usage

Run commands from the project root with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m yieldrep.cli ingest --config configs/default.yaml
PYTHONPATH=src python -m yieldrep.cli normalize --config configs/default.yaml
PYTHONPATH=src python -m yieldrep.cli build-pca --config configs/default.yaml
PYTHONPATH=src python -m yieldrep.cli plot-curves --config configs/default.yaml
PYTHONPATH=src python -m yieldrep.cli plot-pca --config configs/default.yaml
```

Generated data is written under `data/`; generated figures are written under
`reports/figures/`.

## Development Note

This project is developed as a learning and research effort with AI assistance for
code generation, refactoring, and documentation. Design decisions, review, testing,
and project direction are handled by the author.
