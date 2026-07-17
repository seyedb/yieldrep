# yieldrep
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**yieldrep** is a Python research project that investigates whether learned
latent representations of sovereign yield curves capture information beyond
classical term-structure features.

Using publicly available fixed-income data, the project compares learned curve
representations with established baselines, including PCA, Nelson-Siegel factors,
slope, curvature, carry, and roll-down features, across a range of forecasting,
relative-value, volatility, and curve-state classification tasks.

The goal is not to claim novel trading signals or state-of-the-art predictive
performance. Instead, the emphasis is on building a rigorous, reproducible
research framework for systematically evaluating classical and learned
representations of the yield curve.

The project is designed to be modular, enabling new representation-learning
models, benchmark features, and downstream evaluation tasks to be incorporated
with minimal changes.

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

**Development Note**

AI tools are used to assist with code generation, refactoring, and documentation.
All research questions, architectural decisions, experimental design,
implementation review, and interpretation of results are determined by the
author.
