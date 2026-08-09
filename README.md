# yieldrep
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**yieldrep** is a Python research project that investigates whether learned
latent representations of sovereign yield curves capture information beyond
classical term-structure features.

Using public fixed-income and macro/market data, the project compares PCA,
Nelson-Siegel, engineered curve features, denoising autoencoders, maturity
Transformers, and maturity-graph autoencoders.

The goal is not to claim novel trading signals or state-of-the-art predictive
performance. The emphasis is a clean, reproducible framework for evaluating
classical and learned yield-curve representations across reconstruction,
residual relative value, volatility regimes, and macro/market conditioning.

## Scope

Current data coverage:

- US Treasury zero-coupon curves from Federal Reserve Gürkaynak-Sack-Wright data
- Bank of Canada zero-coupon government curves
- ECB euro-area yield curves
- public policy-rate, inflation, unemployment, VIX, and MOVE series

Current model families:

- PCA and Nelson-Siegel
- curve-shape, carry, roll-down, residual, policy, and volatility features
- denoising autoencoder
- maturity Transformer
- maturity-graph autoencoder

Package layout:

```text
yieldrep/
  data/            public-source loaders, normalization, schema checks
  factors/         PCA, Nelson-Siegel, curve, carry, residual, policy features
  models/          forecasting baselines and learned reconstruction models
  graph/           maturity-graph dataset construction
  evaluation/      targets, reconstruction, residual RV, regimes, scorecards
  visualization/   Plotly figures
```

The canonical project summary is:

```text
reports/tables/research_checkpoint_scorecard.csv
```

The residual relative-value protocol is defined in:

```text
reports/tables/residual_rv_protocol.csv
```

## Current Findings

| Task | Current result |
| --- | --- |
| Clean reconstruction | PCA remains the strongest benchmark; the graph autoencoder is the best learned clean reconstructor but does not beat PCA. |
| Masked maturity reconstruction | The maturity-graph autoencoder is the strongest learned model for the current masked reconstruction benchmark. |
| Residual relative value | Nelson-Siegel residual convergence has positive but modest evidence; the current best specification is residual lagged-ridge. |
| Volatility regimes | Realized curve-volatility features remain the strongest classifier; autoencoder states are close but not clearly better. |
| Macro/market-conditioned residual RV | Nelson-Siegel residual features lead overall, while graph-autoencoder states are useful in selected regime cells. |

## Usage

Run commands from the project root. In PyCharm, use the project conda
interpreter and set `PYTHONPATH=src` in the run configuration.

```bash
PYTHONPATH=src python -m yieldrep.cli ingest --config configs/default.yaml
PYTHONPATH=src python -m yieldrep.cli normalize --config configs/default.yaml
PYTHONPATH=src python -m yieldrep.cli plot-curves --config configs/default.yaml
PYTHONPATH=src python -m yieldrep.cli plot-pca --config configs/default.yaml
PYTHONPATH=src python -m yieldrep.cli run-baselines --config configs/default.yaml
PYTHONPATH=src python -m yieldrep.cli train-learned-models --config configs/learned_heavy.yaml
PYTHONPATH=src python -m yieldrep.cli scorecard --config configs/default.yaml
```

Generated data is written under `data/`; generated tables and figures are
written under `reports/`.

## Milestone Status

This version completes the first research milestone: public sovereign curve
data, a common curve schema, classical term-structure baselines, learned
reconstruction baselines, graph-based masked reconstruction, and task-level
evaluation diagnostics.

The empirical conclusion is deliberately modest: PCA remains the clean
reconstruction benchmark, Nelson-Siegel residuals remain the main relative-value
anchor, and learned models are most useful in masked maturity reconstruction and
selected regime-conditioned tasks.

**Development Note**

AI tools are used to assist with code generation, refactoring, and documentation.
All research questions, architectural decisions, experimental design,
implementation review, and interpretation of results are determined by the
author.
