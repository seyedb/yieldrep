# Baseline Summary

## Scope

Current evaluation covers:

- US, Canada, and euro-area zero-coupon yield curves
- PCA and Nelson-Siegel curve representations
- engineered curve-shape, lagged, carry, roll-down, residual, policy, volatility, and macro features
- denoising autoencoder, maturity Transformer, and maturity-graph reconstruction baselines
- maturity-graph node and edge datasets
- residual relative-value diagnostics
- market and macro regime conditioning
- volatility-regime classification
- Plotly figures and CSV scorecards

## Current Evidence

| Scenario | Current result |
| --- | --- |
| Clean curve reconstruction | PCA-5 is the strongest benchmark across US, Canada, and euro-area curves. |
| Masked maturity reconstruction | The maturity-graph autoencoder leads US and euro-area masked reconstruction; the MLP autoencoder remains stronger for Canada. |
| Outright yield-change forecasting | Results are noisy and remain secondary to RV and regime tasks. |
| Residual relative value | Evidence is positive but modest at short horizons, with stronger 20-day results in selected markets. |
| Macro/market RV regimes | High inflation and high MOVE regimes show stronger residual RV mean-reversion in several country/horizon pairs. |
| Volatility regimes | Realized curve volatility and policy features remain main hurdles; AE leads CA 20-day classification and graph AE leads US 1-day classification. |

## Core Outputs

| Output | Path |
| --- | --- |
| Baseline audit | `reports/tables/baseline_audit.csv` |
| Scenario-method map | `reports/tables/scenario_method_comparison.csv` |
| Representation task scorecard | `reports/tables/representation_task_scorecard.csv` |
| Learned model comparison | `reports/tables/learned_model_comparison.csv` |
| Learned reconstruction leaderboard | `reports/tables/learned_reconstruction_leaderboard.csv` |
| Representation comparison | `reports/tables/representation_comparison.csv` |
| Maturity graph nodes | `data/processed/graph/maturity_graph_nodes.parquet` |
| Maturity graph edges | `data/processed/graph/maturity_graph_edges.parquet` |
| Macro-conditioned representation summary | `reports/tables/macro_conditioned_representation_summary.csv` |
| Reconstruction comparison | `reports/tables/reconstruction_oos_comparison.csv` |
| Residual RV scorecard | `reports/tables/residual_relative_value_scorecard.csv` |
| RV regime scorecard | `reports/tables/residual_rv_regime_scorecard.csv` |
| RV regime heatmap | `reports/figures/residual_rv_regime_heatmap.html` |
| Learned-state regime summary | `reports/tables/learned_state_regime_summary.csv` |
| Learned-state regime heatmap | `reports/figures/learned_state_regime_heatmap.html` |

## Interpretation

Classical curve representations remain strong reconstruction benchmarks. PCA is
the main clean-reconstruction hurdle, while Nelson-Siegel residuals provide the
relative-value object.

The autoencoder and graph autoencoder are the strongest learned reconstruction
baselines currently in the project. The graph model leads masked reconstruction
for US and euro-area curves, while the MLP autoencoder remains stronger for
Canada. The current maturity Transformer remains weaker under this protocol.

AE and maturity-Transformer embeddings are now evaluated as standalone
downstream feature sets against the same yield-change, residual-change,
volatility-change, and curve-volatility-regime targets as the classical
benchmarks.

The most relevant research direction is now residual relative value under macro
and market regimes. This is the clearest bridge from classical rates structure
to learned curve-state analysis.

Learned-state diagnostics compare AE, Transformer, and graph-AE latent states
across macro and market regimes. The strongest current separations appear in
Canada unemployment regimes and selected inflation/MOVE regimes.

The representation comparison table consolidates reconstruction quality,
masked-maturity reconstruction, learned-state regime separation, and direct
engineered benchmark feature sets. These columns are reported side by side, but
they are not treated as the same statistical task.

The representation task scorecard includes materiality flags to separate
learned edges from competitive ties and non-material differences.

The macro-conditioned summary reports the same evidence by inflation,
unemployment, VIX, and MOVE regimes where those regime labels are available.

Heavier learned-model training is kept outside the default pipeline through
`configs/learned_heavy.yaml` and the `train-learned-models` CLI command.

## Next Phase

The next phase uses the representation scorecard to choose focused follow-up work:

- compare representation behavior across inflation, unemployment, VIX, and MOVE regimes
- emphasize residual RV and volatility-state questions over average outright forecasts
- keep representation outputs separated from target-definition tools
