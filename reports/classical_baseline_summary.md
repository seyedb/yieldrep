# Research Phase Summary

## Scope

Current evaluation covers US, Canada, and euro-area zero-coupon yield curves.
The implemented representations are PCA, Nelson-Siegel, engineered
term-structure features, denoising autoencoders, maturity Transformers, and a
maturity-graph autoencoder.

The current task set is:

| Task | Role |
| --- | --- |
| Clean curve reconstruction | linear and nonlinear reconstruction benchmark |
| Masked maturity reconstruction | self-supervised representation task |
| Nelson-Siegel residual relative value | curve-dislocation target |
| Macro and market regime conditioning | economic state dependence |
| Curve-volatility regime classification | state-classification benchmark |

## Current Evidence

| Question | Current evidence |
| --- | --- |
| Clean reconstruction | PCA-5 remains the strongest benchmark. |
| Masked maturity reconstruction | The maturity-graph autoencoder leads US and euro-area curves; the first edge ablation favors adjacent-only edges over adjacent+correlation edges in US and euro-area curves. |
| Residual relative value | Raw Nelson-Siegel residuals remain the strongest overall RV feature, while graph-AE states show learned edges in selected EA/US 5-day and 20-day regime cells. |
| Volatility regimes | Realized curve-volatility and policy features remain strong reference baselines; learned models are competitive only in selected cells. |
| Learned state diagnostics | Transformer and graph-AE states separate some macro/market regimes, with the clearest current separation in Canada unemployment regimes. |

## Main Interpretation

The strongest result so far is not outright yield-change prediction. It is the
masked maturity reconstruction task, where graph-aware structure provides a
natural and testable advantage over plain curve-vector models.

PCA remains the clean-reconstruction benchmark. Nelson-Siegel remains the
relative-value reference because it defines interpretable curve residuals.
Learned models are therefore most useful at this stage as alternative curve-state
representations, not as replacements for classical term-structure objects.

## Core Outputs

| Output | Path |
| --- | --- |
| Scenario-method map | `reports/tables/scenario_method_comparison.csv` |
| Representation task scorecard | `reports/tables/representation_task_scorecard.csv` |
| Learned model findings | `reports/tables/learned_model_findings.csv` |
| Learned reconstruction leaderboard | `reports/tables/learned_reconstruction_leaderboard.csv` |
| GNN edge ablation | `reports/tables/gnn_edge_ablation.csv` |
| Reconstruction comparison | `reports/tables/reconstruction_oos_comparison.csv` |
| Residual RV scorecard | `reports/tables/residual_relative_value_scorecard.csv` |
| RV regime scorecard | `reports/tables/residual_rv_regime_scorecard.csv` |
| Representation RV regime scorecard | `reports/tables/residual_rv_representation_regime_scorecard.csv` |
| Learned-state regime summary | `reports/tables/learned_state_regime_summary.csv` |
| Maturity graph nodes | `data/processed/graph/maturity_graph_nodes.parquet` |
| Maturity graph edges | `data/processed/graph/maturity_graph_edges.parquet` |

## Next Phase

The next phase should consolidate around two questions:

1. Can graph-aware masked reconstruction produce stable curve states across
   countries and regimes?
2. Do those states add useful information for Nelson-Siegel residual RV or
   volatility-regime tasks without using model outputs as target definitions?
