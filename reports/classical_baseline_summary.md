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

## Current Findings

| Finding | Evidence |
| --- | --- |
| PCA remains the clean-reconstruction benchmark. | PCA-5 is the strongest clean reconstruction model; the adjacent-edge graph AE is the strongest learned clean reconstructor but ranks behind PCA. |
| Masked maturity reconstruction is the strongest self-supervised result. | The adjacent-edge graph AE leads US and euro-area masked reconstruction; the MLP autoencoder leads Canada. |
| Simple maturity adjacency is the preferred graph structure. | The edge ablation favors adjacent-only edges over adjacent+correlation edges in US and euro-area curves; correlation edges only slightly improve Canada. |
| Nelson-Siegel residuals remain the main RV anchor. | In regime-conditioned residual RV, residual/ridge is best in 41 valid cells, graph_autoencoder/ridge in 24, carry_roll/ridge in 20, and graph_autoencoder_macro_market/ridge in 9. |
| Graph-AE states have conditional RV value, not a universal edge. | Learned representations beat classical features in 33 of 94 valid regime cells; macro/market inputs add value mainly through graph-AE+macro in selected US cells. |
| Volatility-regime evidence is secondary. | Realized curve-volatility and policy features remain strong reference baselines; learned models are competitive only in selected cells. |

## Main Interpretation

The strongest result so far is masked maturity reconstruction, not outright
yield-change prediction. This is consistent with the project goal: learned
representations should first show that they capture curve structure before being
used in noisier forecasting tasks.

PCA remains the clean-reconstruction benchmark. Nelson-Siegel remains the
relative-value reference because it defines interpretable curve residuals. The
adjacent-edge graph AE is therefore best interpreted as a conditional curve-state
representation: useful in selected reconstruction and residual-RV regimes, but
not a replacement for classical term-structure objects.

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
| Representation RV regime findings | `reports/tables/residual_rv_representation_regime_findings.csv` |
| Macro-enhanced RV benchmark audit | `reports/tables/residual_rv_macro_benchmark_audit.csv` |
| Learned-state regime summary | `reports/tables/learned_state_regime_summary.csv` |
| Maturity graph nodes | `data/processed/graph/maturity_graph_nodes.parquet` |
| Maturity graph edges | `data/processed/graph/maturity_graph_edges.parquet` |

## Next Phase

The next phase should consolidate around two questions:

1. Can graph-aware masked reconstruction produce stable curve states across
   countries and regimes?
2. Do those states add useful information for Nelson-Siegel residual RV or
   volatility-regime tasks without using model outputs as target definitions?
