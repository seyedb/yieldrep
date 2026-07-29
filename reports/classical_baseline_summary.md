# Baseline Summary

## Scope

Current evaluation covers:

- US, Canada, and euro-area zero-coupon yield curves
- PCA and Nelson-Siegel curve representations
- engineered curve-shape, lagged, carry, roll-down, residual, policy, volatility, and macro features
- denoising autoencoder and maturity Transformer reconstruction baselines
- residual relative-value diagnostics
- market and macro regime conditioning
- volatility-regime classification
- Plotly figures and CSV scorecards

## Current Evidence

| Scenario | Current result |
| --- | --- |
| Clean curve reconstruction | PCA-5 is the strongest benchmark across US, Canada, and euro-area curves. |
| Masked maturity reconstruction | The masked autoencoder is stronger than the current maturity Transformer. |
| Outright yield-change forecasting | Results are noisy and remain secondary to RV and regime tasks. |
| Residual relative value | Evidence is positive but modest at short horizons, with stronger 20-day results in selected markets. |
| Macro/market RV regimes | High inflation and high MOVE regimes show stronger residual RV mean-reversion in several country/horizon pairs. |
| Volatility regimes | Realized curve volatility and policy features are the main classification hurdles. |

## Core Outputs

| Output | Path |
| --- | --- |
| Baseline audit | `reports/tables/baseline_audit.csv` |
| Scenario-method map | `reports/tables/scenario_method_comparison.csv` |
| Learned model comparison | `reports/tables/learned_model_comparison.csv` |
| Representation comparison | `reports/tables/representation_comparison.csv` |
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

The autoencoder is the strongest learned reconstruction baseline currently in
the project. The maturity Transformer has a more appropriate token-based
architecture; under the current protocol its masked-reconstruction error remains
above the autoencoder benchmark.

The most relevant research direction is now residual relative value under macro
and market regimes. This is the clearest bridge from classical rates structure
to learned curve-state analysis.

Initial learned-state diagnostics compare AE and Transformer latent states across
macro and market regimes. The strongest current separations appear in Canada
unemployment regimes and selected inflation/MOVE regimes.

The representation comparison table consolidates reconstruction quality,
masked-maturity reconstruction, learned-state regime separation, and direct
engineered benchmark feature sets. These columns are reported side by side, but
they are not treated as the same statistical task.

The macro-conditioned summary reports the same evidence by inflation,
unemployment, VIX, and MOVE regimes where those regime labels are available.

## Next Phase

The next phase uses the representation checkpoint for macro-aware evaluation:

- compare representation behavior across inflation, unemployment, VIX, and MOVE regimes
- emphasize residual RV and volatility-state questions over average outright forecasts
- keep representation outputs separated from target-definition tools
