# Classical Baseline Summary

This note summarizes the current classical baseline stage. It is a checkpoint,
not a claim of tradable performance.

## Scope

Current evaluation covers:

- US, Canada, and euro-area public zero-coupon yield curves
- PCA and Nelson-Siegel curve representations
- curve-shape, carry/roll-down, lagged, residual-dynamic, and state-maturity
  linear baselines
- curve reconstruction metrics
- residual relative-value ranking and mean-reversion diagnostics
- curve-level volatility-regime classification
- PCA-defined curve-state classification
- cross-market representation diagnostics
- policy-rate features for curve-level classification tasks

## Metric Hierarchy

The current primary metrics are:

| Task | Primary metric | Secondary/context metrics |
| --- | --- | --- |
| Curve reconstruction | RMSE / MAE | PCA explained variance |
| Residual relative value | residual RV spread score | rank IC, convergence hit rate |
| Outright yield-change forecasting | RMSE / MAE | directional accuracy |
| Volatility regime classification | balanced accuracy / macro F1 | accuracy |

## Headline Results

For residual relative value, `residual_feature / ridge` is currently the best
classical baseline across the evaluated markets and horizons by:

- residual RV spread score
- cross-sectional rank IC

The direct Nelson-Siegel residual mean-reversion diagnostic is now tracked
separately. It checks whether positive residuals tend to fall and negative
residuals tend to rise over the forward horizon.

The concise RV overview table combines the benchmark winner, spread score, rank
IC, and direct residual mean-reversion evidence by country and horizon.

The maturity-aware PCA, Nelson-Siegel, and curve baselines produce valid
cross-sectional RV metrics, which makes the comparison more structurally fair
than using curve-level factors alone. They do not currently outperform the
residual dynamic feature baseline.

For volatility regimes, the project now evaluates curve-level classifiers using
future curve-move magnitude labels assigned from training-sample quantiles.
Recent realized curve volatility is the current hurdle, and it is the strongest
baseline for most evaluated country/horizon pairs.

For curve-state classification, the project evaluates whether current PCA,
Nelson-Siegel, and engineered curve features predict future PCA state buckets
for the first three components.

Cross-market diagnostics compare PCA variance, PCA score co-movement,
Nelson-Siegel factor co-movement, and PCA state overlap across US, Canada, and
the euro-area aggregate curve.

Policy-rate features currently improve euro-area volatility-regime
classification for the 1-day and 20-day horizons. For US and Canada, recent
realized curve volatility remains the stronger hurdle.

## Interpretation

PCA and Nelson-Siegel remain useful curve-level representations. They are most
clearly validated through reconstruction and curve-state summaries.

Residual relative-value ranking is a maturity-level task. In the current
classical setup, features that directly describe maturity-specific residual
dynamics are more effective for that task than curve-level state factors or
state-maturity linear interactions.

## Limitations

The residual RV spread score is a ranking metric, not a tradable PnL. It does
not include duration-neutral construction, transaction costs, liquidity,
financing, or execution constraints.

The euro-area ECB source is an aggregate all-issuers curve, not a single
sovereign issuer. Inflation, labor-market data, and market volatility indices
are not yet included.

## Next Step

The next research step should use the residual mean-reversion and RV ranking
tables to decide which relative-value protocol is stable enough to become the
main benchmark before adding learned representations.
