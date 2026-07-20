# Classical Baseline Summary

This note summarizes the current classical baseline stage. It is a checkpoint,
not a claim of tradable performance.

## Scope

Current evaluation covers:

- US and Canada public zero-coupon sovereign yield curves
- PCA and Nelson-Siegel curve representations
- curve-shape, carry/roll-down, lagged, residual-dynamic, and state-maturity
  linear baselines
- curve reconstruction metrics
- residual relative-value ranking metrics

## Metric Hierarchy

The current primary metrics are:

| Task | Primary metric | Secondary/context metrics |
| --- | --- | --- |
| Curve reconstruction | RMSE / MAE | PCA explained variance |
| Residual relative value | residual RV spread score | rank IC, RMSE, MAE |
| Outright yield-change forecasting | RMSE / MAE | directional accuracy |
| Volatility regime classification | balanced accuracy / macro F1 | accuracy |

## Headline Results

For residual relative value, `residual_feature / ridge` is currently the best
classical baseline across both countries and all tested horizons by:

- residual RV spread score
- cross-sectional rank IC

The maturity-aware PCA, Nelson-Siegel, and curve baselines produce valid
cross-sectional RV metrics, which makes the comparison more structurally fair
than using curve-level factors alone. They do not currently outperform the
residual dynamic feature baseline.

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

The current data scope is limited to US and Canada curves. Macro variables,
policy-rate variables, inflation, labor-market data, and market volatility
indices are not yet included.

## Next Step

The next research step should keep the metric hierarchy fixed and focus on a
curve-level downstream task where PCA and Nelson-Siegel are natural baselines,
such as curve volatility or curve-state regime classification.
