# Classical Baseline Summary

This note summarizes the current classical baseline stage. It is a checkpoint,
not a claim of tradable performance.

## Scope

Current evaluation covers:

- US, Canada, and euro-area public zero-coupon yield curves
- PCA and Nelson-Siegel curve representations
- a first PyTorch autoencoder reconstruction baseline
- curve-shape, carry/roll-down, lagged, policy-rate, and realized-volatility
  baselines
- curve reconstruction metrics
- residual relative-value ranking and mean-reversion diagnostics
- VIX and MOVE market-volatility regime conditioning for residual RV
- inflation and unemployment macro-regime data
- curve-level volatility-regime classification
- cross-market representation diagnostics
- policy-rate features for curve-level classification tasks

## Metric Hierarchy

The current primary metrics are:

| Task | Primary metric | Secondary/context metrics |
| --- | --- | --- |
| Clean curve reconstruction | out-of-sample RMSE / MAE | PCA explained variance |
| Masked maturity reconstruction | masked-maturity RMSE / MAE | autoencoder train/validation diagnostics |
| Residual relative value | residual RV spread score | rank IC, convergence hit rate |
| Outright yield-change forecasting | RMSE / MAE | directional accuracy |
| Volatility regime classification | balanced accuracy / macro F1 | accuracy |

## Headline Results

The direct Nelson-Siegel residual mean-reversion diagnostic is now tracked
separately. It checks whether positive residuals tend to fall and negative
residuals tend to rise over the forward horizon.

The concise RV overview table combines the benchmark winner, spread score, rank
IC, and direct residual mean-reversion evidence by country and horizon.

Market-volatility conditioning now splits residual mean-reversion diagnostics by
VIX and MOVE regimes. MOVE is the more directly rates-relevant conditioning
variable.

The compact market-regime RV summary shows the clearest high-volatility
differentiation in MOVE regimes, especially for US residual RV at 5d and 20d and
Canada at 20d.

Macro-regime conditioning shows stronger residual mean reversion in high
inflation and high unemployment regimes, most clearly at the 20-day horizon.

Maturity-aware curve-shape baselines produce valid cross-sectional RV metrics
without feeding model outputs into another supervised model.

For volatility regimes, the project now evaluates curve-level classifiers using
future curve-move magnitude labels assigned from training-sample quantiles.
Recent realized curve volatility is the current hurdle, and it is the strongest
baseline for most evaluated country/horizon pairs.

Cross-market diagnostics compare PCA variance, PCA score co-movement,
and Nelson-Siegel factor co-movement across US, Canada, and the euro-area
aggregate curve.

Policy-rate features currently improve euro-area volatility-regime
classification for the 1-day and 20-day horizons. For US and Canada, recent
realized curve volatility remains the stronger hurdle.

The first learned baseline is a masked autoencoder: maturities are randomly
hidden during training, the model receives a mask indicator, and the objective
combines masked-maturity reconstruction with clean-curve reconstruction. Its
embeddings are not fed into downstream benchmark models. Clean out-of-sample
reconstruction is now evaluated separately from masked-maturity reconstruction.
PCA-5 remains the strongest clean reconstruction benchmark in US, Canada, and
the euro-area aggregate curve.

## Interpretation

PCA and Nelson-Siegel remain useful curve-level representations. They are most
clearly validated through reconstruction and cross-market factor diagnostics.

The masked autoencoder is a more appropriate learned-representation baseline
than the initial plain autoencoder, but it is still not a win on clean held-out
curve reconstruction. It establishes the PyTorch pipeline and gives future
learned models a clear PCA reconstruction hurdle.

Residual relative-value ranking is a maturity-level task. In the current
classical setup, direct curve-shape, lagged, and carry/roll-down features are
the allowed supervised feature families.

## Limitations

The residual RV spread score is a ranking metric, not a tradable PnL. It does
not include duration-neutral construction, transaction costs, liquidity,
financing, or execution constraints.

The euro-area ECB source is an aggregate all-issuers curve, not a single
sovereign issuer. Euro-area unemployment is not yet included because the simple
FRED/OECD euro-area unemployment endpoint is stale relative to the current
sample.

## Next Step

The next research step should improve the learned representation protocol
itself, for example with better regularization, architecture constraints, or a
more stable masked-reconstruction setup, before moving to larger architectures.
