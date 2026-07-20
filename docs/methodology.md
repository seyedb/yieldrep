# Methodology

This note defines the current classical baseline protocol for `yieldrep`. The
purpose is to keep the project focused while the data infrastructure and
benchmark tasks are still being established.

## Objective

The long-run research question is whether learned latent representations of
sovereign yield curves capture information beyond classical term-structure
features.

The current phase does not train learned representations. It builds the public
data pipeline, classical curve representations, benchmark targets, and evaluation
framework needed before representation learning is introduced.

Current results should be interpreted as research diagnostics, not trading
signals or claims of predictive edge.

## Data Schema

All source curves are normalized to one long-format panel:

```text
date, country, maturity_years, yield, source
```

For country \(c\), date \(t\), and maturity \(m\), the observed zero-coupon
yield is:

```math
r_t^{(c,m)}
```

A country curve at date \(t\) is the maturity vector:

```math
\mathbf{r}_t^{(c)}
=
\left[
r_t^{(c,m_1)},
r_t^{(c,m_2)},
\ldots,
r_t^{(c,m_M)}
\right]
```

## Classical Representations

### PCA

PCA treats each daily curve as a vector and estimates orthogonal linear factors
that explain historical curve variation.

```math
\mathbf{z}_t^{(c)}
=
\left[
PC_{1,t}^{(c)},
PC_{2,t}^{(c)},
\ldots,
PC_{K,t}^{(c)}
\right]
```

In rates applications, the first few components often resemble level, slope, and
curvature. This interpretation is approximate: PCA signs are arbitrary, and the
factor shapes depend on the sample, maturities, and scaling.

### Nelson-Siegel

Nelson-Siegel fits a parametric curve with level, slope, and curvature loadings:

```math
r(m)
=
\beta_0
+ \beta_1
\frac{1 - e^{-m/\tau}}{m/\tau}
+ \beta_2
\left(
\frac{1 - e^{-m/\tau}}{m/\tau}
- e^{-m/\tau}
\right)
```

With fixed \(\tau\), the betas are estimated by ordinary least squares for each
country and date. The fitted residuals measure maturity-specific richness or
cheapness relative to the parametric curve.

### Engineered Curve Features

The current engineered baseline includes transparent curve-shape features:

```text
level
slope_10y_2y
curvature_2s5s10s
front_slope_2y_1y
long_slope_30y_10y
```

It also includes simple zero-curve carry and roll-down proxies:

```math
\mathrm{carry}_{t,u}^{(m)}
=
u r_t^{(m)}
```

```math
\mathrm{rolldown}_{t,u}^{(m)}
=
r_t^{(m-u)}
-
r_t^{(m)}
```

These are not full bond return calculations. They are interpretable
term-structure proxies built from public zero-coupon data.

### Lagged And Residual Features

Lagged yield-change features are included as a simple autoregressive hurdle:

```math
\Delta_\ell r_t^{(c,m)}
=
r_t^{(c,m)}
-
r_{t-\ell}^{(c,m)}
```

Residual dynamic features use Nelson-Siegel residuals and rolling residual
z-scores:

```math
z_{t,W}^{(c,m)}
=
\frac{
e_t^{(c,m)} - \mu_{t,W}^{(c,m)}
}{
\sigma_{t,W}^{(c,m)}
}
```

These features are included because residual mean reversion and local
richness/cheapness are standard relative-value ideas in rates research.

## Targets

### Outright Yield Change

For horizon \(h\):

```math
y_{t,h}^{(c,m)}
=
r_{t+h}^{(c,m)}
-
r_t^{(c,m)}
```

This is the simplest forecasting target, but it is also difficult and often
dominated by macro shocks.

### Standardized Yield Change

Yield changes can also be scaled by trailing realized volatility:

```math
z_{t,h}^{(c,m)}
=
\frac{
r_{t+h}^{(c,m)}
-
r_t^{(c,m)}
}{
\sigma_t^{(c,m)}
}
```

This asks whether features explain risk-adjusted moves rather than raw
basis-point changes.

### Residual Change

For fitted yield \(\hat{r}_t^{(c,m)}\), define the curve residual:

```math
e_t^{(c,m)}
=
r_t^{(c,m)}
-
\hat{r}_t^{(c,m)}
```

The residual-change target is:

```math
y_{t,h,\mathrm{resid}}^{(c,m)}
=
e_{t+h}^{(c,m)}
-
e_t^{(c,m)}
```

This is the current relative-value target. It is more aligned with curve
representation research than outright yield-change prediction.

### Volatility Change And Regime

Realized volatility is estimated from rolling yield changes:

```math
\sigma_t^{(c,m)}
=
\mathrm{std}
\left(
\Delta r_{t-W+1}^{(c,m)},
\ldots,
\Delta r_t^{(c,m)}
\right)
```

The volatility-change target is:

```math
y_{t,h,\mathrm{vol}}^{(c,m)}
=
\sigma_{t+h}^{(c,m)}
-
\sigma_t^{(c,m)}
```

The pipeline also stores empirical future volatility-regime labels:

```math
g_{t,h}^{(c,m)}
\in
\{\mathrm{low}, \mathrm{medium}, \mathrm{high}\}
```

These are early curve-state labels, not a final regime model.

## Evaluation Protocol

### Reconstruction

Reconstruction evaluates whether a representation compresses the observed curve:

```math
e_t^{(m)}
=
r_t^{(m)}
-
\hat{r}_t^{(m)}
```

PCA reconstructs curves from the first \(K\) components. Nelson-Siegel
reconstructs curves from the fitted parametric form. Metrics are reported
overall and by maturity.

This is currently the cleanest representation-quality benchmark in the project.

### Supervised Forecasting

Supervised benchmarks join features available at date \(t\) to a future target:

```math
y_{t,h}^{(c,m)}
=
f(\mathbf{x}_t^{(c,m)})
+ \varepsilon_{t,h}^{(c,m)}
```

The current models are intentionally classical:

```text
train_mean
ridge
elastic_net
```

The training mean is the naive benchmark. Ridge and Elastic Net are regularized
linear hurdles for testing whether each feature family adds incremental
information.

### Splits

The default split is chronological. Within each country and horizon:

```text
train_dates = first 80% of dates
test_dates  = last 20% of dates
```

All maturities for a date stay on the same side of the split.

Non-overlapping target windows are used by default for multi-step horizons:

```text
test_dates_non_overlapping = test_dates[::h]
```

This reduces artificial predictability from overlapping forward-change labels.

Walk-forward evaluation is available as a robustness check, not the main
headline result.

### Evaluation Level

Not every representation is evaluated on every task in the same way. The project
separates curve-level and maturity-level evaluation:

| Level | Natural representations | Natural tasks |
| --- | --- | --- |
| Curve-level | PCA scores, Nelson-Siegel factors, curve-shape features | reconstruction, curve-state summaries, aggregate curve-move forecasting |
| Maturity-level | residual features, lagged maturity moves, carry/roll-down proxies | residual relative value, cross-sectional maturity ranking |

PCA and Nelson-Siegel are currently date/country-level curve representations.
They describe the state of the whole curve at a point in time. In this
implementation, they do not by themselves create maturity-varying predictions
within the same date, so they are not expected to produce valid cross-sectional
rank IC for maturity-level residual RV tasks.

This is an evaluation-design distinction, not a claim that PCA or
Nelson-Siegel are weak representations. A learned representation that is meant
to compete on residual RV ranking should explicitly produce maturity-aware
features or node-level embeddings.

### Metric Protocol

Metrics are interpreted by task. A single pooled error number is not treated as
the universal objective.

The current metric hierarchy is:

| Task | Primary metric | Secondary metric | Context metrics |
| --- | --- | --- | --- |
| Curve reconstruction | RMSE / MAE | PCA explained variance | maturity-level reconstruction error |
| Residual relative value | residual RV spread score | cross-sectional rank IC | RMSE, MAE, directional accuracy |
| Outright yield-change forecasting | RMSE / MAE | directional accuracy | rank IC where valid |
| Volatility-regime classification | balanced accuracy / macro F1 | accuracy | class support |

This hierarchy is intentionally narrow. New metrics should only be added if they
answer a distinct research question that the current set does not cover.

Reconstruction uses RMSE and MAE as primary metrics because the task is curve
compression: the question is whether the representation reproduces observed
yield levels.

Outright yield-change and volatility-change forecasting use RMSE and MAE as
point-forecast metrics. Directional accuracy is reported as a secondary sign
metric, but it is not sufficient on its own because it ignores forecast
magnitude.

Residual relative-value evaluation uses the residual RV spread score as the
primary ranking metric. For each date, country, and horizon, maturities are
sorted by predicted residual change. The score is the realized average residual
change of the top-ranked maturities minus the realized average residual change
of the bottom-ranked maturities:

```math
S_{t,h}^{(c)}
=
\frac{1}{|T_t|}
\sum_{m \in T_t}
y_{t,h}^{(c,m)}
-
\frac{1}{|B_t|}
\sum_{m \in B_t}
y_{t,h}^{(c,m)}
```

where \(T_t\) and \(B_t\) are the top and bottom predicted maturity groups. This
is a cross-sectional ranking score, not a tradable PnL or duration-neutral
backtest.

Cross-sectional rank IC is the secondary residual RV metric. RMSE and MAE remain
useful context, but they are not headline metrics for this task because an RV
workflow often cares more about ordering maturities than minimizing pooled
basis-point error.

Volatility-regime classification is evaluated separately with classification
metrics from the baseline classifier output.

Regression metrics:

```math
RMSE
=
\sqrt{
\frac{1}{N}
\sum_i
\left(
y_i - \hat{y}_i
\right)^2
}
```

```math
MAE
=
\frac{1}{N}
\sum_i
\left|
y_i - \hat{y}_i
\right|
```

```math
\text{Directional Accuracy}
=
\frac{1}{N}
\sum_i
\mathbf{1}
\left[
\mathrm{sign}(y_i)
=
\mathrm{sign}(\hat{y}_i)
\right]
```

Cross-sectional rank IC ranks predicted and realized targets across maturities
within each date, country, and horizon:

```math
IC_t
=
\mathrm{corr}
\left(
\mathrm{rank}(\hat{y}_{t}^{(m)}),
\mathrm{rank}(y_{t}^{(m)})
\right)
```

This metric is included because relative-value research often cares about
cross-sectional ordering more than pooled point forecast error.

For residual relative value, the main report is:

```text
reports/tables/residual_relative_value_spread.csv
```

The rank-IC report is kept as a secondary ranking view:

```text
reports/tables/residual_relative_value_rank_ic.csv
```

The bucket-level RMSE table is kept as supporting context:

```text
reports/tables/residual_relative_value.csv
```

The rank-IC coverage audit explains which feature sets can produce valid
cross-sectional rankings:

```text
reports/tables/residual_relative_value_rank_ic_coverage.csv
```

## Current Scope

Included now:

- public US and Canada zero-coupon curve data
- normalized long-format curve schema
- PCA and Nelson-Siegel curve representations
- engineered slope, curvature, carry, roll-down, lagged, and residual features
- reconstruction evaluation
- classical supervised forecasting baselines
- residual RV ranking metrics for maturity-level feature sets
- chronological, non-overlapping, and walk-forward evaluation checks
- Plotly figures and CSV report tables

Not included yet:

- autoencoders, Transformers, or graph neural networks
- claim of tradable alpha or state-of-the-art forecasting performance
- bond-level total return targets
- transaction costs, RFQ execution constraints, or backtesting
- macro, policy-rate, inflation, unemployment, VIX, or MOVE features

## Current Limitations

The current forecasting benchmarks are preliminary. Outright yield-change
prediction is a noisy task, and strong curve reconstruction does not by itself
imply forecastability.

Near-term extensions should focus on relative-value evaluation, especially
Nelson-Siegel residual dynamics and cross-sectional ranking across maturities,
before introducing learned representations.
