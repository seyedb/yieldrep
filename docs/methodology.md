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

The ECB source is treated as an aggregate euro-area curve (`EA`), not as a
single sovereign issuer. The configured ECB series uses all-issuer euro-area
zero-coupon spot rates.

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

### Policy-Rate Features

Policy-rate features add central-bank context to curve-level classification
tasks. The current sources are the daily effective fed funds rate for the US,
the Bank of Canada target overnight rate, and the ECB deposit facility rate for
the euro area.

For each country, policy observations are aligned to curve dates using the most
recent available policy-rate observation:

```math
p_t^{(c)}
```

The feature set includes policy-rate level, recent changes, and a simple
curve-policy spread:

```math
\Delta_\ell p_t^{(c)}
=
p_t^{(c)}
-
p_{t-\ell}^{(c)}
```

```math
\mathrm{policy\ spread}_t^{(c)}
=
r_t^{(c,2Y)}
-
p_t^{(c)}
```

These are macro-policy context features, not learned representations.

### Market-Volatility Regimes

Market-volatility indicators add broad risk-regime context without becoming a
large macro feature set. The current indicators are VIX and MOVE. VIX proxies
equity volatility; MOVE is the more directly rates-volatility proxy.

Each indicator is normalized to:

```text
date, indicator, value, source
```

Low, medium, and high regimes are assigned with expanding historical terciles.
At date \(t\), the thresholds use only indicator observations available before
that date:

```math
q_{1/3,t}^{(j)}, q_{2/3,t}^{(j)}
```

These regimes are used as conditioning variables for relative-value
diagnostics, not as standalone trading signals.

### Macro Regimes

Macro indicators add slower-moving inflation and labor-market context. The
current macro schema is:

```text
date, country, indicator, value, source
```

Current indicators are:

```text
inflation: US, Canada, euro area
unemployment: US, Canada
```

US and euro-area inflation are computed as 12-month percentage changes from CPI
or HICP index levels. Canada inflation is read directly from a FRED/OECD
year-over-year inflation series. Unemployment rates are read as monthly
seasonally adjusted rates where current public sources are available.

Low, medium, and high macro regimes are assigned with expanding historical
terciles by country and indicator, again using only prior observations for the
thresholds.

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

### State-Maturity Linear Baselines

For residual relative-value evaluation, the project also includes maturity-aware
classical panel baselines. These combine curve-level state variables with a
continuous maturity basis:

```text
maturity
maturity_squared
log_maturity
```

and state-by-maturity interactions. The intent is to test whether a curve-level
state representation, such as PCA scores or Nelson-Siegel factors, can imply
different residual-change forecasts at different points on the curve.

These baselines are not interpreted as new representations. They are stronger
linear classical comparators for maturity-level residual RV ranking.

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

The direct residual mean-reversion diagnostic asks whether the current residual
predicts movement back toward the fitted curve:

```math
C_{t,h}^{(c,m)}
=
-
\mathrm{sign}
\left(
e_t^{(c,m)}
\right)
\left(
e_{t+h}^{(c,m)}
-
e_t^{(c,m)}
\right)
```

A positive value means the residual moved in the opposite direction from its
current sign. This is a simple rich/cheap diagnostic, not a tradable spread PnL.

### Volatility Change

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

### Curve Volatility Regime

Curve-level volatility regimes are defined from the forward root-mean-square
move of the whole curve:

```math
v_{t,h}^{(c)}
=
\sqrt{
\frac{1}{M}
\sum_m
\left(
r_{t+h}^{(c,m)}
-
r_t^{(c,m)}
\right)^2
}
```

The current realized curve volatility feature is the root-mean-square of
trailing maturity-level realized volatilities:

```math
\sigma_{t,\mathrm{curve}}^{(c)}
=
\sqrt{
\frac{1}{M}
\sum_m
\left(
\sigma_t^{(c,m)}
\right)^2
}
```

Low, medium, and high regimes are assigned inside each train/test split using
training-sample terciles of \(v_{t,h}^{(c)}\). The test set is then labeled with
the training thresholds:

```math
g_{t,h}^{(c)}
\in
\{\mathrm{low}, \mathrm{medium}, \mathrm{high}\}
```

This avoids using full-sample regime thresholds and makes PCA, Nelson-Siegel,
engineered curve features, and recent realized curve volatility comparable on a
natural curve-level classification task.

### Curve-State Classification

Curve-state classification uses PCA scores as empirical state coordinates. The
first three components are treated as level, slope, and curvature-like state
axes:

```math
s_{k,t+h}^{(c)}
=
PC_{k,t+h}^{(c)}
```

For each component \(k \in \{1,2,3\}\), the future score is bucketed into low,
medium, or high regimes using training-sample terciles inside each split. This
creates a simple transition question:

```text
given today's curve representation, predict the future PCA state bucket
```

This is a representation benchmark, not a claim that PCA states are the final
economic regime definition.

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
| Curve-level | PCA scores, Nelson-Siegel factors, curve-shape features | reconstruction, volatility regimes, curve-state classification |
| State-maturity panel | PCA/NS/curve factors with maturity basis interactions | residual RV ranking as a stronger classical comparator |
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
| Residual relative value | residual RV spread score | cross-sectional rank IC | mean-reversion hit rate, RMSE, MAE |
| Outright yield-change forecasting | RMSE / MAE | directional accuracy | rank IC where valid |
| Volatility-regime classification | balanced accuracy / macro F1 | accuracy | class support |
| Curve-state classification | balanced accuracy / macro F1 | accuracy | class support |

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

The residual mean-reversion table reports convergence scores and hit rates for
raw residuals and rolling residual z-scores. This is used as an interpretable
sanity check before treating richer representation models as RV signals.

Volatility-regime classification is evaluated separately with classification
metrics from the baseline classifier output.

Balanced accuracy averages recall across regimes:

```math
\mathrm{Balanced\ Accuracy}
=
\frac{1}{K}
\sum_{k=1}^{K}
\frac{TP_k}{TP_k + FN_k}
```

Macro F1 computes F1 for each regime and averages the class-level scores:

```math
\mathrm{Macro\ F1}
=
\frac{1}{K}
\sum_{k=1}^{K}
\frac{
2\,\mathrm{Precision}_k\,\mathrm{Recall}_k
}{
\mathrm{Precision}_k + \mathrm{Recall}_k
}
```

The compact volatility benchmark report compares curve representations against
recent realized curve volatility as the direct persistence hurdle:

```text
reports/tables/volatility_regime_benchmark.csv
```

Curve-state classification is summarized here:

```text
reports/tables/curve_state.csv
```

Curve-state timelines and transition matrices are written to:

```text
reports/figures/*_curve_state_*.html
reports/figures/*_state_transitions_*d.html
```

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

For residual relative value, the main overview report is:

```text
reports/tables/residual_relative_value_overview.csv
```

It combines the best RV ranking benchmark with the direct Nelson-Siegel residual
mean-reversion diagnostic. The detailed spread-score report is:

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

Cross-market diagnostics compare representation behavior across US, Canada, and
the euro-area aggregate curve:

```text
reports/tables/cross_market_summary.csv
reports/figures/cross_market_pca_loadings.html
```

Market-volatility conditioning for residual relative value is reported in:

```text
reports/tables/residual_rv_by_market_regime.csv
reports/tables/market_regime_rv_summary.csv
```

## Current Scope

Included now:

- public US, Canada, and euro-area zero-coupon curve data
- normalized long-format curve schema
- PCA and Nelson-Siegel curve representations
- engineered slope, curvature, carry, roll-down, lagged, and residual features
- policy-rate level, change, and curve-policy spread features
- VIX and MOVE market-volatility regimes for residual RV conditioning
- inflation and unemployment macro regimes where current public sources are configured
- reconstruction evaluation
- classical supervised forecasting baselines
- residual RV ranking metrics for maturity-level feature sets
- curve-level volatility-regime classification
- PCA-defined curve-state classification
- cross-market PCA, Nelson-Siegel, and state diagnostics
- chronological, non-overlapping, and walk-forward evaluation checks
- Plotly figures and CSV report tables

Not included yet:

- autoencoders, Transformers, or graph neural networks
- claim of tradable alpha or state-of-the-art forecasting performance
- bond-level total return targets
- transaction costs, RFQ execution constraints, or backtesting
- euro-area unemployment regime source
- broader macro feature sets

## Current Limitations

The current forecasting benchmarks are preliminary. Outright yield-change
prediction is a noisy task, and strong curve reconstruction does not by itself
imply forecastability.

Near-term extensions should keep the task definitions fixed while improving the
classical benchmark set and adding macro or policy-rate context.
