# Methodology

This note defines the current research setup for yield curve representations,
targets, and baseline evaluation.

## Contents

- Data and curve schema
- Classical representations
- Forecasting targets
- Reconstruction evaluation
- Supervised forecasting benchmarks
- Metrics
- Current limitations

## Data And Curve Schema

For country \(c\), date \(t\), and maturity \(m\), let the observed zero-coupon
yield be:

```math
r_t^{(c,m)}
```

A yield curve at date \(t\) is the vector of yields across maturities:

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

The normalized dataset stores this panel in long format:

```text
date, country, maturity_years, yield, source
```

## Classical Representations

### PCA

Principal component analysis treats each daily curve as a vector and finds the
orthogonal linear directions that explain the most historical variation.

For country \(c\), the PCA representation at date \(t\) is:

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

In rates applications, the first few principal components often resemble:

```text
PC1  level
PC2  slope
PC3  curvature
```

This interpretation is approximate. Signs are arbitrary, and exact shapes depend
on the sample window, maturities, and scaling.

### Nelson-Siegel

Nelson-Siegel imposes three parametric factor shapes: level, slope, and
curvature. For maturity \(m\) and fixed decay parameter \(\tau\):

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

The coefficients are interpreted as:

```text
beta_level      broad level of the curve
beta_slope      short-end versus long-end slope
beta_curvature  intermediate-maturity hump or belly
```

With \(\tau\) fixed, the betas are estimated by ordinary least squares for each
country and date. The fitted residuals and RMSE measure how well the parametric
curve matches the observed curve.

### Curve Shape Features

The engineered curve baseline uses simple cross-sectional shape descriptors:

```math
\mathrm{level}_t
=
\frac{1}{M}
\sum_j r_t^{(m_j)}
```

```math
\mathrm{slope}_{10y,2y,t}
=
r_t^{(10y)}
-
r_t^{(2y)}
```

```math
\mathrm{curvature}_{2s5s10s,t}
=
2r_t^{(5y)}
-
r_t^{(2y)}
-
r_t^{(10y)}
```

When an exact anchor maturity is unavailable, the nearest available maturity is
used. These features are deliberately simple and interpretable; they form a
classical hurdle for learned representations.

### Carry And Roll-Down Proxies

The project includes simple carry and roll-down proxy features built only from
the public zero-coupon curve. For maturity \(m\) and roll horizon \(u\), carry is
approximated by scaling the current zero yield:

```math
\mathrm{carry}_{t,u}^{(m)}
=
u r_t^{(m)}
```

Roll-down is the same-day yield difference between the rolled maturity and the
current maturity:

```math
\mathrm{rolldown}_{t,u}^{(m)}
=
r_t^{(m-u)}
-
r_t^{(m)}
```

The rolled yield is linearly interpolated on the available maturity grid. These
features are not full coupon-bond return estimates; they are transparent
term-structure proxies that help benchmark learned representations against
fixed-income-native classical inputs.

### Lagged Yield Changes

The lagged baseline uses recent maturity-specific curve moves:

```math
\Delta_\ell r_t^{(c,m)}
=
r_t^{(c,m)}
-
r_{t-\ell}^{(c,m)}
```

These features test whether factor representations add value beyond simple
autoregressive information in each yield point.

Lag diagnostics are used to interpret this baseline. For each target family, the
project reports correlation and sign agreement for:

```text
target lag versus current target
lagged yield-change feature versus current target
```

This separates genuine lag-feature signal from autocorrelation introduced by
overlapping forward-return windows.

### Residual Dynamic Features

The residual dynamic baseline adds local relative-value state. Residual z-scores
standardize the Nelson-Siegel residual within a rolling window:

```math
z_{t,W}^{(c,m)}
=
\frac{
e_t^{(c,m)}
-
\mu_{t,W}^{(c,m)}
}{
\sigma_{t,W}^{(c,m)}
}
```

Residual changes capture short-term momentum or reversal in richness/cheapness,
while residual volatility measures recent instability of the fitted-curve
residual.

### Current Feature Sets

```text
PCA:
    PC1, ..., PCK where K is configured by pca.n_components

Nelson-Siegel:
    beta_level, beta_slope, beta_curvature, rmse

Curve shape:
    level, slope_10y_2y, curvature_2s5s10s,
    front_slope_2y_1y, long_slope_30y_10y

Carry and roll-down:
    carry_1m, roll_down_1m, carry_3m, roll_down_3m,
    carry_12m, roll_down_12m

Lagged yield changes:
    lag_1_change, lag_5_change, lag_20_change

Residual dynamic features:
    residual, residual_z_60, residual_z_252,
    residual_change_1, residual_change_5, residual_vol_20
```

## Forecasting Targets

### Outright Yield Changes

For horizon \(h\), the main target is the forward yield change:

```math
y_{t,h}^{(c,m)}
=
r_{t+h}^{(c,m)}
- r_t^{(c,m)}
```

The default horizons are 1, 5, and 20 available observations. These are
observation steps, not exact calendar days.

### Volatility-Scaled Yield Changes

The project also supports a volatility-scaled yield-change target:

```math
z_{t,h}^{(c,m)}
=
\frac{
r_{t+h}^{(c,m)}
- r_t^{(c,m)}
}{
\sigma_t^{(c,m)}
}
```

where \(\sigma_t^{(c,m)}\) is trailing realized volatility of daily yield
changes at the same country and maturity. This target asks whether features help
predict risk-adjusted moves rather than raw basis-point moves.

### Residual / Relative-Value Changes

For fitted yield \(\hat{r}_t^{(c,m)}\), define the relative-value residual:

```math
e_t^{(c,m)}
=
r_t^{(c,m)}
-
\hat{r}_t^{(c,m)}
```

The residual target is:

```math
y_{t,h,\mathrm{resid}}^{(c,m)}
=
e_{t+h}^{(c,m)}
-
e_t^{(c,m)}
```

This target asks whether baseline representations help predict changes in
richness or cheapness after removing the fitted Nelson-Siegel curve.

### Volatility And Regime Targets

For rolling window \(W\), realized volatility is estimated from recent yield
changes:

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

The volatility target is:

```math
y_{t,h,\mathrm{vol}}^{(c,m)}
=
\sigma_{t+h}^{(c,m)}
-
\sigma_t^{(c,m)}
```

The pipeline also stores low, medium, and high future volatility-regime labels:

```math
g_{t,h}^{(c,m)}
\in
\{\mathrm{low}, \mathrm{medium}, \mathrm{high}\}
```

The classes are empirical volatility buckets for each country and maturity. This
is a simple first version of a curve-state task: the model is not asked to
forecast the exact volatility change, but to classify the future volatility
environment.

## Reconstruction Evaluation

Reconstruction metrics evaluate representation quality before any forecasting
task. For PCA, the observed curve is reconstructed from the first \(K\)
components. For Nelson-Siegel, the reconstructed curve is the fitted parametric
curve.

```math
e_t^{(m)}
=
r_t^{(m)}
-
\hat{r}_t^{(m)}
```

The project reports reconstruction RMSE and MAE overall and by maturity. This is
a standard compression/fit benchmark: it asks whether the representation encodes
the observed curve well, not whether it predicts future returns.

The worst-maturity diagnostics rank maturity points by reconstruction RMSE and
report signed bias to identify where each representation fits poorly.

Out-of-sample reconstruction fits the PCA basis on train dates and reconstructs
held-out test curves. This checks whether the linear representation generalizes
through time, rather than only fitting the full sample in-sample.

## Supervised Forecasting Benchmarks

### Supervised Dataset

The supervised dataset joins representation features to the target:

```math
y_{t,h}^{(c,m)}
=
f\left(\mathbf{x}_t^{(c)}\right)
+ \varepsilon_{t,h}^{(c,m)}
```

where \(\mathbf{x}_t^{(c)}\) is a baseline representation available at date
\(t\).

The canonical supervised tables store one row per country, date, maturity, and
horizon, with all currently available classical features joined to a specific
forecast target. The current canonical supervised targets are outright
yield-change, Nelson-Siegel residual-change, and realized-volatility change.
Each table also stores the chronological train/test split labels used by the
evaluation pipeline, so downstream models can reuse the same experimental
design.

### Models

The naive benchmark predicts the training-sample mean:

```math
\hat{y}
=
\frac{1}{N_{\text{train}}}
\sum_{i \in \text{train}} y_i
```

The main linear benchmark is Ridge regression:

```math
\min_{\alpha,\beta}
\sum_i
\left(
y_i
- \alpha
- \mathbf{x}_i^\top \beta
\right)^2
+ \lambda \lVert \beta \rVert_2^2
```

Ridge is ordinary least squares with an L2 penalty. The penalty helps stabilize
coefficients when factors are correlated or noisy.

The supervised benchmark also includes Elastic Net. Ridge tests whether dense
linear combinations of a feature set are useful; Elastic Net adds sparsity
pressure, so it can reveal whether only a small subset of features carries most
of the signal.

The benchmark reports improvement versus the training-set mean forecast, both
overall and by maturity bucket. Coefficient tables are based on standardized
features, so coefficient magnitudes are useful for inspecting which inputs the
linear models rely on within each country and horizon.

This is a standard linear hurdle: learned representations should eventually be
evaluated against this same supervised table and split design, not against a
different target construction.

### Splits

The default train/test split is date ordered. Within each country and forecast
horizon, unique dates are sorted chronologically:

```text
train_dates = first 80% of dates
test_dates  = last 20% of dates
```

All maturities from the same date remain on the same side of the split. This
avoids mixing observations from the same date across train and test samples.

Main forecasting reports use non-overlapping target windows by default. The test
set keeps every \(h\)-th date for an \(h\)-day forecast horizon:

```text
test_dates_non_overlapping = test_dates[::h]
```

This reduces the chance that apparent predictability comes from overlapping
forward-change labels rather than from genuinely forecastable curve information.

The overlap-sensitivity report runs both protocols and compares ranks and RMSE
side by side. The overlapping version is treated as diagnostic context rather
than the main benchmark.

Walk-forward evaluation is also supported. Each split trains on an expanding
history and tests on the next chronological block:

```text
window 0: train dates [0, T)      test dates [T, T + H)
window 1: train dates [0, T + S)  test dates [T + S, T + S + H)
```

where \(T\) is the minimum number of training dates, \(H\) is the test-window
length, and \(S\) is the step size. This better matches how forecasting models
would be evaluated through time.

### Reporting

Metrics are reported overall, by maturity bucket, and by exact maturity point:

```text
front_end  maturity <= 2 years
belly      2 years < maturity <= 10 years
long_end   maturity > 10 years
```

The exact-maturity output helps identify whether a representation works better
at specific points such as the front end, belly, or long end of the curve.

For volatility-regime classification, the baseline comparison is:

```math
\hat{g}_{t,h}
=
\arg\max_k
P(g_{t,h}=k \mid \mathbf{x}_t)
```

The current classical classifiers are a training-set mode baseline and
L2-regularized multinomial logistic regression. Logistic regression is a useful
transparent hurdle because it is regularized, interpretable, and can combine
multiple curve features without assuming a neural representation.

## Metrics

### Regression Metrics

RMSE:

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

MAE:

```math
MAE
=
\frac{1}{N}
\sum_i
\left|
y_i - \hat{y}_i
\right|
```

Directional accuracy:

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

### Cross-Sectional Rank IC

Cross-sectional rank IC is computed by date, country, and horizon. For each
date, predicted and realized target values are ranked across maturities, then a
Spearman correlation is computed:

```math
IC_t
=
\mathrm{corr}
\left(
\mathrm{rank}(\hat{y}_{t}^{(m)}),
\mathrm{rank}(y_{t}^{(m)})
\right)
```

The reported value is the mean of valid date-level rank correlations. This
metric asks whether a model orders curve points correctly, even when RMSE is not
the right objective.

## Current Limitations

The current evaluation is a first sanity check, not a final forecasting protocol.
Important next improvements:

- Report finer metrics by individual maturity and market regime.
- Extend the canonical supervised benchmark to curve-state transition targets.
- Add richer fixed-income return proxies when public bond-level cash-flow data is
  introduced; current carry/roll-down features are zero-curve approximations.
- Run and summarize walk-forward robustness as a first-class report.
- Only after the classical pipeline is stable, compare against learned
  representations such as autoencoders, masked reconstruction models,
  Transformers, and graph models.
