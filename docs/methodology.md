# Methodology

This project studies whether low-dimensional representations of sovereign yield
curves contain useful information for forecasting future curve moves.

## Curve Panel

For a country \(c\), date \(t\), and maturity \(m\), let

\[
r_t^{(c,m)}
\]

denote the observed zero-coupon yield. A yield curve at date \(t\) can be viewed
as a vector across maturities:

\[
\mathbf{r}_t^{(c)} =
\left[
r_t^{(c,m_1)}, r_t^{(c,m_2)}, \ldots, r_t^{(c,m_M)}
\right].
\]

The normalized dataset stores this panel in long format with:

```text
date, country, maturity_years, yield, source
```

## PCA Representation

Principal component analysis provides a linear statistical representation of the
yield curve. For each country, curves are pivoted into a date-by-maturity matrix
and standardized by maturity. PCA then finds orthogonal directions of maximum
historical variation:

\[
\mathbf{z}_t^{(c)} =
\left[
PC_{1,t}^{(c)}, PC_{2,t}^{(c)}, \ldots, PC_{K,t}^{(c)}
\right].
\]

In rates applications, the first few principal components often resemble level,
slope, and curvature factors, although signs and exact shapes are sample
dependent. PCA is a strong classical benchmark because it is simple, linear, and
usually explains most yield-curve variance with a small number of factors.

## Nelson-Siegel Representation

The Nelson-Siegel model imposes economically interpretable factor shapes. For
maturity \(m\) and fixed decay parameter \(\tau\):

\[
r(m) =
\beta_0
+ \beta_1 \frac{1 - e^{-m/\tau}}{m/\tau}
+ \beta_2
\left(
\frac{1 - e^{-m/\tau}}{m/\tau} - e^{-m/\tau}
\right).
\]

The coefficients are interpreted as:

```text
beta_level      broad level of the curve
beta_slope      short-end versus long-end slope
beta_curvature  intermediate-maturity hump or belly
```

With \(\tau\) fixed, the betas are estimated by ordinary least squares for each
country and date. The fitted residuals and RMSE are stored as diagnostics of how
well the parametric curve matches the observed curve.

## Prediction Target

For horizon \(h\), the current target is the forward yield change:

\[
y_{t,h}^{(c,m)} =
r_{t+h}^{(c,m)} - r_t^{(c,m)}.
\]

The default horizons are 1, 5, and 20 available observations. In the current
dataset this corresponds to business-day-style steps, not exact calendar days.

## Baseline Forecasting Evaluation

The current supervised evaluation joins representation features to the target:

\[
\hat{y}_{t,h}^{(c,m)} = f(\mathbf{x}_t^{(c)}),
\]

where \(\mathbf{x}_t^{(c)}\) is either a PCA feature vector or a Nelson-Siegel
feature vector.

The first model is a naive train-mean predictor:

\[
\hat{y} = \frac{1}{N_{\text{train}}}\sum_{i \in \text{train}} y_i.
\]

The second model is Ridge regression:

\[
\min_{\alpha,\beta}
\sum_i
\left(
y_i - \alpha - \mathbf{x}_i^\top \beta
\right)^2
+ \lambda \lVert \beta \rVert_2^2.
\]

Metrics are computed on the held-out test sample:

\[
RMSE = \sqrt{\frac{1}{N}\sum_i (y_i - \hat{y}_i)^2}
\]

\[
MAE = \frac{1}{N}\sum_i |y_i - \hat{y}_i|
\]

\[
\text{Directional Accuracy}
=
\frac{1}{N}
\sum_i
\mathbf{1}
\left[
\operatorname{sign}(y_i) =
\operatorname{sign}(\hat{y}_i)
\right].
\]

This is a first sanity-check evaluation, not a final forecasting protocol.

## Current Limitations

The current evaluation intentionally stays simple. Important next improvements:

- Use date-level train/test splits so all maturities from the same date remain
  in the same split.
- Add walk-forward or expanding-window validation.
- Report metrics by country, maturity, horizon, and regime.
- Add stronger classical baselines, including lagged yield changes,
  slope/curvature features, and carry/roll-down proxies.
- Evaluate residual and relative-value targets, not only outright yield changes.
- Add volatility and curve-state transition targets.
- Only after the classical pipeline is stable, compare against learned
  representations such as autoencoders, masked reconstruction models,
  Transformers, and graph models.
