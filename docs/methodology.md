# Methodology

This note defines the current research setup for yield curve representations,
targets, and baseline evaluation.

## Curve Panel

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

## PCA Representation

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

## Nelson-Siegel Representation

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

## Prediction Target

For horizon \(h\), the current target is the forward yield change:

```math
y_{t,h}^{(c,m)}
=
r_{t+h}^{(c,m)}
- r_t^{(c,m)}
```

The default horizons are 1, 5, and 20 available observations. These are
observation steps, not exact calendar days.

## Baseline Forecasting Evaluation

The supervised dataset joins representation features to the target:

```math
y_{t,h}^{(c,m)}
=
f\left(\mathbf{x}_t^{(c)}\right)
+ \varepsilon_{t,h}^{(c,m)}
```

where \(\mathbf{x}_t^{(c)}\) is either a PCA feature vector or a Nelson-Siegel
feature vector.

Current feature sets:

```text
PCA:
    PC1, PC2, PC3, PC4, PC5

Nelson-Siegel:
    beta_level, beta_slope, beta_curvature, rmse
```

The naive benchmark predicts the training-sample mean:

```math
\hat{y}
=
\frac{1}{N_{\text{train}}}
\sum_{i \in \text{train}} y_i
```

The linear benchmark is Ridge regression:

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

## Metrics

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
\operatorname{sign}(y_i)
=
\operatorname{sign}(\hat{y}_i)
\right]
```

## Current Limitations

The current evaluation is a first sanity check, not a final forecasting protocol.
Important next improvements:

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
