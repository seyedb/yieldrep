# Methodology

## Objective

The project evaluates representations of sovereign zero-coupon yield curves for
curve reconstruction, residual relative-value diagnostics, volatility regimes,
and macro/market regime conditioning.

The central comparison is between classical term-structure representations and
learned curve representations:

- PCA factors
- Nelson-Siegel factors and residuals
- engineered curve-shape, carry, roll-down, policy, and volatility features
- denoising autoencoder embeddings
- maturity-aware Transformer embeddings
- maturity-graph autoencoder embeddings

Representations are evaluated within task-specific protocols. Each model is
assessed against the task it is designed to solve.

## Data

The normalized curve schema is long format:

| Column | Meaning |
| --- | --- |
| `date` | observation date |
| `country` | market identifier |
| `maturity_years` | maturity in years |
| `yield` | zero-coupon yield |
| `source` | raw data source |

Current curve sources:

- Federal Reserve Gürkaynak-Sack-Wright nominal Treasury curve data
- Bank of Canada zero-coupon government curve data
- ECB euro-area yield curve data

The ECB source is treated as an aggregate euro-area curve (`EA`).

Additional public data used for conditioning and benchmark features:

- policy rates
- VIX and MOVE
- inflation
- unemployment

For each country \(c\), the normalized long data is pivoted into a curve panel:

```math
X^{(c)}
=
\left[
y_t^{(c,m_1)}, \ldots, y_t^{(c,m_M)}
\right]_{t=1}^{T_c}
```

where rows are dates and columns are maturities.

## Representations

### PCA

PCA is fit by country on the curve panel. It provides the main linear
reconstruction hurdle. PCA scores and explained variance are stored per market.

The reconstruction for \(K\) components is:

```math
\hat{x}_t
=
\mu
+
W_K z_{t,K}
```

where \(x_t\) is the curve vector, \(\mu\) is the sample mean, \(W_K\) are the
first \(K\) loading vectors, and \(z_{t,K}\) are the component scores.

### Nelson-Siegel

Nelson-Siegel represents each curve using level, slope, and curvature factors:

```math
y(\tau)
=
\beta_0
+
\beta_1
\left(
\frac{1-e^{-\tau/\lambda}}{\tau/\lambda}
\right)
+
\beta_2
\left(
\frac{1-e^{-\tau/\lambda}}{\tau/\lambda}
-
e^{-\tau/\lambda}
\right)
```

The project uses a fixed decay parameter and estimates factors by date and
country. Residuals from this fitted curve define the current relative-value
object.

### Engineered Features

Engineered baselines include:

- level
- slope
- curvature
- maturity-specific curve interactions
- lagged yield and residual changes
- carry and roll-down proxies
- policy-rate levels and changes
- realized curve volatility

These are direct features derived from observed curve or macro/market data.

Typical curve-shape features are:

```math
\mathrm{level}_t
=
y_t^{(10Y)}
```

```math
\mathrm{slope}_t
=
y_t^{(10Y)}
-
y_t^{(2Y)}
```

```math
\mathrm{curvature}_t
=
2y_t^{(5Y)}
-
y_t^{(2Y)}
-
y_t^{(10Y)}
```

A simple roll-down proxy compares the current yield at maturity \(m\) with the
interpolated yield at the rolled maturity:

```math
\mathrm{roll}_{t,h}^{(m)}
=
y_t^{(m)}
-
y_t^{(m-h)}
```

### Learned Representations

The autoencoder is a denoising MLP trained on chronological train data. It
receives masked curves and mask indicators, and is optimized for both clean and
masked reconstruction.

Let \(x_t\) be the standardized curve vector and \(b_t \in \{0,1\}^M\) be a
random mask. The corrupted input is:

```math
\tilde{x}_t
=
(1-b_t) \odot x_t
```

The autoencoder learns:

```math
z_t
=
f_\theta(\tilde{x}_t, b_t)
```

```math
\hat{x}_t
=
g_\phi(z_t)
```

with loss:

```math
\mathcal{L}
=
\frac{1}{|\mathcal{M}_t|}
\sum_{j:b_{t,j}=1}
\left(
x_{t,j}-\hat{x}_{t,j}
\right)^2
+
\alpha
\frac{1}{M}
\sum_{j=1}^{M}
\left(
x_{t,j}-\hat{x}_{t,j}^{clean}
\right)^2
```

The maturity Transformer treats maturities as tokens. Each token contains the
observed or masked yield and a mask indicator. The model adds a continuous
maturity coordinate, learned maturity embeddings, and a learned curve token. It
is evaluated on the same masked-maturity reconstruction task as the autoencoder.

For maturity \(m_j\), the token input is:

```math
u_{t,j}
=
\left[
\tilde{x}_{t,j},
b_{t,j}
\right]
+
e_j
+
p(m_j)
```

where \(e_j\) is a learned maturity embedding and \(p(m_j)\) is a continuous
maturity-position embedding. A learned curve token \(u_{t,0}\) produces the
curve-level state.

Learned embeddings are currently evaluated through reconstruction and
curve-state diagnostics.

For a learned state vector \(z_t\), regime separation is measured by comparing
latent means across regimes. For high and low regimes:

```math
D_{high,low}
=
\left\|
\bar{z}_{high}
-
\bar{z}_{low}
\right\|_2
```

The variance ratio compares between-regime dispersion with within-regime
dispersion:

```math
R
=
\frac{
\sum_g n_g
\left\|
\bar{z}_g-\bar{z}
\right\|_2^2
}{
\sum_g
\sum_{t \in g}
\left\|
z_t-\bar{z}_g
\right\|_2^2
}
```

where \(g\) indexes low, medium, and high regimes.

### Maturity Graph Dataset

For each country \(c\) and date \(t\), the maturity graph is:

```math
G_t^{(c)}
=
\left(
V_t^{(c)}, E^{(c)}
\right)
```

Each node corresponds to one maturity \(m\):

```math
v_{t,m}^{(c)}
=
\left[
y_t^{(c,m)},
\Delta y_t^{(c,m)},
\sigma_t^{(c,m)},
\mathrm{carry}_t^{(c,m)},
\mathrm{roll}_t^{(c,m)}
\right]
```

where \(\Delta y_t^{(c,m)} = y_t^{(c,m)} - y_{t-1}^{(c,m)}\), and
\(\sigma_t^{(c,m)}\) is trailing realized volatility of daily yield changes.

The initial edge set links adjacent maturities:

```math
(m_i, m_j) \in E_{adj}^{(c)}
\quad
\mathrm{if}
\quad
j=i+1
```

with distance weight:

```math
w_{ij}^{dist}
=
\frac{1}{1 + |m_i - m_j|}
```

Correlation edges connect each maturity to its strongest historical
yield-change peers:

```math
\rho_{ij}^{(c)}
=
\mathrm{corr}
\left(
\Delta y^{(c,m_i)},
\Delta y^{(c,m_j)}
\right)
```

These graph datasets are saved as node and edge parquet tables. They define the
data object for graph-learning experiments.

The first graph model uses this same maturity graph for masked maturity
reconstruction. Given node-feature matrix \(H_t^{(0)}\), a graph convolution
layer is:

```math
H_t^{(\ell+1)}
=
\phi
\left(
\tilde{A}^{(c)}
H_t^{(\ell)}
W^{(\ell)}
\right)
```

where \(\tilde{A}^{(c)}\) is the row-normalized maturity adjacency matrix with
self-loops, distance edges, and correlation edges. The decoder predicts each
masked maturity yield from its graph-updated node state:

```math
\hat{y}_t^{(c,m)}
=
q_\theta
\left(
h_{t,m}^{(L)}
\right)
```

The model is evaluated only on masked reconstruction at this stage. Its inputs
are observed curve-derived node features, not PCA, Nelson-Siegel, autoencoder,
or Transformer outputs.

## Targets

### Yield Change

For maturity \(m\), country \(c\), and horizon \(h\):

```math
\Delta y_{t,h}^{(c,m)}
=
y_{t+h}^{(c,m)}
-
y_t^{(c,m)}
```

Outright yield-change forecasting is retained as a secondary benchmark task.

### Residual Change

Nelson-Siegel residuals are:

```math
r_t^{(c,m)}
=
y_t^{(c,m)}
-
\hat{y}_t^{(c,m)}
```

Residual change is:

```math
\Delta r_{t,h}^{(c,m)}
=
r_{t+h}^{(c,m)}
-
r_t^{(c,m)}
```

Residuals define the relative-value object: rich or cheap maturities relative to
the fitted curve.

### Volatility Regimes

Curve volatility regimes are defined from future realized curve-move magnitude.
Labels are assigned with training-sample quantiles to avoid look-ahead.

For a horizon \(h\), realized curve-move magnitude is:

```math
v_{t,h}^{(c)}
=
\sqrt{
\frac{1}{M}
\sum_{m}
\left(
\Delta y_{t,h}^{(c,m)}
\right)^2
}
```

Regimes are assigned by training-sample quantiles:

```math
q_{low}, q_{high}
=
Q_{train}(v_{t,h}^{(c)})
```

Macro and market regimes are constructed from expanding historical quantiles of
public indicators such as inflation, unemployment, VIX, and MOVE.

For indicator \(a_t\), the expanding percentile at date \(t\) is:

```math
p_t
=
\frac{1}{t-1}
\sum_{s<t}
\mathbf{1}
\left[
a_s \le a_t
\right]
```

Low, medium, and high regimes are assigned from this historical percentile.

## Evaluation Protocol

### Splits

Default evaluation uses chronological splits:

```text
train = first 80% of dates
test  = last 20% of dates
```

All maturities for a date remain on the same side of the split. Multi-step
forecast targets use non-overlapping test windows by default.

### Task Families

| Task | Valid methods | Primary metrics |
| --- | --- | --- |
| Clean reconstruction | PCA, Nelson-Siegel, autoencoder, Transformer, maturity-graph autoencoder | RMSE, MAE |
| Masked maturity reconstruction | masked autoencoder, maturity Transformer, maturity-graph autoencoder | masked RMSE, masked MAE |
| Residual relative value | Nelson-Siegel residual diagnostics, maturity-aware curve features | spread score, hit rate, rank IC |
| Outright yield forecasting | lagged moves, curve-shape features, carry/roll-down proxies | RMSE, MAE |
| Volatility-regime classification | curve-shape, policy-rate, realized-volatility features | balanced accuracy, macro F1 |
| Macro/market RV regimes | residual RV diagnostics by regime | high-minus-low hit rate, high-minus-low rank IC |

The generated scenario map is:

```text
reports/tables/scenario_method_comparison.csv
```

The current scenario-level audit is:

```text
reports/tables/baseline_audit.csv
```

### Reconstruction Metrics

For observed curves \(x_i\) and fitted curves \(\hat{x}_i\):

```math
RMSE
=
\sqrt{
\frac{1}{N}
\sum_i
\left(
x_i-\hat{x}_i
\right)^2
}
```

```math
MAE
=
\frac{1}{N}
\sum_i
\left|
x_i-\hat{x}_i
\right|
```

For masked reconstruction, the sums are restricted to masked entries:

```math
RMSE_{mask}
=
\sqrt{
\frac{1}{|\mathcal{M}|}
\sum_{(t,j):b_{t,j}=1}
\left(
x_{t,j}-\hat{x}_{t,j}
\right)^2
}
```

### Residual RV Metrics

For each date, country, and horizon, maturities are ranked by predicted residual
change. The spread score is the realized average target of the top-ranked group
minus the bottom-ranked group:

```math
S_{t,h}^{(c)}
=
\frac{1}{|T_t|}
\sum_{m \in T_t}
\Delta r_{t,h}^{(c,m)}
-
\frac{1}{|B_t|}
\sum_{m \in B_t}
\Delta r_{t,h}^{(c,m)}
```

Rank IC is the cross-sectional correlation between predicted and realized
maturity ranks:

```math
IC_t
=
\mathrm{corr}
\left(
\mathrm{rank}(\hat{y}_t^{(m)}),
\mathrm{rank}(y_t^{(m)})
\right)
```

The mean-reversion hit rate for residual signal \(s_t^{(m)}\) is:

```math
H
=
\frac{1}{N}
\sum_{t,m}
\mathbf{1}
\left[
\mathrm{sign}(s_t^{(m)})
=
-
\mathrm{sign}(\Delta r_{t,h}^{(m)})
\right]
```

The RV regime scorecard compares residual mean-reversion diagnostics across
high and low macro/market regimes:

```math
\Delta H_{high-low}
=
H_{high}
-
H_{low}
```

```math
\Delta IC_{high-low}
=
IC_{high}
-
IC_{low}
```

```text
reports/tables/residual_rv_regime_scorecard.csv
reports/figures/residual_rv_regime_heatmap.html
```

Learned-state regime diagnostics are reported in:

```text
data/processed/learned_states/regime_states.parquet
reports/tables/learned_state_regime_summary.csv
reports/tables/learned_state_regime_means.csv
reports/figures/learned_state_regime_heatmap.html
reports/figures/learned_state_space_regimes.html
```

### Classification Metrics

Balanced accuracy:

```math
\mathrm{BalancedAccuracy}
=
\frac{1}{K}
\sum_{k=1}^{K}
\frac{TP_k}{TP_k + FN_k}
```

Macro F1:

```math
\mathrm{MacroF1}
=
\frac{1}{K}
\sum_{k=1}^{K}
\frac{
2\,\mathrm{Precision}_k\,\mathrm{Recall}_k
}{
\mathrm{Precision}_k + \mathrm{Recall}_k
}
```

## Current Scope

Implemented scope:

- public US, Canada, and euro-area curve data
- common long-format curve schema
- parquet data pipeline
- PCA and Nelson-Siegel baselines
- engineered curve, carry, roll-down, residual, policy, volatility, and macro features
- residual RV diagnostics
- market and macro regime conditioning
- autoencoder, maturity Transformer, and maturity-graph autoencoder reconstruction baselines
- learned-state regime diagnostics
- maturity-graph node and edge datasets
- Plotly figures and CSV scorecards

Planned next phase:

- consolidate graph-aware masked reconstruction as the main self-supervised task
- evaluate learned states against residual RV and volatility-regime tasks without
  using one model's outputs as another model's inputs or targets
- keep macro and market conditioning focused on interpretable regimes such as
  inflation, unemployment, VIX, and MOVE
