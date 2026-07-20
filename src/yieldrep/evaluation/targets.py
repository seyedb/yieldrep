from __future__ import annotations

from pathlib import Path

import pandas as pd

from yieldrep.config import ProjectConfig

TARGET_COLUMNS = (
    "date",
    "country",
    "maturity_years",
    "horizon_days",
    "yield",
    "future_yield",
    "target_yield_change",
)
STANDARDIZED_TARGET_COLUMNS = (
    "date",
    "country",
    "maturity_years",
    "horizon_days",
    "yield",
    "future_yield",
    "realized_vol",
    "target_yield_change",
    "target_standardized_yield_change",
)
RESIDUAL_TARGET_COLUMNS = (
    "date",
    "country",
    "maturity_years",
    "horizon_days",
    "residual",
    "future_residual",
    "target_residual_change",
    "fitted_yield",
)
VOL_TARGET_COLUMNS = (
    "date",
    "country",
    "maturity_years",
    "horizon_days",
    "realized_vol",
    "future_realized_vol",
    "target_vol_change",
    "future_vol_regime",
)
CURVE_VOL_REGIME_TARGET_COLUMNS = (
    "date",
    "country",
    "horizon_days",
    "realized_curve_vol",
    "future_curve_move_rms",
    "available_maturities",
)
CURVE_STATE_TARGET_BASE_COLUMNS = ("date", "country", "horizon_days")


def build_targets(config: ProjectConfig) -> Path:
    """Build forward yield-change targets from normalized curves."""
    curves = pd.read_parquet(config.curves_path)
    targets = make_forward_yield_change_targets(curves, config.targets.horizons_days)

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(config.targets_path, index=False)
    return config.targets_path


def build_standardized_targets(config: ProjectConfig) -> Path:
    """Build volatility-scaled forward yield-change targets."""
    curves = pd.read_parquet(config.curves_path)
    targets = make_forward_standardized_yield_change_targets(
        curves,
        horizons_days=config.targets.horizons_days,
        realized_vol_window=config.targets.realized_vol_window,
    )

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(config.standardized_targets_path, index=False)
    return config.standardized_targets_path


def build_residual_targets(config: ProjectConfig) -> Path:
    """Build forward Nelson-Siegel residual-change targets."""
    fitted = _read_nelson_siegel_fitted(config)
    targets = make_forward_residual_change_targets(fitted, config.targets.horizons_days)

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(config.residual_targets_path, index=False)
    return config.residual_targets_path


def build_vol_targets(config: ProjectConfig) -> list[Path]:
    """Build volatility target datasets from normalized curves."""
    curves = pd.read_parquet(config.curves_path)
    targets = make_forward_vol_change_targets(
        curves,
        horizons_days=config.targets.horizons_days,
        realized_vol_window=config.targets.realized_vol_window,
    )
    curve_regime_targets = make_forward_curve_vol_regime_targets(
        curves,
        horizons_days=config.targets.horizons_days,
        realized_vol_window=config.targets.realized_vol_window,
    )

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(config.vol_targets_path, index=False)
    curve_regime_targets.to_parquet(config.curve_vol_regime_targets_path, index=False)
    return [config.vol_targets_path, config.curve_vol_regime_targets_path]


def build_curve_state_targets(config: ProjectConfig) -> Path:
    """Build future PCA-state targets from country-level PCA scores."""
    scores = _read_pca_scores(config)
    targets = make_forward_curve_state_targets(
        scores,
        horizons_days=config.targets.horizons_days,
        n_components=min(3, config.pca.n_components),
    )

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(config.curve_state_targets_path, index=False)
    return config.curve_state_targets_path


def make_forward_yield_change_targets(
    curves: pd.DataFrame,
    horizons_days: list[int],
) -> pd.DataFrame:
    """Create yield(t+h) - yield(t) targets by country and maturity."""
    if not horizons_days:
        raise ValueError("At least one target horizon is required")
    if any(horizon <= 0 for horizon in horizons_days):
        raise ValueError("Target horizons must be positive")

    base = curves.loc[:, ["date", "country", "maturity_years", "yield"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    base = base.sort_values(["country", "maturity_years", "date"]).reset_index(drop=True)

    frames = [_make_horizon_targets(base, horizon) for horizon in horizons_days]
    return pd.concat(frames, ignore_index=True).loc[:, TARGET_COLUMNS]


def make_forward_standardized_yield_change_targets(
    curves: pd.DataFrame,
    horizons_days: list[int],
    realized_vol_window: int,
) -> pd.DataFrame:
    """Create forward yield changes scaled by trailing realized volatility."""
    if not horizons_days:
        raise ValueError("At least one target horizon is required")
    if any(horizon <= 0 for horizon in horizons_days):
        raise ValueError("Target horizons must be positive")
    if realized_vol_window <= 1:
        raise ValueError("realized_vol_window must be greater than 1")

    base = curves.loc[:, ["date", "country", "maturity_years", "yield"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    base = base.sort_values(["country", "maturity_years", "date"]).reset_index(drop=True)
    grouped = base.groupby(["country", "maturity_years"], sort=False)["yield"]
    base["realized_vol"] = grouped.transform(
        lambda series: series.diff().rolling(realized_vol_window).std()
    )

    frames = [_make_horizon_standardized_targets(base, horizon) for horizon in horizons_days]
    return pd.concat(frames, ignore_index=True).loc[:, STANDARDIZED_TARGET_COLUMNS]


def make_forward_residual_change_targets(
    fitted_curves: pd.DataFrame,
    horizons_days: list[int],
) -> pd.DataFrame:
    """Create residual(t+h) - residual(t) targets by country and maturity."""
    if not horizons_days:
        raise ValueError("At least one target horizon is required")
    if any(horizon <= 0 for horizon in horizons_days):
        raise ValueError("Target horizons must be positive")

    base = fitted_curves.loc[
        :,
        ["date", "country", "maturity_years", "residual", "fitted_yield"],
    ].copy()
    base["date"] = pd.to_datetime(base["date"])
    base = base.sort_values(["country", "maturity_years", "date"]).reset_index(drop=True)

    frames = [_make_horizon_residual_targets(base, horizon) for horizon in horizons_days]
    return pd.concat(frames, ignore_index=True).loc[:, RESIDUAL_TARGET_COLUMNS]


def make_forward_vol_change_targets(
    curves: pd.DataFrame,
    horizons_days: list[int],
    realized_vol_window: int,
) -> pd.DataFrame:
    """Create future realized-volatility change targets by country and maturity."""
    if not horizons_days:
        raise ValueError("At least one target horizon is required")
    if any(horizon <= 0 for horizon in horizons_days):
        raise ValueError("Target horizons must be positive")
    if realized_vol_window <= 1:
        raise ValueError("realized_vol_window must be greater than 1")

    base = curves.loc[:, ["date", "country", "maturity_years", "yield"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    base = base.sort_values(["country", "maturity_years", "date"]).reset_index(drop=True)
    grouped = base.groupby(["country", "maturity_years"], sort=False)["yield"]
    base["yield_change"] = grouped.diff()
    base["realized_vol"] = grouped.transform(
        lambda series: series.diff().rolling(realized_vol_window).std()
    )
    base["future_vol_regime"] = _future_vol_regime(base)

    frames = [_make_horizon_vol_targets(base, horizon) for horizon in horizons_days]
    return pd.concat(frames, ignore_index=True).loc[:, VOL_TARGET_COLUMNS]


def make_forward_curve_vol_regime_targets(
    curves: pd.DataFrame,
    horizons_days: list[int],
    realized_vol_window: int,
) -> pd.DataFrame:
    """Create curve-level future move magnitudes for volatility-regime classification.

    The target is the root-mean-square yield change across available maturities over
    the forward horizon. Low/medium/high regimes are assigned later inside each
    train/test split from training quantiles to avoid using full-sample thresholds.
    """
    if not horizons_days:
        raise ValueError("At least one target horizon is required")
    if any(horizon <= 0 for horizon in horizons_days):
        raise ValueError("Target horizons must be positive")
    if realized_vol_window <= 1:
        raise ValueError("realized_vol_window must be greater than 1")

    base = curves.loc[:, ["date", "country", "maturity_years", "yield"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    base = base.sort_values(["country", "maturity_years", "date"]).reset_index(drop=True)
    grouped = base.groupby(["country", "maturity_years"], sort=False)["yield"]
    base["yield_change"] = grouped.diff()
    base["realized_vol"] = grouped.transform(
        lambda series: series.diff().rolling(realized_vol_window).std()
    )

    frames = [_make_horizon_curve_vol_regime_targets(base, horizon) for horizon in horizons_days]
    return pd.concat(frames, ignore_index=True).loc[:, CURVE_VOL_REGIME_TARGET_COLUMNS]


def make_forward_curve_state_targets(
    scores: pd.DataFrame,
    horizons_days: list[int],
    n_components: int = 3,
) -> pd.DataFrame:
    """Create future PCA-score targets for curve-state classification."""
    if not horizons_days:
        raise ValueError("At least one target horizon is required")
    if any(horizon <= 0 for horizon in horizons_days):
        raise ValueError("Target horizons must be positive")
    if n_components <= 0:
        raise ValueError("n_components must be positive")

    components = [f"PC{index}" for index in range(1, n_components + 1)]
    missing = [component for component in components if component not in scores.columns]
    if missing:
        raise ValueError(f"Missing PCA score columns: {missing}")

    base = scores.loc[:, ["date", "country", *components]].copy()
    base["date"] = pd.to_datetime(base["date"])
    base = base.sort_values(["country", "date"]).reset_index(drop=True)

    frames = [_make_horizon_curve_state_targets(base, horizon, components) for horizon in horizons_days]
    columns = [*CURVE_STATE_TARGET_BASE_COLUMNS, *[f"future_{component}" for component in components]]
    return pd.concat(frames, ignore_index=True).loc[:, columns]


def _make_horizon_targets(curves: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    target = curves.copy()
    grouped = target.groupby(["country", "maturity_years"], sort=False)["yield"]
    target["future_yield"] = grouped.shift(-horizon_days)
    target["target_yield_change"] = target["future_yield"] - target["yield"]
    target["horizon_days"] = horizon_days
    return target.dropna(subset=["future_yield", "target_yield_change"])


def _make_horizon_standardized_targets(curves: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    target = _make_horizon_targets(curves, horizon_days)
    target["target_standardized_yield_change"] = (
        target["target_yield_change"] / target["realized_vol"]
    )
    return target.dropna(subset=["realized_vol", "target_standardized_yield_change"]).loc[
        target["realized_vol"] > 0
    ]


def _make_horizon_residual_targets(
    fitted_curves: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    target = fitted_curves.copy()
    grouped = target.groupby(["country", "maturity_years"], sort=False)["residual"]
    target["future_residual"] = grouped.shift(-horizon_days)
    target["target_residual_change"] = target["future_residual"] - target["residual"]
    target["horizon_days"] = horizon_days
    return target.dropna(subset=["future_residual", "target_residual_change"])


def _make_horizon_vol_targets(curves: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    target = curves.copy()
    grouped = target.groupby(["country", "maturity_years"], sort=False)["realized_vol"]
    target["future_realized_vol"] = grouped.shift(-horizon_days)
    target["target_vol_change"] = target["future_realized_vol"] - target["realized_vol"]
    target["future_vol_regime"] = target.groupby(["country", "maturity_years"], sort=False)[
        "future_vol_regime"
    ].shift(-horizon_days)
    target["horizon_days"] = horizon_days
    return target.dropna(
        subset=["realized_vol", "future_realized_vol", "target_vol_change", "future_vol_regime"]
    )


def _make_horizon_curve_vol_regime_targets(
    curves: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    target = curves.copy()
    grouped = target.groupby(["country", "maturity_years"], sort=False)["yield"]
    target["future_yield"] = grouped.shift(-horizon_days)
    target["forward_change_squared"] = (target["future_yield"] - target["yield"]) ** 2
    target = target.dropna(subset=["realized_vol", "forward_change_squared"])
    summary = (
        target.groupby(["date", "country"], sort=True)
        .agg(
            future_curve_move_rms=("forward_change_squared", lambda values: values.mean() ** 0.5),
            realized_curve_vol=("realized_vol", lambda values: (values.pow(2).mean()) ** 0.5),
            available_maturities=("maturity_years", "nunique"),
        )
        .reset_index()
    )
    summary["horizon_days"] = horizon_days
    return summary.dropna(subset=["future_curve_move_rms", "realized_curve_vol"])


def _make_horizon_curve_state_targets(
    scores: pd.DataFrame,
    horizon_days: int,
    components: list[str],
) -> pd.DataFrame:
    target = scores.copy()
    grouped = target.groupby("country", sort=False)
    future_columns: list[str] = []
    for component in components:
        future_column = f"future_{component}"
        target[future_column] = grouped[component].shift(-horizon_days)
        future_columns.append(future_column)
    target["horizon_days"] = horizon_days
    return target.dropna(subset=future_columns)


def _future_vol_regime(curves: pd.DataFrame) -> pd.Series:
    return curves.groupby(["country", "maturity_years"], sort=False)["realized_vol"].transform(
        _vol_regime
    )


def _vol_regime(realized_vol: pd.Series) -> pd.Series:
    ranked = realized_vol.rank(method="first")
    regimes = pd.qcut(
        ranked,
        q=3,
        labels=["low", "medium", "high"],
        duplicates="drop",
    )
    return regimes.astype("string")


def _read_nelson_siegel_fitted(config: ProjectConfig) -> pd.DataFrame:
    frames = [
        pd.read_parquet(fitted_path)
        for fitted_path in sorted(config.nelson_siegel_dir.glob("*_fitted.parquet"))
    ]
    if not frames:
        raise FileNotFoundError(
            f"No Nelson-Siegel fitted curve files found in {config.nelson_siegel_dir}"
        )
    return pd.concat(frames, ignore_index=True)


def _read_pca_scores(config: ProjectConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for scores_path in sorted(config.pca_dir.glob("*_scores.parquet")):
        country = scores_path.name.removesuffix("_scores.parquet").upper()
        scores = pd.read_parquet(scores_path)
        scores["country"] = country
        frames.append(scores)
    if not frames:
        raise FileNotFoundError(f"No PCA score files found in {config.pca_dir}")
    return pd.concat(frames, ignore_index=True)
