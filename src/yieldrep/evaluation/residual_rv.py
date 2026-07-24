from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from yieldrep.config import ProjectConfig

KEY_COLUMNS = ["date", "country", "maturity_years"]
SUMMARY_COLUMNS = [
    "country",
    "horizon_days",
    "maturity_bucket",
    "sample",
    "signal",
    "rows",
    "dates",
    "mean_abs_signal",
    "mean_target_residual_change",
    "mean_convergence_score",
    "convergence_hit_rate",
    "mean_rank_ic",
    "rank_ic_dates",
]
REGIME_SUMMARY_COLUMNS = [
    "indicator",
    "market_vol_regime",
    "country",
    "horizon_days",
    "rows",
    "dates",
    "mean_convergence_score",
    "convergence_hit_rate",
    "mean_rank_ic",
    "rank_ic_dates",
]
MACRO_REGIME_SUMMARY_COLUMNS = [
    "indicator",
    "macro_regime",
    "country",
    "horizon_days",
    "rows",
    "dates",
    "mean_convergence_score",
    "convergence_hit_rate",
    "mean_rank_ic",
    "rank_ic_dates",
]


def build_residual_mean_reversion_report(config: ProjectConfig) -> Path:
    """Summarize direct Nelson-Siegel residual mean-reversion behavior."""
    features = pd.read_parquet(config.residual_features_path)
    targets = pd.read_parquet(config.residual_targets_path)
    report = residual_mean_reversion_summary(features, targets)

    config.tables_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.residual_mean_reversion_table_path, index=False)
    return config.residual_mean_reversion_table_path


def build_residual_rv_by_market_regime_report(config: ProjectConfig) -> Path:
    """Summarize residual mean reversion conditional on VIX/MOVE regimes."""
    features = pd.read_parquet(config.residual_features_path)
    targets = pd.read_parquet(config.residual_targets_path)
    regimes = pd.read_parquet(config.market_regimes_path)
    report = residual_rv_by_market_regime_summary(features, targets, regimes)

    config.tables_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.residual_rv_by_market_regime_table_path, index=False)
    return config.residual_rv_by_market_regime_table_path


def build_residual_rv_by_macro_regime_report(config: ProjectConfig) -> Path:
    """Summarize residual mean reversion conditional on macro regimes."""
    features = pd.read_parquet(config.residual_features_path)
    targets = pd.read_parquet(config.residual_targets_path)
    regimes = pd.read_parquet(config.macro_regimes_path)
    report = residual_rv_by_macro_regime_summary(features, targets, regimes)

    config.tables_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.residual_rv_by_macro_regime_table_path, index=False)
    return config.residual_rv_by_macro_regime_table_path


def residual_mean_reversion_summary(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    z_threshold: float = 1.0,
) -> pd.DataFrame:
    """Measure whether current NS residuals tend to move back toward zero.

    Positive convergence score means the future residual change has the opposite
    sign from the current residual signal. This is the direct relative-value
    question behind a rich/cheap residual screen.
    """
    data = _merge_residual_data(features, targets)
    if data.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    rows: list[dict[str, object]] = []
    for signal in ["residual", "residual_z_252"]:
        signal_frame = data.dropna(subset=[signal, "target_residual_change"]).copy()
        rows.extend(_summary_rows(signal_frame, signal=signal, sample="all"))

        stressed = signal_frame.loc[signal_frame["residual_z_252"].abs() >= z_threshold].copy()
        rows.extend(_summary_rows(stressed, signal=signal, sample=f"abs_z_ge_{z_threshold:g}"))

    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return (
        pd.DataFrame(rows)
        .loc[:, SUMMARY_COLUMNS]
        .sort_values(["country", "horizon_days", "maturity_bucket", "sample", "signal"])
        .reset_index(drop=True)
    )


def residual_rv_by_market_regime_summary(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    market_regimes: pd.DataFrame,
    z_threshold: float = 1.0,
) -> pd.DataFrame:
    """Split direct residual mean reversion by market-volatility regime."""
    data = _merge_residual_data(features, targets)
    data = data.dropna(subset=["residual_z_252", "target_residual_change"])
    data = data.loc[data["residual_z_252"].abs() >= z_threshold].copy()
    if data.empty or market_regimes.empty:
        return pd.DataFrame(columns=REGIME_SUMMARY_COLUMNS)

    data = _attach_market_regimes(data, market_regimes)
    if data.empty:
        return pd.DataFrame(columns=REGIME_SUMMARY_COLUMNS)

    rows: list[dict[str, object]] = []
    group_columns = ["indicator", "market_vol_regime", "country", "horizon_days"]
    for group_values, group in data.groupby(group_columns, sort=True):
        signal_values = group["residual_z_252"].to_numpy(dtype=float)
        target_values = group["target_residual_change"].to_numpy(dtype=float)
        convergence_score = -np.sign(signal_values) * target_values
        rank_ic = _rank_ic(group, "residual_z_252")
        rows.append(
            {
                **dict(zip(group_columns, group_values, strict=True)),
                "rows": len(group),
                "dates": group["date"].nunique(),
                "mean_convergence_score": float(np.mean(convergence_score)),
                "convergence_hit_rate": float(np.mean(convergence_score > 0.0)),
                "mean_rank_ic": rank_ic["mean_rank_ic"],
                "rank_ic_dates": rank_ic["rank_ic_dates"],
            }
        )
    if not rows:
        return pd.DataFrame(columns=REGIME_SUMMARY_COLUMNS)
    return (
        pd.DataFrame(rows)
        .loc[:, REGIME_SUMMARY_COLUMNS]
        .sort_values(["indicator", "country", "horizon_days", "market_vol_regime"])
        .reset_index(drop=True)
    )


def residual_rv_by_macro_regime_summary(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    macro_regimes: pd.DataFrame,
    z_threshold: float = 1.0,
) -> pd.DataFrame:
    """Split direct residual mean reversion by country-level macro regime."""
    data = _merge_residual_data(features, targets)
    data = data.dropna(subset=["residual_z_252", "target_residual_change"])
    data = data.loc[data["residual_z_252"].abs() >= z_threshold].copy()
    if data.empty or macro_regimes.empty:
        return pd.DataFrame(columns=MACRO_REGIME_SUMMARY_COLUMNS)

    data = _attach_macro_regimes(data, macro_regimes)
    if data.empty:
        return pd.DataFrame(columns=MACRO_REGIME_SUMMARY_COLUMNS)

    rows: list[dict[str, object]] = []
    group_columns = ["indicator", "macro_regime", "country", "horizon_days"]
    for group_values, group in data.groupby(group_columns, sort=True):
        signal_values = group["residual_z_252"].to_numpy(dtype=float)
        target_values = group["target_residual_change"].to_numpy(dtype=float)
        convergence_score = -np.sign(signal_values) * target_values
        rank_ic = _rank_ic(group, "residual_z_252")
        rows.append(
            {
                **dict(zip(group_columns, group_values, strict=True)),
                "rows": len(group),
                "dates": group["date"].nunique(),
                "mean_convergence_score": float(np.mean(convergence_score)),
                "convergence_hit_rate": float(np.mean(convergence_score > 0.0)),
                "mean_rank_ic": rank_ic["mean_rank_ic"],
                "rank_ic_dates": rank_ic["rank_ic_dates"],
            }
        )
    if not rows:
        return pd.DataFrame(columns=MACRO_REGIME_SUMMARY_COLUMNS)
    return (
        pd.DataFrame(rows)
        .loc[:, MACRO_REGIME_SUMMARY_COLUMNS]
        .sort_values(["indicator", "country", "horizon_days", "macro_regime"])
        .reset_index(drop=True)
    )


def _merge_residual_data(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [*KEY_COLUMNS, "residual", "residual_z_252"]
    target_columns = [*KEY_COLUMNS, "horizon_days", "target_residual_change"]
    data = features.loc[:, feature_columns].merge(
        targets.loc[:, target_columns],
        on=KEY_COLUMNS,
        how="inner",
    )
    data["date"] = pd.to_datetime(data["date"])
    data["maturity_bucket"] = _maturity_bucket(data["maturity_years"])
    return data


def _attach_market_regimes(data: pd.DataFrame, market_regimes: pd.DataFrame) -> pd.DataFrame:
    regime_columns = ["date", "indicator", "market_vol_regime"]
    regimes = market_regimes.loc[:, regime_columns].copy()
    regimes["date"] = pd.to_datetime(regimes["date"])
    regimes = regimes.dropna(subset=["market_vol_regime"])

    frames: list[pd.DataFrame] = []
    for indicator, indicator_regimes in regimes.groupby("indicator", sort=True):
        aligned = pd.merge_asof(
            data.sort_values("date"),
            indicator_regimes.sort_values("date"),
            on="date",
            direction="backward",
        )
        aligned["indicator"] = indicator
        frames.append(aligned.dropna(subset=["market_vol_regime"]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _attach_macro_regimes(data: pd.DataFrame, macro_regimes: pd.DataFrame) -> pd.DataFrame:
    regime_columns = ["date", "country", "indicator", "macro_regime"]
    regimes = macro_regimes.loc[:, regime_columns].copy()
    regimes["date"] = pd.to_datetime(regimes["date"])
    regimes = regimes.dropna(subset=["macro_regime"])

    frames: list[pd.DataFrame] = []
    group_columns = ["country", "indicator"]
    for (country, indicator), indicator_regimes in regimes.groupby(group_columns, sort=True):
        country_data = data.loc[data["country"] == country].copy()
        if country_data.empty:
            continue
        aligned = pd.merge_asof(
            country_data.sort_values("date"),
            indicator_regimes.sort_values("date"),
            on="date",
            by="country",
            direction="backward",
        )
        aligned["indicator"] = indicator
        frames.append(aligned.dropna(subset=["macro_regime"]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _summary_rows(data: pd.DataFrame, signal: str, sample: str) -> list[dict[str, object]]:
    if data.empty:
        return []

    rows: list[dict[str, object]] = []
    group_columns = ["country", "horizon_days", "maturity_bucket"]
    for group_values, group in data.groupby(group_columns, sort=True, observed=True):
        signal_values = group[signal].to_numpy(dtype=float)
        target_values = group["target_residual_change"].to_numpy(dtype=float)
        convergence_score = -np.sign(signal_values) * target_values
        rank_ic = _rank_ic(group, signal)

        rows.append(
            {
                **dict(zip(group_columns, group_values, strict=True)),
                "sample": sample,
                "signal": signal,
                "rows": len(group),
                "dates": group["date"].nunique(),
                "mean_abs_signal": float(np.mean(np.abs(signal_values))),
                "mean_target_residual_change": float(np.mean(target_values)),
                "mean_convergence_score": float(np.mean(convergence_score)),
                "convergence_hit_rate": float(np.mean(convergence_score > 0.0)),
                "mean_rank_ic": rank_ic["mean_rank_ic"],
                "rank_ic_dates": rank_ic["rank_ic_dates"],
            }
        )
    return rows


def _rank_ic(group: pd.DataFrame, signal: str) -> dict[str, object]:
    rank_ics: list[float] = []
    for _, date_group in group.groupby("date", sort=False):
        if date_group[signal].nunique(dropna=True) < 2:
            continue
        expected_change_rank = (-date_group[signal]).rank(method="average")
        realized_change_rank = date_group["target_residual_change"].rank(method="average")
        correlation = expected_change_rank.corr(realized_change_rank, method="pearson")
        if pd.notna(correlation):
            rank_ics.append(float(correlation))
    return {
        "mean_rank_ic": float(np.mean(rank_ics)) if rank_ics else float("nan"),
        "rank_ic_dates": len(rank_ics),
    }


def _maturity_bucket(maturity_years: pd.Series) -> pd.Series:
    return pd.cut(
        maturity_years,
        bins=[0.0, 2.0, 10.0, float("inf")],
        labels=["front_end", "belly", "long_end"],
        right=True,
    ).astype("string")
