from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from yieldrep.config import ProjectConfig


COMPARISON_COLUMNS = [
    "representation",
    "country",
    "representation_family",
    "feature_dim",
    "observations",
    "dates",
    "clean_reconstruction_rmse",
    "clean_reconstruction_rank",
    "masked_reconstruction_rmse",
    "masked_reconstruction_rank",
    "pca_explained_variance_3",
    "pca_explained_variance_5",
    "best_regime_type",
    "best_regime_indicator",
    "best_regime_separation_ratio",
    "evidence_scope",
]


def build_representation_comparison(config: ProjectConfig) -> Path:
    """Write a compact comparison table for available representation artifacts."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    table = representation_comparison_table(config)
    table.to_csv(config.representation_comparison_table_path, index=False)
    return config.representation_comparison_table_path


def representation_comparison_table(config: ProjectConfig) -> pd.DataFrame:
    rows = [
        _reconstruction_rows(config),
        _engineered_feature_rows(config),
    ]
    comparison = pd.concat([row for row in rows if not row.empty], ignore_index=True)
    if comparison.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    comparison = _attach_masked_reconstruction(comparison, config)
    comparison = _attach_pca_variance(comparison, config)
    comparison = _attach_regime_separation(comparison, config)
    return comparison.loc[:, COMPARISON_COLUMNS].sort_values(
        ["country", "representation_family", "representation"]
    ).reset_index(drop=True)


def _reconstruction_rows(config: ProjectConfig) -> pd.DataFrame:
    if not config.reconstruction_oos_comparison_table_path.exists():
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    comparison = pd.read_csv(config.reconstruction_oos_comparison_table_path)
    clean = comparison.loc[comparison["reconstruction_task"] == "clean_reconstruction"].copy()
    if clean.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    clean = clean.sort_values(["country", "representation", "rmse", "n_components"])
    best = clean.groupby(["country", "representation"], as_index=False).first()
    best["representation_family"] = best["representation"].map(_representation_family)
    best["feature_dim"] = best["n_components"]
    best["clean_reconstruction_rmse"] = best["rmse"]
    best["clean_reconstruction_rank"] = best["rmse_rank"]
    best["evidence_scope"] = "curve reconstruction"
    return _align_columns(best)


def _engineered_feature_rows(config: ProjectConfig) -> pd.DataFrame:
    rows = [
        _feature_rows(
            path=config.curve_features_path,
            representation="curve",
            family="engineered_curve",
            feature_columns=[
                "level",
                "slope_10y_2y",
                "curvature_2s5s10s",
                "front_slope_2y_1y",
                "long_slope_30y_10y",
            ],
        ),
        _feature_rows(
            path=config.carry_roll_features_path,
            representation="carry_roll",
            family="engineered_curve",
            feature_columns=[
                "carry_1m",
                "roll_down_1m",
                "carry_3m",
                "roll_down_3m",
                "carry_12m",
                "roll_down_12m",
            ],
        ),
        _feature_rows(
            path=config.residual_features_path,
            representation="nelson_siegel_residual",
            family="relative_value",
            feature_columns=[
                "residual",
                "residual_z_60",
                "residual_z_252",
                "residual_change_1",
                "residual_change_5",
                "residual_vol_20",
            ],
        ),
        _feature_rows(
            path=config.policy_features_path,
            representation="policy",
            family="macro_policy",
            feature_columns=[
                "policy_rate",
                "policy_change_21d",
                "policy_change_63d",
                "policy_change_252d",
                "policy_2y_spread",
            ],
        ),
    ]
    return pd.concat([row for row in rows if not row.empty], ignore_index=True)


def _feature_rows(
    path: Path,
    representation: str,
    family: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    frame = pd.read_parquet(path)
    available_features = [column for column in feature_columns if column in frame.columns]
    if frame.empty or "country" not in frame.columns:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    summary = (
        frame.groupby("country", sort=True)
        .agg(observations=("country", "size"), dates=("date", "nunique"))
        .reset_index()
    )
    summary["representation"] = representation
    summary["representation_family"] = family
    summary["feature_dim"] = len(available_features)
    summary["evidence_scope"] = "direct benchmark features"
    return _align_columns(summary)


def _attach_masked_reconstruction(comparison: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    if not config.reconstruction_oos_comparison_table_path.exists():
        return comparison

    metrics = pd.read_csv(config.reconstruction_oos_comparison_table_path)
    masked = metrics.loc[metrics["reconstruction_task"] == "masked_maturity_reconstruction"].copy()
    if masked.empty:
        return comparison

    masked["representation"] = masked["representation"].str.replace("masked_", "", regex=False)
    masked = masked.sort_values(["country", "representation", "rmse", "n_components"])
    best = masked.groupby(["country", "representation"], as_index=False).first()
    best = best.loc[
        :,
        ["country", "representation", "rmse", "rmse_rank"],
    ].rename(
        columns={
            "rmse": "masked_reconstruction_rmse",
            "rmse_rank": "masked_reconstruction_rank",
        }
    )
    return comparison.drop(
        columns=["masked_reconstruction_rmse", "masked_reconstruction_rank"],
        errors="ignore",
    ).merge(best, on=["country", "representation"], how="left")


def _attach_pca_variance(comparison: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    rows = []
    for variance_path in sorted(config.pca_dir.glob("*_variance.parquet")):
        country = variance_path.stem.removesuffix("_variance").upper()
        variance = pd.read_parquet(variance_path)
        ratios = variance["explained_variance_ratio"].astype(float)
        rows.append(
            {
                "country": country,
                "representation": "pca",
                "pca_explained_variance_3": ratios.head(3).sum(),
                "pca_explained_variance_5": ratios.head(5).sum(),
            }
        )
    if not rows:
        return comparison

    pca_variance = pd.DataFrame(rows)
    return comparison.drop(
        columns=["pca_explained_variance_3", "pca_explained_variance_5"],
        errors="ignore",
    ).merge(pca_variance, on=["country", "representation"], how="left")


def _attach_regime_separation(comparison: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    if not config.learned_state_regime_summary_table_path.exists():
        return comparison

    regimes = pd.read_csv(config.learned_state_regime_summary_table_path)
    if regimes.empty:
        return comparison

    best = regimes.sort_values(
        ["country", "representation", "separation_ratio"],
        ascending=[True, True, False],
    ).groupby(["country", "representation"], as_index=False).first()
    best = best.loc[
        :,
        ["country", "representation", "regime_type", "indicator", "separation_ratio"],
    ].rename(
        columns={
            "regime_type": "best_regime_type",
            "indicator": "best_regime_indicator",
            "separation_ratio": "best_regime_separation_ratio",
        }
    )
    return comparison.drop(
        columns=[
            "best_regime_type",
            "best_regime_indicator",
            "best_regime_separation_ratio",
        ],
        errors="ignore",
    ).merge(best, on=["country", "representation"], how="left")


def _representation_family(representation: str) -> str:
    if representation == "pca":
        return "linear_factor"
    if representation == "nelson_siegel":
        return "parametric_curve"
    if representation in {"autoencoder", "transformer"}:
        return "learned_reconstruction"
    return "other"


def _align_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aligned = frame.copy()
    for column in COMPARISON_COLUMNS:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned.loc[:, COMPARISON_COLUMNS]
