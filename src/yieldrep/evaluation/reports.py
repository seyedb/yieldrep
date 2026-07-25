from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from yieldrep.config import ProjectConfig
from yieldrep.evaluation.datasets import (
    make_supervised_feature_dataset,
)
from yieldrep.evaluation.residual_rv import build_residual_mean_reversion_report
from yieldrep.evaluation.residual_rv import build_residual_rv_by_macro_regime_report
from yieldrep.evaluation.residual_rv import build_residual_rv_by_market_regime_report
from yieldrep.models.baselines import evaluate_baseline_frames
from yieldrep.models.forecasting import (
    TargetFrameSpec,
    feature_sets,
    rank_supervised_forecasts,
    summarize_supervised_forecasts,
    supervised_forecast_frames_from_unsplit_data,
)

SUMMARY_GROUP_COLUMNS = ["target", "representation", "model"]
BUCKET_GROUP_COLUMNS = [
    "target",
    "country",
    "horizon_days",
    "maturity_bucket",
    "representation",
    "model",
]
RANK_GROUP_COLUMNS = ["target", "country", "horizon_days"]
METRIC_COLUMNS = ["rmse", "mae", "directional_accuracy", "mean_rank_ic", "rank_ic_dates"]


def summarize_baselines(config: ProjectConfig, top_n: int = 100) -> list[Path]:
    """Write human-readable CSV summaries from baseline metric parquets."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_parquet(config.baseline_metrics_path)

    summary = summarize_metrics(metrics)
    summary.to_csv(config.baseline_summary_table_path, index=False)

    rank_table = rank_baselines(metrics)
    rank_table.to_csv(config.baseline_rank_table_path, index=False)

    residual_rv_rank_ic = residual_relative_value_rank_ic_summary(rank_table)
    residual_rv_rank_ic.to_csv(config.residual_relative_value_rank_ic_table_path, index=False)
    residual_rv_rank_ic_coverage = residual_relative_value_rank_ic_coverage(rank_table)
    residual_rv_rank_ic_coverage.to_csv(
        config.residual_relative_value_rank_ic_coverage_table_path,
        index=False,
    )
    residual_rv_spread = residual_relative_value_spread_summary(config)
    residual_rv_spread.to_csv(config.residual_relative_value_spread_table_path, index=False)
    residual_rv_benchmark = residual_relative_value_benchmark_summary(
        spread_summary=residual_rv_spread,
        rank_ic_summary=residual_rv_rank_ic,
    )
    residual_rv_benchmark.to_csv(
        config.residual_relative_value_benchmark_table_path,
        index=False,
    )
    residual_mean_reversion_path = None
    residual_mean_reversion = pd.DataFrame()
    if config.residual_mean_reversion_table_path.exists():
        residual_mean_reversion_path = config.residual_mean_reversion_table_path
        residual_mean_reversion = pd.read_csv(residual_mean_reversion_path)
    elif config.residual_features_path.exists() and config.residual_targets_path.exists():
        residual_mean_reversion_path = build_residual_mean_reversion_report(config)
        residual_mean_reversion = pd.read_csv(residual_mean_reversion_path)
    residual_rv_overview = residual_relative_value_overview(
        benchmark=residual_rv_benchmark,
        mean_reversion=residual_mean_reversion,
    )
    residual_rv_overview.to_csv(config.residual_relative_value_overview_table_path, index=False)
    residual_rv_by_market_regime_path = None
    if (
        config.residual_features_path.exists()
        and config.residual_targets_path.exists()
        and config.market_regimes_path.exists()
    ):
        residual_rv_by_market_regime_path = build_residual_rv_by_market_regime_report(config)
    market_regime_rv_summary_path = None
    if config.residual_rv_by_market_regime_table_path.exists():
        market_regime_rv_summary = market_regime_rv_summary_table(
            pd.read_csv(config.residual_rv_by_market_regime_table_path)
        )
        market_regime_rv_summary.to_csv(
            config.market_regime_rv_summary_table_path,
            index=False,
        )
        market_regime_rv_summary_path = config.market_regime_rv_summary_table_path
    residual_rv_by_macro_regime_path = None
    if (
        config.residual_features_path.exists()
        and config.residual_targets_path.exists()
        and config.macro_regimes_path.exists()
    ):
        residual_rv_by_macro_regime_path = build_residual_rv_by_macro_regime_report(config)

    winners = baseline_winners(rank_table)
    winners.to_csv(config.baseline_winners_table_path, index=False)

    volatility_regime = volatility_regime_summary(config)
    volatility_regime.to_csv(config.volatility_regime_table_path, index=False)
    volatility_regime_benchmark = volatility_regime_benchmark_summary(volatility_regime)
    volatility_regime_benchmark.to_csv(
        config.volatility_regime_benchmark_table_path,
        index=False,
    )
    curve_state = curve_state_summary(config)
    curve_state.to_csv(config.curve_state_table_path, index=False)
    curve_state_transition_benchmark = curve_state_transition_benchmark_summary(curve_state)
    curve_state_transition_benchmark.to_csv(
        config.curve_state_transition_benchmark_table_path,
        index=False,
    )
    sequence_readiness = sequence_readiness_summary(curve_state_transition_benchmark)
    sequence_readiness.to_csv(config.sequence_readiness_summary_table_path, index=False)
    curve_state_probe_importance = curve_state_probe_importance_summary(config)
    curve_state_probe_importance.to_csv(
        config.curve_state_probe_importance_table_path,
        index=False,
    )
    ae_classical_factor_correlations = ae_classical_factor_correlation_summary(config)
    ae_classical_factor_correlations.to_csv(
        config.ae_classical_factor_correlations_table_path,
        index=False,
    )

    bucket_summary = summarize_metrics(
        pd.read_parquet(config.baseline_metrics_by_maturity_path),
        group_columns=BUCKET_GROUP_COLUMNS,
    )
    bucket_summary.to_csv(config.baseline_by_maturity_bucket_table_path, index=False)

    residual_rv = residual_relative_value_summary(bucket_summary)
    residual_rv.to_csv(config.residual_relative_value_table_path, index=False)

    maturity_point_top = top_maturity_point_metrics(
        pd.read_parquet(config.baseline_metrics_by_maturity_point_path),
        top_n=top_n,
    )
    maturity_point_top.to_csv(config.baseline_by_maturity_point_top_table_path, index=False)

    benchmark_conclusions = benchmark_conclusion_summary(
        config=config,
        rank_table=rank_table,
        residual_rv_benchmark=residual_rv_benchmark,
        volatility_regime_benchmark=volatility_regime_benchmark,
        curve_state_transition_benchmark=curve_state_transition_benchmark,
    )
    benchmark_conclusions.to_csv(config.benchmark_conclusions_table_path, index=False)

    output_paths = [
        config.baseline_summary_table_path,
        config.baseline_rank_table_path,
        config.residual_relative_value_rank_ic_table_path,
        config.residual_relative_value_rank_ic_coverage_table_path,
        config.residual_relative_value_spread_table_path,
        config.residual_relative_value_benchmark_table_path,
        config.residual_relative_value_overview_table_path,
        config.baseline_winners_table_path,
        config.volatility_regime_table_path,
        config.volatility_regime_benchmark_table_path,
        config.curve_state_table_path,
        config.curve_state_transition_benchmark_table_path,
        config.sequence_readiness_summary_table_path,
        config.curve_state_probe_importance_table_path,
        config.ae_classical_factor_correlations_table_path,
        config.benchmark_conclusions_table_path,
        config.baseline_by_maturity_bucket_table_path,
        config.residual_relative_value_table_path,
        config.baseline_by_maturity_point_top_table_path,
    ]
    if residual_mean_reversion_path is not None:
        output_paths.insert(7, residual_mean_reversion_path)
    if residual_rv_by_market_regime_path is not None:
        output_paths.insert(8, residual_rv_by_market_regime_path)
    if market_regime_rv_summary_path is not None:
        output_paths.insert(9, market_regime_rv_summary_path)
    if residual_rv_by_macro_regime_path is not None:
        output_paths.insert(10, residual_rv_by_macro_regime_path)
    return output_paths


def build_overlap_sensitivity_report(config: ProjectConfig) -> Path:
    """Compare baseline ranks with overlapping and non-overlapping target windows."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)

    overlapping = _evaluate_with_target_window(config, non_overlapping_targets=False)
    non_overlapping = _evaluate_with_target_window(config, non_overlapping_targets=True)
    report = overlap_sensitivity_table(overlapping, non_overlapping)
    report.to_csv(config.overlap_sensitivity_table_path, index=False)
    return config.overlap_sensitivity_table_path


def build_supervised_walk_forward_report(config: ProjectConfig) -> list[Path]:
    """Evaluate canonical supervised benchmarks with expanding walk-forward splits."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)

    walk_config = config.model_copy(
        update={"evaluation": config.evaluation.model_copy(update={"method": "walk_forward"})}
    )
    target_specs = _walk_forward_target_specs(walk_config)
    if not target_specs:
        return []

    frames = supervised_forecast_frames_from_unsplit_data(
        target_specs=target_specs,
        feature_sets=feature_sets(walk_config),
        config=walk_config,
    )
    summary = summarize_supervised_forecasts(frames.metrics)
    summary.to_csv(config.supervised_walk_forward_summary_table_path, index=False)
    rank = rank_supervised_forecasts(frames.metrics)
    rank.to_csv(config.supervised_walk_forward_rank_table_path, index=False)

    output_paths = [
        config.supervised_walk_forward_summary_table_path,
        config.supervised_walk_forward_rank_table_path,
    ]
    if config.supervised_forecast_metrics_path.exists():
        comparison = supervised_walk_forward_comparison(
            date_ordered_metrics=pd.read_parquet(config.supervised_forecast_metrics_path),
            walk_forward_metrics=frames.metrics,
        )
        comparison.to_csv(config.supervised_walk_forward_comparison_table_path, index=False)
        output_paths.append(config.supervised_walk_forward_comparison_table_path)

    return output_paths


def supervised_walk_forward_comparison(
    date_ordered_metrics: pd.DataFrame,
    walk_forward_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Compare date-ordered and walk-forward supervised benchmark ranks."""
    date_ordered = _rank_for_supervised_method(date_ordered_metrics, "date_ordered")
    walk_forward = _rank_for_supervised_method(walk_forward_metrics, "walk_forward")
    join_columns = [*RANK_GROUP_COLUMNS, "representation", "model"]
    report = date_ordered.merge(walk_forward, on=join_columns, how="outer")
    report["rmse_change_walk_forward_minus_date_ordered"] = (
        report["walk_forward_mean_rmse"] - report["date_ordered_mean_rmse"]
    )
    report["rank_change_walk_forward_minus_date_ordered"] = (
        report["walk_forward_rank"] - report["date_ordered_rank"]
    )
    return report.sort_values(
        [*RANK_GROUP_COLUMNS, "walk_forward_rank", "date_ordered_rank", "representation", "model"],
        na_position="last",
    ).reset_index(drop=True)


def overlap_sensitivity_table(
    overlapping_metrics: pd.DataFrame,
    non_overlapping_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact side-by-side comparison of two evaluation protocols."""
    overlapping = _rank_for_target_window(overlapping_metrics, target_window="overlapping")
    non_overlapping = _rank_for_target_window(
        non_overlapping_metrics,
        target_window="non_overlapping",
    )
    join_columns = [*RANK_GROUP_COLUMNS, "representation", "model"]
    report = overlapping.merge(non_overlapping, on=join_columns, how="outer")
    report["rmse_change_non_overlapping_minus_overlapping"] = (
        report["non_overlapping_mean_rmse"] - report["overlapping_mean_rmse"]
    )
    report["rank_change_non_overlapping_minus_overlapping"] = (
        report["non_overlapping_rank"] - report["overlapping_rank"]
    )
    return report.sort_values(
        [*RANK_GROUP_COLUMNS, "non_overlapping_rank", "overlapping_rank", "representation", "model"],
        na_position="last",
    ).reset_index(drop=True)


def _walk_forward_target_specs(config: ProjectConfig) -> list[TargetFrameSpec]:
    curves = pd.read_parquet(config.curves_path)
    specs: list[TargetFrameSpec] = []
    if config.targets_path.exists():
        targets = pd.read_parquet(config.targets_path)
        specs.append(
            TargetFrameSpec(
                target="yield_change",
                data=make_supervised_feature_dataset(config, targets, curves),
                target_column="target_yield_change",
            )
        )
    if config.residual_targets_path.exists():
        residual_targets = pd.read_parquet(config.residual_targets_path)
        specs.append(
            TargetFrameSpec(
                target="residual_change",
                data=make_supervised_feature_dataset(config, residual_targets, curves),
                target_column="target_residual_change",
            )
        )
    if config.vol_targets_path.exists():
        vol_targets = pd.read_parquet(config.vol_targets_path)
        specs.append(
            TargetFrameSpec(
                target="vol_change",
                data=make_supervised_feature_dataset(config, vol_targets, curves),
                target_column="target_vol_change",
            )
        )
    return specs


def _rank_for_supervised_method(metrics: pd.DataFrame, method: str) -> pd.DataFrame:
    rank = rank_supervised_forecasts(metrics)
    columns = [
        *RANK_GROUP_COLUMNS,
        "representation",
        "model",
        "mean_rmse",
        "mean_mae",
        "mean_directional_accuracy",
        "mean_pct_improvement_vs_train_mean",
        "rank",
        "rmse_gap_to_best",
        "pct_gap_to_best",
    ]
    if "mean_test_dates" in rank.columns:
        columns.append("mean_test_dates")
    renamed = {
        column: f"{method}_{column}"
        for column in columns
        if column not in [*RANK_GROUP_COLUMNS, "representation", "model"]
    }
    return rank.loc[:, columns].rename(columns=renamed)


def summarize_metrics(
    metrics: pd.DataFrame,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate metric rows into compact mean-performance tables."""
    groups = group_columns or SUMMARY_GROUP_COLUMNS
    aggregations = {
        "rows": ("rmse", "size"),
        "countries": ("country", "nunique"),
        "horizons": ("horizon_days", "nunique"),
        "mean_rmse": ("rmse", "mean"),
        "mean_mae": ("mae", "mean"),
        "mean_directional_accuracy": ("directional_accuracy", "mean"),
    }
    if "mean_rank_ic" in metrics.columns:
        aggregations["mean_rank_ic"] = ("mean_rank_ic", "mean")
    if "rank_ic_dates" in metrics.columns:
        aggregations["rank_ic_dates"] = ("rank_ic_dates", "sum")

    summary = (
        metrics.groupby(groups, sort=True)
        .agg(**aggregations)
        .reset_index()
    )
    return summary.sort_values([*groups, "mean_rmse"]).reset_index(drop=True)


def top_maturity_point_metrics(metrics: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """Return the best exact-maturity metric rows ranked by RMSE."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    columns = [
        "target",
        "representation",
        "model",
        "split_method",
        "window_id",
        "country",
        "horizon_days",
        "maturity_years",
        *METRIC_COLUMNS,
        "train_rows",
        "test_rows",
        "train_dates",
        "test_dates",
    ]
    available_columns = [column for column in columns if column in metrics.columns]
    return (
        metrics.sort_values(["rmse", "mae", "target", "representation", "model"])
        .loc[:, available_columns]
        .head(top_n)
        .reset_index(drop=True)
    )


def residual_relative_value_summary(bucket_summary: pd.DataFrame) -> pd.DataFrame:
    """Rank residual-change baselines by country, horizon, and maturity bucket."""
    columns = [
        "country",
        "horizon_days",
        "maturity_bucket",
        "representation",
        "model",
        "rows",
        "mean_rmse",
        "mean_mae",
        "mean_directional_accuracy",
        "mean_rank_ic",
        "rank_ic_dates",
        "rank",
        "rmse_gap_to_best",
        "pct_gap_to_best",
    ]
    residual = bucket_summary.loc[bucket_summary["target"] == "residual_change"].copy()
    if residual.empty:
        return pd.DataFrame(columns=columns)

    rank_groups = ["country", "horizon_days", "maturity_bucket"]
    naive = _naive_residual_rows(residual, rank_groups)
    residual = pd.concat(
        [naive, residual.loc[residual["model"] != "train_mean"]],
        ignore_index=True,
    )
    residual["rank"] = residual.groupby(rank_groups)["mean_rmse"].rank(
        method="min",
        ascending=True,
    )
    best_rmse = residual.groupby(rank_groups)["mean_rmse"].transform("min")
    residual["rmse_gap_to_best"] = residual["mean_rmse"] - best_rmse
    residual["pct_gap_to_best"] = residual["rmse_gap_to_best"] / best_rmse

    available_columns = [column for column in columns if column in residual.columns]
    return (
        residual.sort_values([*rank_groups, "rank", "mean_mae", "representation", "model"])
        .loc[:, available_columns]
        .reset_index(drop=True)
    )


def volatility_regime_summary(config: ProjectConfig) -> pd.DataFrame:
    """Rank curve-level volatility-regime classifiers by balanced accuracy."""
    columns = [
        "country",
        "horizon_days",
        "representation",
        "model",
        "rows",
        "mean_balanced_accuracy",
        "mean_macro_f1",
        "mean_accuracy",
        "mean_true_classes",
        "mean_test_dates",
        "rank",
        "balanced_accuracy_gap_to_best",
    ]
    if not config.baseline_classification_metrics_path.exists():
        return pd.DataFrame(columns=columns)

    metrics = pd.read_parquet(config.baseline_classification_metrics_path)
    metrics = metrics.loc[metrics["target"] == "curve_vol_regime"].copy()
    if metrics.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        metrics.groupby(["country", "horizon_days", "representation", "model"], sort=True)
        .agg(
            rows=("balanced_accuracy", "size"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_true_classes=("true_classes", "mean"),
            mean_test_dates=("test_dates", "mean"),
        )
        .reset_index()
    )
    rank_groups = ["country", "horizon_days"]
    summary["rank"] = summary.groupby(rank_groups)["mean_balanced_accuracy"].rank(
        method="min",
        ascending=False,
    )
    best = summary.groupby(rank_groups)["mean_balanced_accuracy"].transform("max")
    summary["balanced_accuracy_gap_to_best"] = best - summary["mean_balanced_accuracy"]
    return (
        summary.loc[:, columns]
        .sort_values(
            [*rank_groups, "rank", "mean_macro_f1", "representation", "model"],
            ascending=[True, True, True, False, True, True],
        )
        .reset_index(drop=True)
    )


def volatility_regime_benchmark_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Build a compact hurdle table for curve-volatility regime classification."""
    columns = [
        "country",
        "horizon_days",
        "best_model",
        "best_balanced_accuracy",
        "curve_vol_balanced_accuracy",
        "policy_balanced_accuracy",
        "pca_balanced_accuracy",
        "autoencoder_balanced_accuracy",
        "nelson_siegel_balanced_accuracy",
        "curve_balanced_accuracy",
        "policy_beats_curve_vol",
        "pca_beats_curve_vol",
        "autoencoder_beats_curve_vol",
        "nelson_siegel_beats_curve_vol",
        "curve_beats_curve_vol",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)

    logistic = summary.loc[summary["model"] == "logistic_l2"].copy()
    if logistic.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for group_values, group in logistic.groupby(["country", "horizon_days"], sort=True):
        country, horizon_days = group_values
        best = group.sort_values(
            ["rank", "mean_macro_f1", "representation"],
            ascending=[True, False, True],
        ).iloc[0]
        scores = {
            str(row["representation"]): float(row["mean_balanced_accuracy"])
            for row in group.to_dict("records")
        }
        curve_vol_score = scores.get("curve_vol")
        rows.append(
            {
                "country": country,
                "horizon_days": horizon_days,
                "best_model": f"{best['representation']}/{best['model']}",
                "best_balanced_accuracy": float(best["mean_balanced_accuracy"]),
                "curve_vol_balanced_accuracy": curve_vol_score,
                "policy_balanced_accuracy": scores.get("policy"),
                "pca_balanced_accuracy": scores.get("pca"),
                "autoencoder_balanced_accuracy": scores.get("autoencoder"),
                "nelson_siegel_balanced_accuracy": scores.get("nelson_siegel"),
                "curve_balanced_accuracy": scores.get("curve"),
                "policy_beats_curve_vol": _beats_hurdle(scores.get("policy"), curve_vol_score),
                "pca_beats_curve_vol": _beats_hurdle(scores.get("pca"), curve_vol_score),
                "autoencoder_beats_curve_vol": _beats_hurdle(
                    scores.get("autoencoder"),
                    curve_vol_score,
                ),
                "nelson_siegel_beats_curve_vol": _beats_hurdle(
                    scores.get("nelson_siegel"),
                    curve_vol_score,
                ),
                "curve_beats_curve_vol": _beats_hurdle(scores.get("curve"), curve_vol_score),
            }
        )

    return pd.DataFrame(rows).loc[:, columns]


def curve_state_summary(config: ProjectConfig) -> pd.DataFrame:
    """Rank future PCA-state classifiers by balanced accuracy."""
    columns = [
        "state",
        "country",
        "horizon_days",
        "representation",
        "model",
        "rows",
        "mean_balanced_accuracy",
        "mean_macro_f1",
        "mean_accuracy",
        "mean_true_classes",
        "mean_test_dates",
        "rank",
        "balanced_accuracy_gap_to_best",
    ]
    if not config.baseline_classification_metrics_path.exists():
        return pd.DataFrame(columns=columns)

    metrics = pd.read_parquet(config.baseline_classification_metrics_path)
    metrics = metrics.loc[metrics["target"].str.startswith("curve_state_pc")].copy()
    if metrics.empty:
        return pd.DataFrame(columns=columns)

    metrics["state"] = metrics["target"].str.removeprefix("curve_state_")
    summary = (
        metrics.groupby(
            ["state", "country", "horizon_days", "representation", "model"],
            sort=True,
        )
        .agg(
            rows=("balanced_accuracy", "size"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_true_classes=("true_classes", "mean"),
            mean_test_dates=("test_dates", "mean"),
        )
        .reset_index()
    )
    rank_groups = ["state", "country", "horizon_days"]
    summary["rank"] = summary.groupby(rank_groups)["mean_balanced_accuracy"].rank(
        method="min",
        ascending=False,
    )
    best = summary.groupby(rank_groups)["mean_balanced_accuracy"].transform("max")
    summary["balanced_accuracy_gap_to_best"] = best - summary["mean_balanced_accuracy"]
    return (
        summary.loc[:, columns]
        .sort_values(
            [*rank_groups, "rank", "mean_macro_f1", "representation", "model"],
            ascending=[True, True, True, True, False, True, True],
        )
        .reset_index(drop=True)
    )


def curve_state_transition_benchmark_summary(curve_state: pd.DataFrame) -> pd.DataFrame:
    """Build a compact benchmark table for future PCA-state classification."""
    columns = [
        "state",
        "country",
        "horizon_days",
        "best_model",
        "best_balanced_accuracy",
        "autoencoder_rank",
        "autoencoder_balanced_accuracy",
        "autoencoder_gap_to_best",
        "autoencoder_temporal_rank",
        "autoencoder_temporal_balanced_accuracy",
        "autoencoder_temporal_gap_to_best",
        "pca_rank",
        "pca_balanced_accuracy",
        "pca_temporal_rank",
        "pca_temporal_balanced_accuracy",
        "nelson_siegel_rank",
        "nelson_siegel_balanced_accuracy",
        "nelson_siegel_temporal_rank",
        "nelson_siegel_temporal_balanced_accuracy",
        "curve_rank",
        "curve_balanced_accuracy",
        "policy_rank",
        "policy_balanced_accuracy",
        "learned_representation_status",
    ]
    if curve_state.empty:
        return pd.DataFrame(columns=columns)

    logistic = curve_state.loc[
        (curve_state["model"] == "logistic_l2")
        & (curve_state["mean_true_classes"] >= 2)
    ].copy()
    if logistic.empty:
        return pd.DataFrame(columns=columns)

    rank_groups = ["state", "country", "horizon_days"]
    logistic["rank"] = logistic.groupby(rank_groups)["mean_balanced_accuracy"].rank(
        method="min",
        ascending=False,
    )
    best_score = logistic.groupby(rank_groups)["mean_balanced_accuracy"].transform("max")
    logistic["balanced_accuracy_gap_to_best"] = (
        best_score - logistic["mean_balanced_accuracy"]
    )

    rows: list[dict[str, object]] = []
    for group_values, group in logistic.groupby(["state", "country", "horizon_days"], sort=True):
        state, country, horizon_days = group_values
        best = group.sort_values(
            ["rank", "mean_macro_f1", "representation"],
            ascending=[True, False, True],
        ).iloc[0]
        rows.append(
            {
                "state": state,
                "country": country,
                "horizon_days": horizon_days,
                "best_model": f"{best['representation']}/{best['model']}",
                "best_balanced_accuracy": float(best["mean_balanced_accuracy"]),
                **_curve_state_representation_values(group, "autoencoder"),
                **_curve_state_representation_values(group, "autoencoder_temporal"),
                **_curve_state_representation_values(group, "pca"),
                **_curve_state_representation_values(group, "pca_temporal"),
                **_curve_state_representation_values(group, "nelson_siegel"),
                **_curve_state_representation_values(group, "nelson_siegel_temporal"),
                **_curve_state_representation_values(group, "curve"),
                **_curve_state_representation_values(group, "policy"),
                "learned_representation_status": _curve_state_learned_status(group),
            }
        )
    return pd.DataFrame(rows).loc[:, columns]


def sequence_readiness_summary(transition_benchmark: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether temporal representation history improves state classification."""
    columns = [
        "state",
        "country",
        "horizon_days",
        "best_static_family",
        "best_static_balanced_accuracy",
        "best_temporal_family",
        "best_temporal_balanced_accuracy",
        "temporal_minus_static_best",
        "temporal_wins",
        "autoencoder_temporal_improvement",
        "pca_temporal_improvement",
        "nelson_siegel_temporal_improvement",
        "strongest_temporal_improvement_family",
        "strongest_temporal_improvement",
        "sequence_readiness_label",
    ]
    if transition_benchmark.empty:
        return pd.DataFrame(columns=columns)

    rows = [_sequence_readiness_row(row) for row in transition_benchmark.to_dict("records")]
    return pd.DataFrame(rows).loc[:, columns]


def curve_state_probe_importance_summary(config: ProjectConfig) -> pd.DataFrame:
    """Summarize standardized logistic coefficients for curve-state probes."""
    columns = [
        "state",
        "country",
        "horizon_days",
        "representation",
        "feature",
        "rows",
        "classes",
        "mean_coefficient",
        "mean_abs_coefficient",
        "max_abs_coefficient",
        "importance_rank",
    ]
    if not config.baseline_classification_coefficients_path.exists():
        return pd.DataFrame(columns=columns)

    coefficients = pd.read_parquet(config.baseline_classification_coefficients_path)
    if coefficients.empty:
        return pd.DataFrame(columns=columns)

    coefficients = coefficients.loc[
        coefficients["target"].str.startswith("curve_state_pc")
    ].copy()
    if coefficients.empty:
        return pd.DataFrame(columns=columns)

    coefficients["state"] = coefficients["target"].str.removeprefix("curve_state_")
    summary = (
        coefficients.groupby(
            ["state", "country", "horizon_days", "representation", "feature"],
            sort=True,
        )
        .agg(
            rows=("coefficient", "size"),
            classes=("class_label", "nunique"),
            mean_coefficient=("coefficient", "mean"),
            mean_abs_coefficient=("abs_coefficient", "mean"),
            max_abs_coefficient=("abs_coefficient", "max"),
        )
        .reset_index()
    )
    summary["importance_rank"] = summary.groupby(
        ["state", "country", "horizon_days", "representation"]
    )["mean_abs_coefficient"].rank(method="min", ascending=False)
    return (
        summary.loc[:, columns]
        .sort_values(
            [
                "state",
                "country",
                "horizon_days",
                "representation",
                "importance_rank",
                "feature",
            ]
        )
        .reset_index(drop=True)
    )


def ae_classical_factor_correlation_summary(config: ProjectConfig) -> pd.DataFrame:
    """Correlate autoencoder latent dimensions with PCA and Nelson-Siegel factors."""
    columns = [
        "country",
        "ae_feature",
        "classical_family",
        "classical_feature",
        "observations",
        "correlation",
        "abs_correlation",
        "match_rank",
    ]
    ae_frames = _read_autoencoder_embedding_frames(config)
    if not ae_frames:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for country, ae_frame in ae_frames.items():
        classical_frames = [
            ("pca", _read_country_factor_frame(config.pca_dir / f"{country.lower()}_scores.parquet")),
            (
                "nelson_siegel",
                _read_country_factor_frame(
                    config.nelson_siegel_dir / f"{country.lower()}_factors.parquet"
                ),
            ),
        ]
        for family, classical_frame in classical_frames:
            if classical_frame.empty:
                continue
            rows.extend(_ae_factor_correlation_rows(country, ae_frame, family, classical_frame))

    if not rows:
        return pd.DataFrame(columns=columns)

    summary = pd.DataFrame(rows)
    summary["match_rank"] = summary.groupby(["country", "ae_feature"])["abs_correlation"].rank(
        method="min",
        ascending=False,
    )
    return (
        summary.loc[:, columns]
        .sort_values(["country", "ae_feature", "match_rank", "classical_family"])
        .reset_index(drop=True)
    )


def benchmark_conclusion_summary(
    config: ProjectConfig,
    rank_table: pd.DataFrame,
    residual_rv_benchmark: pd.DataFrame,
    volatility_regime_benchmark: pd.DataFrame,
    curve_state_transition_benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize current strongest baselines by research question."""
    rows = [
        _curve_reconstruction_conclusion(config),
        _yield_forecasting_conclusion(rank_table),
        _residual_rv_conclusion(residual_rv_benchmark),
        _volatility_regime_conclusion(volatility_regime_benchmark),
        _curve_state_conclusion(curve_state_transition_benchmark),
    ]
    return pd.DataFrame(rows)


def _curve_reconstruction_conclusion(config: ProjectConfig) -> dict[str, object]:
    if not config.reconstruction_oos_summary_table_path.exists():
        return _conclusion_row(
            research_question="curve_reconstruction",
            current_best_baseline="not_evaluated",
            learned_representation_status="not_available_in_current_summary",
            evidence_table=config.reconstruction_oos_summary_table_path.name,
            conclusion="Run reconstruction to compare PCA, Nelson-Siegel, and autoencoder.",
        )

    reconstruction = pd.read_csv(config.reconstruction_oos_summary_table_path)
    if reconstruction.empty:
        return _conclusion_row(
            research_question="curve_reconstruction",
            current_best_baseline="not_evaluated",
            learned_representation_status="not_available_in_current_summary",
            evidence_table=config.reconstruction_oos_summary_table_path.name,
            conclusion="No reconstruction rows are available.",
        )

    best = (
        reconstruction.sort_values(["country", "rmse", "representation"])
        .groupby("country", sort=True)
        .first()
        .reset_index()
    )
    best_label = _country_winner_label(best, metric_column="rmse", lower_is_better=True)
    learned = reconstruction.loc[
        reconstruction["representation"].isin(["autoencoder", "masked_autoencoder"])
    ]
    learned_status = (
        "evaluated_but_not_best"
        if not learned.empty and not best["representation"].isin(learned["representation"]).any()
        else "mixed_or_best_in_some_markets"
        if not learned.empty
        else "not_evaluated"
    )
    return _conclusion_row(
        research_question="curve_reconstruction",
        current_best_baseline=best_label,
        learned_representation_status=learned_status,
        evidence_table=config.reconstruction_oos_summary_table_path.name,
        conclusion="PCA remains the main reconstruction hurdle for learned curve embeddings.",
    )


def _yield_forecasting_conclusion(rank_table: pd.DataFrame) -> dict[str, object]:
    target = rank_table.loc[rank_table["target"] == "yield_change"].copy()
    if target.empty:
        return _conclusion_row(
            research_question="outright_yield_forecasting",
            current_best_baseline="not_evaluated",
            learned_representation_status="not_evaluated",
            evidence_table="baseline_rank.csv",
            conclusion="No yield-change forecast rows are available.",
        )

    best = target.loc[target["rank"] == 1].sort_values(
        ["country", "horizon_days", "representation", "model"]
    )
    learned_best = best["representation"].eq("autoencoder").any()
    return _conclusion_row(
        research_question="outright_yield_forecasting",
        current_best_baseline=_winner_frequency_label(best),
        learned_representation_status=(
            "mixed_some_best_ranks" if learned_best else "evaluated_but_not_dominant"
        ),
        evidence_table="baseline_rank.csv",
        conclusion="Outright yield-change forecasting remains noisy and is not the central win condition.",
    )


def _residual_rv_conclusion(benchmark: pd.DataFrame) -> dict[str, object]:
    if benchmark.empty:
        return _conclusion_row(
            research_question="residual_relative_value",
            current_best_baseline="not_evaluated",
            learned_representation_status="not_evaluated",
            evidence_table="residual_relative_value_benchmark.csv",
            conclusion="No residual relative-value benchmark rows are available.",
        )

    spread_winner = _mode_label(benchmark["best_by_spread"])
    rank_ic_winner = _mode_label(benchmark["best_by_rank_ic"])
    learned_best = benchmark["best_by_spread"].astype(str).str.contains("autoencoder").any() or benchmark[
        "best_by_rank_ic"
    ].astype(str).str.contains("autoencoder").any()
    return _conclusion_row(
        research_question="residual_relative_value",
        current_best_baseline=f"spread={spread_winner}; rank_ic={rank_ic_winner}",
        learned_representation_status=(
            "mixed_some_best_ranks" if learned_best else "evaluated_but_not_best"
        ),
        evidence_table="residual_relative_value_benchmark.csv",
        conclusion="Explicit maturity residual features remain the strongest RV benchmark.",
    )


def _volatility_regime_conclusion(benchmark: pd.DataFrame) -> dict[str, object]:
    if benchmark.empty:
        return _conclusion_row(
            research_question="volatility_regime_classification",
            current_best_baseline="not_evaluated",
            learned_representation_status="not_evaluated",
            evidence_table="volatility_regime_benchmark.csv",
            conclusion="No volatility-regime benchmark rows are available.",
        )

    best_label = _mode_label(benchmark["best_model"])
    learned_best = benchmark["best_model"].astype(str).str.contains("autoencoder").any()
    return _conclusion_row(
        research_question="volatility_regime_classification",
        current_best_baseline=best_label,
        learned_representation_status=(
            "mixed_some_best_ranks" if learned_best else "evaluated_but_not_best"
        ),
        evidence_table="volatility_regime_benchmark.csv",
        conclusion="Realized curve-volatility and policy features remain the main hurdles.",
    )


def _curve_state_conclusion(transition_benchmark: pd.DataFrame) -> dict[str, object]:
    if transition_benchmark.empty:
        return _conclusion_row(
            research_question="curve_state_classification",
            current_best_baseline="not_evaluated",
            learned_representation_status="not_evaluated",
            evidence_table="curve_state_transition_benchmark.csv",
            conclusion="No curve-state classification rows are available.",
        )

    learned_status = transition_benchmark["learned_representation_status"].astype(str)
    learned_best = learned_status.eq("best").any()
    learned_competitive = learned_status.eq("competitive_with_best").any()
    return _conclusion_row(
        research_question="curve_state_classification",
        current_best_baseline=_mode_label(transition_benchmark["best_model"]),
        learned_representation_status=(
            "mixed_some_best_ranks"
            if learned_best
            else "competitive_in_some_markets"
            if learned_competitive
            else "evaluated_but_not_best"
        ),
        evidence_table="curve_state_transition_benchmark.csv",
        conclusion="Temporal PCA/NS baselines are strongest overall; AE remains useful in PC1/PC2 pockets, while PC3 is weak.",
    )


def _beats_hurdle(score: float | None, hurdle: float | None) -> bool | None:
    if score is None or hurdle is None:
        return None
    return score > hurdle


def residual_relative_value_rank_ic_summary(rank_table: pd.DataFrame) -> pd.DataFrame:
    """Rank residual-change baselines by cross-sectional rank IC."""
    columns = [
        "country",
        "horizon_days",
        "representation",
        "model",
        "rows",
        "mean_rank_ic",
        "rank_ic_dates",
        "mean_rmse",
        "mean_mae",
        "mean_directional_accuracy",
        "rank_ic_rank",
        "rank_ic_gap_to_best",
    ]
    if not {"mean_rank_ic", "rank_ic_dates"}.issubset(rank_table.columns):
        return pd.DataFrame(columns=columns)

    residual = rank_table.loc[
        (rank_table["target"] == "residual_change")
        & rank_table["mean_rank_ic"].notna()
        & (rank_table["rank_ic_dates"] > 0)
    ].copy()
    if residual.empty:
        return pd.DataFrame(columns=columns)

    rank_groups = ["country", "horizon_days"]
    residual["rank_ic_rank"] = residual.groupby(rank_groups)["mean_rank_ic"].rank(
        method="min",
        ascending=False,
        na_option="bottom",
    )
    best_rank_ic = residual.groupby(rank_groups)["mean_rank_ic"].transform("max")
    residual["rank_ic_gap_to_best"] = best_rank_ic - residual["mean_rank_ic"]

    available_columns = [column for column in columns if column in residual.columns]
    return (
        residual.sort_values(
            [*rank_groups, "rank_ic_rank", "mean_rmse", "representation", "model"],
            na_position="last",
        )
        .loc[:, available_columns]
        .reset_index(drop=True)
    )


def residual_relative_value_rank_ic_coverage(rank_table: pd.DataFrame) -> pd.DataFrame:
    """Audit which residual-change baselines have valid cross-sectional rank IC."""
    columns = [
        "country",
        "horizon_days",
        "representation",
        "model",
        "rows",
        "mean_rank_ic",
        "rank_ic_dates",
        "has_valid_rank_ic",
        "has_maturity_specific_features",
        "rank_ic_status",
    ]
    if not {"mean_rank_ic", "rank_ic_dates"}.issubset(rank_table.columns):
        return pd.DataFrame(columns=columns)

    residual = rank_table.loc[rank_table["target"] == "residual_change"].copy()
    if residual.empty:
        return pd.DataFrame(columns=columns)

    residual["has_valid_rank_ic"] = residual["mean_rank_ic"].notna() & (
        residual["rank_ic_dates"] > 0
    )
    residual["has_maturity_specific_features"] = residual["representation"].isin(
        ["carry_roll", "lagged", "residual_feature"]
    )
    residual["rank_ic_status"] = residual.apply(_rank_ic_status, axis=1)
    return (
        residual.loc[:, columns]
        .sort_values(["country", "horizon_days", "representation", "model"])
        .reset_index(drop=True)
    )


def _rank_ic_status(row: pd.Series) -> str:
    if bool(row["has_valid_rank_ic"]):
        return "valid"
    if not bool(row["has_maturity_specific_features"]):
        return "undefined_for_date_level_features"
    return "undefined"


def residual_relative_value_spread_summary(config: ProjectConfig) -> pd.DataFrame:
    """Rank residual relative-value baselines by top-minus-bottom spread score."""
    columns = [
        "country",
        "horizon_days",
        "representation",
        "model",
        "dates",
        "mean_spread_score",
        "spread_t_stat",
        "hit_rate",
        "mean_top_realized",
        "mean_bottom_realized",
        "mean_leg_size",
        "spread_rank",
        "spread_gap_to_best",
    ]
    if not config.baseline_residual_rv_spread_path.exists():
        return pd.DataFrame(columns=columns)

    spreads = pd.read_parquet(config.baseline_residual_rv_spread_path)
    if spreads.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        spreads.groupby(["country", "horizon_days", "representation", "model"], sort=True)
        .agg(
            dates=("dates", "sum"),
            mean_spread_score=("mean_spread_score", "mean"),
            spread_t_stat=("spread_t_stat", "mean"),
            hit_rate=("hit_rate", "mean"),
            mean_top_realized=("mean_top_realized", "mean"),
            mean_bottom_realized=("mean_bottom_realized", "mean"),
            mean_leg_size=("mean_leg_size", "mean"),
        )
        .reset_index()
    )
    rank_groups = ["country", "horizon_days"]
    summary["spread_rank"] = summary.groupby(rank_groups)["mean_spread_score"].rank(
        method="min",
        ascending=False,
    )
    best_spread = summary.groupby(rank_groups)["mean_spread_score"].transform("max")
    summary["spread_gap_to_best"] = best_spread - summary["mean_spread_score"]
    return (
        summary.loc[:, columns]
        .sort_values([*rank_groups, "spread_rank", "representation", "model"])
        .reset_index(drop=True)
    )


def residual_relative_value_benchmark_summary(
    spread_summary: pd.DataFrame,
    rank_ic_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build one compact interpretation table for residual relative-value benchmarks."""
    columns = [
        "country",
        "horizon_days",
        "best_by_spread",
        "best_spread_score",
        "best_spread_t_stat",
        "best_hit_rate",
        "best_by_rank_ic",
        "best_rank_ic",
        "residual_feature_spread_rank",
        "residual_feature_rank_ic_rank",
        "pca_maturity_spread_rank",
        "pca_maturity_rank_ic_rank",
        "autoencoder_maturity_spread_rank",
        "autoencoder_maturity_rank_ic_rank",
        "nelson_siegel_maturity_spread_rank",
        "nelson_siegel_maturity_rank_ic_rank",
        "curve_maturity_spread_rank",
        "curve_maturity_rank_ic_rank",
    ]
    if spread_summary.empty and rank_ic_summary.empty:
        return pd.DataFrame(columns=columns)

    keys = ["country", "horizon_days"]
    key_frame = pd.concat(
        [
            spread_summary.loc[:, keys] if not spread_summary.empty else pd.DataFrame(columns=keys),
            rank_ic_summary.loc[:, keys] if not rank_ic_summary.empty else pd.DataFrame(columns=keys),
        ],
        ignore_index=True,
    ).drop_duplicates()

    rows: list[dict[str, object]] = []
    for key_values in key_frame.sort_values(keys).itertuples(index=False):
        country = str(key_values.country)
        horizon_days = int(str(key_values.horizon_days))
        spread_group = _group_for_key(spread_summary, country, horizon_days)
        rank_ic_group = _group_for_key(rank_ic_summary, country, horizon_days)
        rows.append(
            {
                "country": country,
                "horizon_days": horizon_days,
                **_best_spread_values(spread_group),
                **_best_rank_ic_values(rank_ic_group),
                **_representation_rank_values(spread_group, rank_ic_group, "residual_feature"),
                **_representation_rank_values(spread_group, rank_ic_group, "pca_maturity"),
                **_representation_rank_values(
                    spread_group,
                    rank_ic_group,
                    "autoencoder_maturity",
                ),
                **_representation_rank_values(
                    spread_group,
                    rank_ic_group,
                    "nelson_siegel_maturity",
                ),
                **_representation_rank_values(spread_group, rank_ic_group, "curve_maturity"),
            }
        )
    return pd.DataFrame(rows).loc[:, columns]


def residual_relative_value_overview(
    benchmark: pd.DataFrame,
    mean_reversion: pd.DataFrame,
) -> pd.DataFrame:
    """Combine residual RV ranking and direct mean-reversion evidence."""
    columns = [
        "country",
        "horizon_days",
        "best_by_spread",
        "best_spread_score",
        "best_spread_t_stat",
        "best_hit_rate",
        "best_by_rank_ic",
        "best_rank_ic",
        "residual_feature_spread_rank",
        "residual_feature_rank_ic_rank",
        "mean_reversion_hit_rate",
        "mean_reversion_score",
        "mean_reversion_rank_ic",
        "mean_reversion_dates",
        "evidence_label",
    ]
    if benchmark.empty:
        return pd.DataFrame(columns=columns)

    overview = benchmark.copy()
    if not mean_reversion.empty:
        overview = overview.merge(
            _mean_reversion_overview(mean_reversion),
            on=["country", "horizon_days"],
            how="left",
        )

    for column in [
        "mean_reversion_hit_rate",
        "mean_reversion_score",
        "mean_reversion_rank_ic",
        "mean_reversion_dates",
    ]:
        if column not in overview.columns:
            overview[column] = pd.NA
    overview["evidence_label"] = overview.apply(_residual_rv_evidence_label, axis=1)
    return (
        overview.loc[:, columns]
        .sort_values(["country", "horizon_days"])
        .reset_index(drop=True)
    )


def _mean_reversion_overview(mean_reversion: pd.DataFrame) -> pd.DataFrame:
    focused = mean_reversion.loc[
        (mean_reversion["sample"] == "abs_z_ge_1")
        & (mean_reversion["signal"] == "residual_z_252")
    ].copy()
    if focused.empty:
        return pd.DataFrame(
            columns=[
                "country",
                "horizon_days",
                "mean_reversion_hit_rate",
                "mean_reversion_score",
                "mean_reversion_rank_ic",
                "mean_reversion_dates",
            ]
        )

    rows: list[dict[str, object]] = []
    for group_values, group in focused.groupby(["country", "horizon_days"], sort=True):
        country, horizon_days = group_values
        rows.append(
            {
                "country": country,
                "horizon_days": horizon_days,
                "mean_reversion_hit_rate": _weighted_mean(
                    group["convergence_hit_rate"],
                    group["rows"],
                ),
                "mean_reversion_score": _weighted_mean(
                    group["mean_convergence_score"],
                    group["rows"],
                ),
                "mean_reversion_rank_ic": _weighted_mean(
                    group["mean_rank_ic"],
                    group["rank_ic_dates"],
                ),
                "mean_reversion_dates": int(group["dates"].max()),
            }
        )
    return pd.DataFrame(rows)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values.loc[valid].to_numpy(dtype=float), weights=weights.loc[valid]))


def _residual_rv_evidence_label(row: pd.Series) -> str:
    hit_rate = row.get("mean_reversion_hit_rate")
    spread_score = row.get("best_spread_score")
    rank_ic = row.get("best_rank_ic")
    horizon = int(row["horizon_days"])
    has_positive_ranking = (
        pd.notna(spread_score)
        and pd.notna(rank_ic)
        and float(spread_score) > 0.0
        and float(rank_ic) > 0.0
    )
    if pd.notna(hit_rate) and float(hit_rate) >= 0.57 and has_positive_ranking:
        return "moderate_positive_20d" if horizon >= 20 else "moderate_positive"
    if pd.notna(hit_rate) and float(hit_rate) >= 0.52 and has_positive_ranking:
        return "weak_positive"
    if has_positive_ranking:
        return "ranking_positive"
    return "mixed"


def market_regime_rv_summary_table(regime_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize high/low market-regime differences in residual RV diagnostics."""
    columns = [
        "indicator",
        "country",
        "horizon_days",
        "best_regime",
        "best_hit_rate",
        "worst_regime",
        "worst_hit_rate",
        "high_minus_low_hit_rate",
        "high_minus_low_rank_ic",
        "interpretation",
    ]
    if regime_summary.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for group_values, group in regime_summary.groupby(
        ["indicator", "country", "horizon_days"],
        sort=True,
    ):
        indicator, country, horizon_days = group_values
        best = group.sort_values(
            ["convergence_hit_rate", "mean_rank_ic", "market_vol_regime"],
            ascending=[False, False, True],
        ).iloc[0]
        worst = group.sort_values(
            ["convergence_hit_rate", "mean_rank_ic", "market_vol_regime"],
            ascending=[True, True, True],
        ).iloc[0]
        high = _regime_row(group, "high")
        low = _regime_row(group, "low")
        high_minus_low_hit = _difference(high, low, "convergence_hit_rate")
        high_minus_low_rank_ic = _difference(high, low, "mean_rank_ic")
        rows.append(
            {
                "indicator": indicator,
                "country": country,
                "horizon_days": horizon_days,
                "best_regime": best["market_vol_regime"],
                "best_hit_rate": best["convergence_hit_rate"],
                "worst_regime": worst["market_vol_regime"],
                "worst_hit_rate": worst["convergence_hit_rate"],
                "high_minus_low_hit_rate": high_minus_low_hit,
                "high_minus_low_rank_ic": high_minus_low_rank_ic,
                "interpretation": _market_regime_interpretation(high_minus_low_hit),
            }
        )
    return pd.DataFrame(rows).loc[:, columns]


def _regime_row(group: pd.DataFrame, regime: str) -> pd.Series | None:
    rows = group.loc[group["market_vol_regime"] == regime]
    if rows.empty:
        return None
    return rows.iloc[0]


def _difference(left: pd.Series | None, right: pd.Series | None, column: str) -> float | None:
    if left is None or right is None:
        return None
    return float(left[column] - right[column])


def _market_regime_interpretation(high_minus_low_hit_rate: float | None) -> str:
    if high_minus_low_hit_rate is None:
        return "insufficient_regime_coverage"
    if high_minus_low_hit_rate >= 0.03:
        return "stronger_in_high_vol"
    if high_minus_low_hit_rate <= -0.03:
        return "stronger_in_low_vol"
    return "similar_across_regimes"


def _group_for_key(data: pd.DataFrame, country: str, horizon_days: int) -> pd.DataFrame:
    if data.empty:
        return data
    return data.loc[(data["country"] == country) & (data["horizon_days"] == horizon_days)]


def _best_spread_values(group: pd.DataFrame) -> dict[str, object]:
    if group.empty:
        return {
            "best_by_spread": None,
            "best_spread_score": None,
            "best_spread_t_stat": None,
            "best_hit_rate": None,
        }
    best = group.sort_values(["spread_rank", "representation", "model"]).iloc[0]
    return {
        "best_by_spread": f"{best['representation']}/{best['model']}",
        "best_spread_score": best["mean_spread_score"],
        "best_spread_t_stat": best["spread_t_stat"],
        "best_hit_rate": best["hit_rate"],
    }


def _best_rank_ic_values(group: pd.DataFrame) -> dict[str, object]:
    if group.empty:
        return {"best_by_rank_ic": None, "best_rank_ic": None}
    best = group.sort_values(["rank_ic_rank", "representation", "model"]).iloc[0]
    return {
        "best_by_rank_ic": f"{best['representation']}/{best['model']}",
        "best_rank_ic": best["mean_rank_ic"],
    }


def _representation_rank_values(
    spread_group: pd.DataFrame,
    rank_ic_group: pd.DataFrame,
    representation: str,
) -> dict[str, object]:
    prefix = representation
    return {
        f"{prefix}_spread_rank": _rank_for_representation(
            spread_group,
            representation,
            "spread_rank",
        ),
        f"{prefix}_rank_ic_rank": _rank_for_representation(
            rank_ic_group,
            representation,
            "rank_ic_rank",
        ),
    }


def _rank_for_representation(
    group: pd.DataFrame,
    representation: str,
    rank_column: str,
) -> float | None:
    rows = group.loc[group["representation"] == representation]
    if rows.empty:
        return None
    return float(rows.sort_values([rank_column, "model"]).iloc[0][rank_column])


def _curve_state_representation_values(
    group: pd.DataFrame,
    representation: str,
) -> dict[str, object]:
    prefix = representation
    rows = group.loc[group["representation"] == representation]
    if rows.empty:
        return {
            f"{prefix}_rank": None,
            f"{prefix}_balanced_accuracy": None,
            f"{prefix}_gap_to_best": None,
        }
    row = rows.sort_values(["rank", "mean_macro_f1", "model"], ascending=[True, False, True]).iloc[
        0
    ]
    return {
        f"{prefix}_rank": float(row["rank"]),
        f"{prefix}_balanced_accuracy": float(row["mean_balanced_accuracy"]),
        f"{prefix}_gap_to_best": float(row["balanced_accuracy_gap_to_best"]),
    }


def _curve_state_learned_status(group: pd.DataFrame) -> str:
    rows = group.loc[group["representation"].isin(["autoencoder", "autoencoder_temporal"])]
    if rows.empty:
        return "not_evaluated"
    row = rows.sort_values(["rank", "mean_macro_f1", "model"], ascending=[True, False, True]).iloc[
        0
    ]
    rank = float(row["rank"])
    gap = float(row["balanced_accuracy_gap_to_best"])
    if rank == 1.0:
        return "best"
    if gap <= 0.05:
        return "competitive_with_best"
    return "behind_best"


def _sequence_readiness_row(row: dict[str, object]) -> dict[str, object]:
    static_scores = {
        "autoencoder": _as_float(row.get("autoencoder_balanced_accuracy")),
        "pca": _as_float(row.get("pca_balanced_accuracy")),
        "nelson_siegel": _as_float(row.get("nelson_siegel_balanced_accuracy")),
    }
    temporal_scores = {
        "autoencoder": _as_float(row.get("autoencoder_temporal_balanced_accuracy")),
        "pca": _as_float(row.get("pca_temporal_balanced_accuracy")),
        "nelson_siegel": _as_float(row.get("nelson_siegel_temporal_balanced_accuracy")),
    }
    improvements = {
        family: _difference_or_nan(temporal_scores[family], static_scores[family])
        for family in static_scores
    }
    best_static_family, best_static_score = _best_score(static_scores)
    best_temporal_family, best_temporal_score = _best_score(temporal_scores)
    strongest_family, strongest_improvement = _best_score(improvements)
    temporal_minus_static = _difference_or_nan(best_temporal_score, best_static_score)
    return {
        "state": row["state"],
        "country": row["country"],
        "horizon_days": row["horizon_days"],
        "best_static_family": best_static_family,
        "best_static_balanced_accuracy": best_static_score,
        "best_temporal_family": best_temporal_family,
        "best_temporal_balanced_accuracy": best_temporal_score,
        "temporal_minus_static_best": temporal_minus_static,
        "temporal_wins": _positive_or_false(temporal_minus_static),
        "autoencoder_temporal_improvement": improvements["autoencoder"],
        "pca_temporal_improvement": improvements["pca"],
        "nelson_siegel_temporal_improvement": improvements["nelson_siegel"],
        "strongest_temporal_improvement_family": strongest_family,
        "strongest_temporal_improvement": strongest_improvement,
        "sequence_readiness_label": _sequence_readiness_label(
            temporal_minus_static,
            strongest_improvement,
        ),
    }


def _as_float(value: object) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    if isinstance(value, int | float | str | np.integer | np.floating):
        return float(value)
    return float("nan")


def _difference_or_nan(left: float, right: float) -> float:
    if pd.isna(left) or pd.isna(right):
        return float("nan")
    return left - right


def _best_score(scores: dict[str, float]) -> tuple[str | None, float]:
    valid = {key: value for key, value in scores.items() if not pd.isna(value)}
    if not valid:
        return None, float("nan")
    best_key = max(valid, key=valid.__getitem__)
    return best_key, valid[best_key]


def _positive_or_false(value: float) -> bool:
    return bool(not pd.isna(value) and value > 0.0)


def _sequence_readiness_label(
    temporal_minus_static: float,
    strongest_improvement: float,
) -> str:
    if pd.isna(temporal_minus_static):
        return "insufficient_data"
    if temporal_minus_static >= 0.05 or strongest_improvement >= 0.05:
        return "strong_temporal_signal"
    if temporal_minus_static > 0.0 or strongest_improvement > 0.0:
        return "modest_temporal_signal"
    return "no_temporal_gain"


def _read_autoencoder_embedding_frames(config: ProjectConfig) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    if not config.autoencoder_dir.exists():
        return frames

    for path in sorted(config.autoencoder_dir.glob("*_embeddings.parquet")):
        country = path.name.removesuffix("_embeddings.parquet").upper()
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame["date"] = pd.to_datetime(frame["date"])
        frames[country] = frame
    return frames


def _read_country_factor_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _ae_factor_correlation_rows(
    country: str,
    ae_frame: pd.DataFrame,
    classical_family: str,
    classical_frame: pd.DataFrame,
) -> list[dict[str, object]]:
    ae_features = [column for column in ae_frame.columns if column.startswith("AE")]
    classical_features = [
        column
        for column in classical_frame.columns
        if column not in {"date", "country", "split", "tau", "rmse"}
    ]
    if not ae_features or not classical_features:
        return []

    merged = ae_frame.loc[:, ["date", *ae_features]].merge(
        classical_frame.loc[:, ["date", *classical_features]],
        on="date",
        how="inner",
    )
    if merged.empty:
        return []

    rows: list[dict[str, object]] = []
    for ae_feature in ae_features:
        for classical_feature in classical_features:
            valid = merged[[ae_feature, classical_feature]].dropna()
            if len(valid) < 2:
                continue
            correlation = valid[ae_feature].corr(valid[classical_feature])
            rows.append(
                {
                    "country": country,
                    "ae_feature": ae_feature,
                    "classical_family": classical_family,
                    "classical_feature": classical_feature,
                    "observations": len(valid),
                    "correlation": float(correlation),
                    "abs_correlation": float(abs(correlation)),
                    "match_rank": None,
                }
            )
    return rows


def _conclusion_row(
    research_question: str,
    current_best_baseline: str,
    learned_representation_status: str,
    evidence_table: str,
    conclusion: str,
) -> dict[str, object]:
    return {
        "research_question": research_question,
        "current_best_baseline": current_best_baseline,
        "learned_representation_status": learned_representation_status,
        "evidence_table": evidence_table,
        "conclusion": conclusion,
    }


def _country_winner_label(
    winners: pd.DataFrame,
    metric_column: str,
    lower_is_better: bool,
) -> str:
    if winners.empty:
        return "not_evaluated"
    metric_label = "rmse" if lower_is_better else metric_column
    labels = [
        f"{row.country}:{row.representation}/{int(row.n_components)} {metric_label}={getattr(row, metric_column):.4f}"
        for row in winners.itertuples(index=False)
    ]
    return "; ".join(labels)


def _winner_frequency_label(winners: pd.DataFrame) -> str:
    if winners.empty:
        return "not_evaluated"
    labels = winners["representation"].astype(str) + "/" + winners["model"].astype(str)
    counts = labels.value_counts()
    return "; ".join(f"{label} ({count})" for label, count in counts.items())


def _mode_label(values: pd.Series) -> str:
    valid = values.dropna().astype(str)
    if valid.empty:
        return "not_evaluated"
    return str(valid.value_counts().index[0])


def _naive_residual_rows(residual: pd.DataFrame, rank_groups: list[str]) -> pd.DataFrame:
    naive = residual.loc[residual["model"] == "train_mean"]
    if naive.empty:
        return naive

    aggregations = {
        "rows": ("rows", "sum"),
        "mean_rmse": ("mean_rmse", "mean"),
        "mean_mae": ("mean_mae", "mean"),
        "mean_directional_accuracy": ("mean_directional_accuracy", "mean"),
    }
    if "mean_rank_ic" in naive.columns:
        aggregations["mean_rank_ic"] = ("mean_rank_ic", "mean")
    if "rank_ic_dates" in naive.columns:
        aggregations["rank_ic_dates"] = ("rank_ic_dates", "sum")

    rows = naive.groupby(rank_groups, sort=True).agg(**aggregations).reset_index()
    rows["representation"] = "naive"
    rows["model"] = "train_mean"
    return rows


def rank_baselines(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank baseline representations within each target/country/horizon task."""
    summary = (
        metrics.groupby([*RANK_GROUP_COLUMNS, "representation", "model"], sort=True)
        .agg(**_rank_aggregations(metrics))
        .reset_index()
    )
    summary["rank"] = summary.groupby(RANK_GROUP_COLUMNS)["mean_rmse"].rank(
        method="min",
        ascending=True,
    )
    best_rmse = summary.groupby(RANK_GROUP_COLUMNS)["mean_rmse"].transform("min")
    summary["rmse_gap_to_best"] = summary["mean_rmse"] - best_rmse
    summary["pct_gap_to_best"] = summary["rmse_gap_to_best"] / best_rmse
    return summary.sort_values([*RANK_GROUP_COLUMNS, "rank", "mean_mae"]).reset_index(drop=True)


def baseline_winners(rank_table: pd.DataFrame) -> pd.DataFrame:
    """Create a compact winner table with PCA and lagged gaps to best."""
    rows: list[dict[str, object]] = []
    for group_values, group in rank_table.groupby(RANK_GROUP_COLUMNS, sort=True):
        keys = dict(zip(RANK_GROUP_COLUMNS, group_values, strict=True))
        best = group.sort_values(["rank", "mean_mae", "representation", "model"]).iloc[0]
        pca = _best_representation_row(group, "pca")
        lagged = _best_representation_row(group, "lagged")
        rows.append(
            {
                **keys,
                "best_representation": best["representation"],
                "best_model": best["model"],
                "best_rmse": best["mean_rmse"],
                "best_mae": best["mean_mae"],
                "pca_rank": _rank_value(pca),
                "pca_rmse_gap_to_best": _gap_value(pca),
                "pca_pct_gap_to_best": _pct_gap_value(pca),
                "lagged_rank": _rank_value(lagged),
                "lagged_rmse_gap_to_best": _gap_value(lagged),
                "lagged_pct_gap_to_best": _pct_gap_value(lagged),
            }
        )
    return pd.DataFrame(rows)


def _rank_aggregations(metrics: pd.DataFrame) -> dict[str, tuple[str, str]]:
    aggregations = {
        "rows": ("rmse", "size"),
        "mean_rmse": ("rmse", "mean"),
        "mean_mae": ("mae", "mean"),
        "mean_directional_accuracy": ("directional_accuracy", "mean"),
        "mean_test_dates": ("test_dates", "mean"),
    }
    if "mean_rank_ic" in metrics.columns:
        aggregations["mean_rank_ic"] = ("mean_rank_ic", "mean")
    if "rank_ic_dates" in metrics.columns:
        aggregations["rank_ic_dates"] = ("rank_ic_dates", "sum")
    return aggregations


def _best_representation_row(group: pd.DataFrame, representation: str) -> pd.Series | None:
    rows = group.loc[group["representation"] == representation]
    if rows.empty:
        return None
    return rows.sort_values(["rank", "mean_mae", "model"]).iloc[0]


def _rank_value(row: pd.Series | None) -> float | None:
    return None if row is None else float(row["rank"])


def _gap_value(row: pd.Series | None) -> float | None:
    return None if row is None else float(row["rmse_gap_to_best"])


def _pct_gap_value(row: pd.Series | None) -> float | None:
    return None if row is None else float(row["pct_gap_to_best"])


def _evaluate_with_target_window(
    config: ProjectConfig,
    non_overlapping_targets: bool,
) -> pd.DataFrame:
    evaluation = config.evaluation.model_copy(
        update={"non_overlapping_targets": non_overlapping_targets}
    )
    evaluation_config = config.model_copy(update={"evaluation": evaluation})
    return evaluate_baseline_frames(evaluation_config).metrics


def _rank_for_target_window(metrics: pd.DataFrame, target_window: str) -> pd.DataFrame:
    rank_table = rank_baselines(metrics)
    columns = [
        *RANK_GROUP_COLUMNS,
        "representation",
        "model",
        "mean_rmse",
        "mean_mae",
        "mean_directional_accuracy",
        "rank",
        "rmse_gap_to_best",
        "pct_gap_to_best",
    ]
    if "mean_rank_ic" in rank_table.columns:
        columns.append("mean_rank_ic")
    if "rank_ic_dates" in rank_table.columns:
        columns.append("rank_ic_dates")

    renamed = {
        column: f"{target_window}_{column}"
        for column in columns
        if column not in [*RANK_GROUP_COLUMNS, "representation", "model"]
    }
    return rank_table.loc[:, columns].rename(columns=renamed)
