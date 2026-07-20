from __future__ import annotations

from pathlib import Path

import pandas as pd

from yieldrep.config import ProjectConfig
from yieldrep.evaluation.datasets import (
    make_supervised_feature_dataset,
)
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

    winners = baseline_winners(rank_table)
    winners.to_csv(config.baseline_winners_table_path, index=False)

    volatility_regime = volatility_regime_summary(config)
    volatility_regime.to_csv(config.volatility_regime_table_path, index=False)

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

    return [
        config.baseline_summary_table_path,
        config.baseline_rank_table_path,
        config.residual_relative_value_rank_ic_table_path,
        config.residual_relative_value_rank_ic_coverage_table_path,
        config.residual_relative_value_spread_table_path,
        config.residual_relative_value_benchmark_table_path,
        config.baseline_winners_table_path,
        config.volatility_regime_table_path,
        config.baseline_by_maturity_bucket_table_path,
        config.residual_relative_value_table_path,
        config.baseline_by_maturity_point_top_table_path,
    ]


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
                    "nelson_siegel_maturity",
                ),
                **_representation_rank_values(spread_group, rank_ic_group, "curve_maturity"),
            }
        )
    return pd.DataFrame(rows).loc[:, columns]


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
