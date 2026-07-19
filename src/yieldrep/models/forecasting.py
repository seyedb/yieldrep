from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from yieldrep.config import ProjectConfig
from yieldrep.evaluation.metrics import directional_accuracy, mae, rmse
from yieldrep.evaluation.splits import evaluation_splits

GROUP_COLUMNS = ["country", "horizon_days", "split_method", "window_id"]
METRIC_GROUP_COLUMNS = ["target", "representation", "model"]
RANK_GROUP_COLUMNS = ["target", "country", "horizon_days"]


@dataclass(frozen=True)
class FeatureSet:
    representation: str
    columns: list[str]


@dataclass(frozen=True)
class TargetSpec:
    target: str
    path: Path
    target_column: str


@dataclass(frozen=True)
class TargetFrameSpec:
    target: str
    data: pd.DataFrame
    target_column: str


@dataclass(frozen=True)
class PredictionResult:
    model: str
    y_pred: NDArray[np.float64]
    coefficients: dict[str, float]


@dataclass(frozen=True)
class SupervisedForecastFrames:
    metrics: pd.DataFrame
    by_maturity_bucket: pd.DataFrame
    coefficients: pd.DataFrame


def evaluate_supervised_forecasts(config: ProjectConfig) -> list[Path]:
    """Evaluate classical forecasting benchmarks from the canonical supervised table."""
    target_specs = _target_specs(config)
    if not target_specs:
        return []

    frames = supervised_forecast_frames(
        target_specs=target_specs,
        feature_sets=_feature_sets(config),
        config=config,
    )

    config.evaluation_dir.mkdir(parents=True, exist_ok=True)
    config.tables_dir.mkdir(parents=True, exist_ok=True)

    frames.metrics.to_parquet(config.supervised_forecast_metrics_path, index=False)
    frames.by_maturity_bucket.to_parquet(
        config.supervised_forecast_by_maturity_bucket_path,
        index=False,
    )
    frames.coefficients.to_parquet(config.supervised_forecast_coefficients_path, index=False)

    summary = summarize_supervised_forecasts(frames.metrics)
    summary.to_csv(config.supervised_forecast_summary_table_path, index=False)
    rank = rank_supervised_forecasts(frames.metrics)
    rank.to_csv(config.supervised_forecast_rank_table_path, index=False)
    bucket_summary = summarize_supervised_forecasts_by_bucket(frames.by_maturity_bucket)
    bucket_summary.to_csv(config.supervised_forecast_by_maturity_bucket_table_path, index=False)
    coefficient_summary = summarize_coefficients(frames.coefficients)
    coefficient_summary.to_csv(config.supervised_forecast_coefficients_table_path, index=False)

    return [
        config.supervised_forecast_metrics_path,
        config.supervised_forecast_by_maturity_bucket_path,
        config.supervised_forecast_coefficients_path,
        config.supervised_forecast_summary_table_path,
        config.supervised_forecast_rank_table_path,
        config.supervised_forecast_by_maturity_bucket_table_path,
        config.supervised_forecast_coefficients_table_path,
    ]


def supervised_forecast_frames(
    target_specs: list[TargetSpec],
    feature_sets: list[FeatureSet],
    config: ProjectConfig,
) -> SupervisedForecastFrames:
    """Compute forecast metrics, maturity-bucket metrics, and coefficients from paths."""
    target_frames = [
        TargetFrameSpec(
            target=target_spec.target,
            data=pd.read_parquet(target_spec.path),
            target_column=target_spec.target_column,
        )
        for target_spec in target_specs
    ]
    return supervised_forecast_frames_from_data(target_frames, feature_sets, config)


def supervised_forecast_frames_from_data(
    target_specs: list[TargetFrameSpec],
    feature_sets: list[FeatureSet],
    config: ProjectConfig,
) -> SupervisedForecastFrames:
    """Compute forecast metrics, maturity-bucket metrics, and coefficients from frames."""
    metric_rows: list[dict[str, object]] = []
    bucket_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []

    for target_spec in target_specs:
        data = target_spec.data
        for feature_set in feature_sets:
            columns = [column for column in feature_set.columns if column in data.columns]
            if not columns:
                continue

            required = [
                "date",
                "maturity_years",
                *GROUP_COLUMNS,
                "split",
                target_spec.target_column,
                *columns,
            ]
            sample = data.dropna(subset=required).loc[:, required]
            for group_values, group in sample.groupby(GROUP_COLUMNS, sort=True):
                train = group.loc[group["split"] == "train"]
                test = group.loc[group["split"] == "test"]
                if train.empty or test.empty:
                    continue

                x_train = train[columns].to_numpy(dtype=float)
                y_train = train[target_spec.target_column].to_numpy(dtype=float)
                x_test = test[columns].to_numpy(dtype=float)
                y_test = test[target_spec.target_column].to_numpy(dtype=float)

                predictions = _predictions(config, columns, x_train, y_train, x_test)
                metric_rows.extend(
                    _metric_rows(
                        target=target_spec.target,
                        group_values=group_values,
                        feature_set=feature_set,
                        feature_count=len(columns),
                        y_true=y_test,
                        predictions=predictions,
                        train_rows=len(train),
                        test_rows=len(test),
                        train_dates=train["date"].nunique(),
                        test_dates=test["date"].nunique(),
                    )
                )
                bucket_rows.extend(
                    _bucket_metric_rows(
                        target=target_spec.target,
                        group_values=group_values,
                        feature_set=feature_set,
                        feature_count=len(columns),
                        test=test,
                        y_true=y_test,
                        predictions=predictions,
                        train_rows=len(train),
                    )
                )
                coefficient_rows.extend(
                    _coefficient_rows(
                        target=target_spec.target,
                        group_values=group_values,
                        feature_set=feature_set,
                        predictions=predictions,
                    )
                )

    metrics = _add_improvement_vs_train_mean(pd.DataFrame(metric_rows), ["target", *GROUP_COLUMNS])
    by_maturity_bucket = _add_improvement_vs_train_mean(
        pd.DataFrame(bucket_rows),
        ["target", *GROUP_COLUMNS, "maturity_bucket"],
    )
    coefficients = pd.DataFrame(coefficient_rows)
    return SupervisedForecastFrames(
        metrics=metrics,
        by_maturity_bucket=by_maturity_bucket,
        coefficients=coefficients,
    )


def supervised_forecast_frames_from_unsplit_data(
    target_specs: list[TargetFrameSpec],
    feature_sets: list[FeatureSet],
    config: ProjectConfig,
) -> SupervisedForecastFrames:
    """Compute supervised forecast frames while assigning splits lazily by target group."""
    metric_rows: list[dict[str, object]] = []
    bucket_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []

    for target_spec in target_specs:
        data = target_spec.data
        for feature_set in feature_sets:
            columns = [column for column in feature_set.columns if column in data.columns]
            if not columns:
                continue

            required = [
                "date",
                "country",
                "horizon_days",
                "maturity_years",
                target_spec.target_column,
                *columns,
            ]
            sample = data.dropna(subset=required).loc[:, required]
            for group_values, group in sample.groupby(["country", "horizon_days"], sort=True):
                country, horizon_days = group_values
                for split in evaluation_splits(
                    group,
                    method=config.evaluation.method,
                    test_fraction=config.evaluation.test_fraction,
                    min_train_dates=config.evaluation.min_train_dates,
                    test_window_dates=config.evaluation.test_window_dates,
                    step_dates=config.evaluation.step_dates,
                    max_windows=config.evaluation.walk_forward_max_windows,
                    horizon_days=int(horizon_days),
                    non_overlapping_targets=config.evaluation.non_overlapping_targets,
                ):
                    train = split.train
                    test = split.test
                    if train.empty or test.empty:
                        continue

                    x_train = train[columns].to_numpy(dtype=float)
                    y_train = train[target_spec.target_column].to_numpy(dtype=float)
                    x_test = test[columns].to_numpy(dtype=float)
                    y_test = test[target_spec.target_column].to_numpy(dtype=float)
                    split_group_values = (country, horizon_days, split.method, split.window_id)
                    predictions = _predictions(config, columns, x_train, y_train, x_test)
                    metric_rows.extend(
                        _metric_rows(
                            target=target_spec.target,
                            group_values=split_group_values,
                            feature_set=feature_set,
                            feature_count=len(columns),
                            y_true=y_test,
                            predictions=predictions,
                            train_rows=len(train),
                            test_rows=len(test),
                            train_dates=train["date"].nunique(),
                            test_dates=test["date"].nunique(),
                        )
                    )
                    bucket_rows.extend(
                        _bucket_metric_rows(
                            target=target_spec.target,
                            group_values=split_group_values,
                            feature_set=feature_set,
                            feature_count=len(columns),
                            test=test,
                            y_true=y_test,
                            predictions=predictions,
                            train_rows=len(train),
                        )
                    )
                    coefficient_rows.extend(
                        _coefficient_rows(
                            target=target_spec.target,
                            group_values=split_group_values,
                            feature_set=feature_set,
                            predictions=predictions,
                        )
                    )

    metrics = _add_improvement_vs_train_mean(pd.DataFrame(metric_rows), ["target", *GROUP_COLUMNS])
    by_maturity_bucket = _add_improvement_vs_train_mean(
        pd.DataFrame(bucket_rows),
        ["target", *GROUP_COLUMNS, "maturity_bucket"],
    )
    coefficients = pd.DataFrame(coefficient_rows)
    return SupervisedForecastFrames(
        metrics=metrics,
        by_maturity_bucket=by_maturity_bucket,
        coefficients=coefficients,
    )


def summarize_supervised_forecasts(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate supervised forecast metrics by representation and model."""
    return _summarize_metrics(metrics, METRIC_GROUP_COLUMNS)


def summarize_supervised_forecasts_by_bucket(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate supervised forecast metrics by maturity bucket."""
    return _summarize_metrics(metrics, [*METRIC_GROUP_COLUMNS, "maturity_bucket"])


def rank_supervised_forecasts(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank supervised feature sets within each country and horizon."""
    summary = _summarize_metrics(
        metrics,
        [*RANK_GROUP_COLUMNS, "representation", "model"],
        include_mean_test_dates=True,
    )
    summary["rank"] = summary.groupby(RANK_GROUP_COLUMNS)["mean_rmse"].rank(
        method="min",
        ascending=True,
    )
    best_rmse = summary.groupby(RANK_GROUP_COLUMNS)["mean_rmse"].transform("min")
    summary["rmse_gap_to_best"] = summary["mean_rmse"] - best_rmse
    summary["pct_gap_to_best"] = summary["rmse_gap_to_best"] / best_rmse
    return summary.sort_values(
        [*RANK_GROUP_COLUMNS, "rank", "mean_mae", "representation", "model"]
    ).reset_index(drop=True)


def summarize_coefficients(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Summarize standardized linear model coefficients across evaluation groups."""
    if coefficients.empty:
        return coefficients

    summary = (
        coefficients.groupby(["target", "representation", "model", "feature"], sort=True)
        .agg(
            rows=("coefficient", "size"),
            mean_coefficient=("coefficient", "mean"),
            mean_abs_coefficient=("coefficient", lambda values: values.abs().mean()),
        )
        .reset_index()
    )
    return summary.sort_values(
        ["target", "representation", "model", "mean_abs_coefficient", "feature"],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)


def _summarize_metrics(
    metrics: pd.DataFrame,
    group_columns: list[str],
    include_mean_test_dates: bool = False,
) -> pd.DataFrame:
    aggregations = {
        "rows": ("rmse", "size"),
        "countries": ("country", "nunique"),
        "horizons": ("horizon_days", "nunique"),
        "mean_rmse": ("rmse", "mean"),
        "mean_mae": ("mae", "mean"),
        "mean_directional_accuracy": ("directional_accuracy", "mean"),
        "mean_pct_improvement_vs_train_mean": ("pct_improvement_vs_train_mean", "mean"),
    }
    if include_mean_test_dates:
        aggregations["mean_test_dates"] = ("test_dates", "mean")

    return (
        metrics.groupby(group_columns, sort=True)
        .agg(**aggregations)
        .reset_index()
        .sort_values(["mean_rmse", "mean_mae", *group_columns])
        .reset_index(drop=True)
    )


def _feature_sets(config: ProjectConfig) -> list[FeatureSet]:
    return [
        FeatureSet("pca", [f"PC{index}" for index in range(1, config.pca.n_components + 1)]),
        FeatureSet("nelson_siegel", ["beta_level", "beta_slope", "beta_curvature", "rmse"]),
        FeatureSet(
            "curve",
            [
                "level",
                "slope_10y_2y",
                "curvature_2s5s10s",
                "front_slope_2y_1y",
                "long_slope_30y_10y",
            ],
        ),
        FeatureSet("lagged", [f"lag_{lag}_change" for lag in config.evaluation.lag_days]),
        FeatureSet(
            "carry_roll",
            [
                "carry_1m",
                "roll_down_1m",
                "carry_3m",
                "roll_down_3m",
                "carry_12m",
                "roll_down_12m",
            ],
        ),
        FeatureSet(
            "residual_feature",
            [
                "residual",
                "residual_z_60",
                "residual_z_252",
                "residual_change_1",
                "residual_change_5",
                "residual_vol_20",
            ],
        ),
    ]


def feature_sets(config: ProjectConfig) -> list[FeatureSet]:
    """Return configured supervised forecast feature sets."""
    return _feature_sets(config)


def _target_specs(config: ProjectConfig) -> list[TargetSpec]:
    specs: list[TargetSpec] = []
    if config.supervised_yield_change_path.exists():
        specs.append(
            TargetSpec(
                target="yield_change",
                path=config.supervised_yield_change_path,
                target_column="target_yield_change",
            )
        )
    if config.supervised_residual_change_path.exists():
        specs.append(
            TargetSpec(
                target="residual_change",
                path=config.supervised_residual_change_path,
                target_column="target_residual_change",
            )
        )
    if config.supervised_vol_change_path.exists():
        specs.append(
            TargetSpec(
                target="vol_change",
                path=config.supervised_vol_change_path,
                target_column="target_vol_change",
            )
        )
    return specs


def _predictions(
    config: ProjectConfig,
    columns: list[str],
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
) -> list[PredictionResult]:
    predictions = [
        PredictionResult(
            model="train_mean",
            y_pred=np.full(x_test.shape[0], float(np.mean(y_train))),
            coefficients={},
        )
    ]

    for model_name, model in [
        ("ridge", Ridge(alpha=config.evaluation.ridge_alpha)),
        (
            "elastic_net",
            ElasticNet(
                alpha=config.evaluation.elastic_net_alpha,
                l1_ratio=config.evaluation.elastic_net_l1_ratio,
                max_iter=10_000,
            ),
        ),
    ]:
        pipeline = make_pipeline(StandardScaler(), model)
        pipeline.fit(x_train, y_train)
        predictions.append(
            PredictionResult(
                model=model_name,
                y_pred=pipeline.predict(x_test),
                coefficients=_standardized_coefficients(pipeline, columns),
            )
        )

    return predictions


def _standardized_coefficients(pipeline: Pipeline, columns: list[str]) -> dict[str, float]:
    model = pipeline[-1]
    coefficients = np.asarray(model.coef_, dtype=float)
    return dict(zip(columns, coefficients, strict=True))


def _metric_rows(
    target: str,
    group_values: tuple[object, ...],
    feature_set: FeatureSet,
    feature_count: int,
    y_true: NDArray[np.float64],
    predictions: list[PredictionResult],
    train_rows: int,
    test_rows: int,
    train_dates: int,
    test_dates: int,
) -> list[dict[str, object]]:
    common = _common_metric_values(
        target=target,
        group_values=group_values,
        feature_set=feature_set,
        feature_count=feature_count,
        train_rows=train_rows,
        test_rows=test_rows,
        train_dates=train_dates,
        test_dates=test_dates,
    )
    return [{**common, **_error_metrics(result.model, y_true, result.y_pred)} for result in predictions]


def _bucket_metric_rows(
    target: str,
    group_values: tuple[object, ...],
    feature_set: FeatureSet,
    feature_count: int,
    test: pd.DataFrame,
    y_true: NDArray[np.float64],
    predictions: list[PredictionResult],
    train_rows: int,
) -> list[dict[str, object]]:
    eval_frame = pd.DataFrame(
        {
            "date": test["date"].to_numpy(),
            "maturity_years": test["maturity_years"].to_numpy(dtype=float),
            "y_true": y_true,
        }
    )
    eval_frame["maturity_bucket"] = maturity_bucket(eval_frame["maturity_years"])

    rows: list[dict[str, object]] = []
    for result in predictions:
        eval_frame["y_pred"] = result.y_pred
        for bucket, bucket_frame in eval_frame.groupby("maturity_bucket", sort=True, observed=True):
            common = _common_metric_values(
                target=target,
                group_values=group_values,
                feature_set=feature_set,
                feature_count=feature_count,
                train_rows=train_rows,
                test_rows=len(bucket_frame),
                train_dates=0,
                test_dates=bucket_frame["date"].nunique(),
            )
            rows.append(
                {
                    **common,
                    "maturity_bucket": str(bucket),
                    **_error_metrics(
                        result.model,
                        bucket_frame["y_true"].to_numpy(dtype=float),
                        bucket_frame["y_pred"].to_numpy(dtype=float),
                    ),
                }
            )
    return rows


def _coefficient_rows(
    target: str,
    group_values: tuple[object, ...],
    feature_set: FeatureSet,
    predictions: list[PredictionResult],
) -> list[dict[str, object]]:
    country, horizon_days, split_method, window_id = group_values
    rows: list[dict[str, object]] = []
    for result in predictions:
        for feature, coefficient in result.coefficients.items():
            rows.append(
                {
                    "target": target,
                    "representation": feature_set.representation,
                    "model": result.model,
                    "country": country,
                    "horizon_days": int(str(horizon_days)),
                    "split_method": split_method,
                    "window_id": int(str(window_id)),
                    "feature": feature,
                    "coefficient": coefficient,
                    "abs_coefficient": abs(coefficient),
                }
            )
    return rows


def _common_metric_values(
    target: str,
    group_values: tuple[object, ...],
    feature_set: FeatureSet,
    feature_count: int,
    train_rows: int,
    test_rows: int,
    train_dates: int,
    test_dates: int,
) -> dict[str, object]:
    country, horizon_days, split_method, window_id = group_values
    return {
        "target": target,
        "representation": feature_set.representation,
        "country": country,
        "horizon_days": int(str(horizon_days)),
        "split_method": split_method,
        "window_id": int(str(window_id)),
        "feature_count": feature_count,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_dates": train_dates,
        "test_dates": test_dates,
    }


def _error_metrics(
    model: str,
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> dict[str, object]:
    return {
        "model": model,
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


def _add_improvement_vs_train_mean(metrics: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if metrics.empty:
        return metrics

    baseline_columns = [*group_columns, "representation"]
    baseline = metrics.loc[metrics["model"] == "train_mean", [*baseline_columns, "rmse"]].rename(
        columns={"rmse": "train_mean_rmse"}
    )
    output = metrics.merge(baseline, on=baseline_columns, how="left")
    output["rmse_improvement_vs_train_mean"] = output["train_mean_rmse"] - output["rmse"]
    output["pct_improvement_vs_train_mean"] = (
        output["rmse_improvement_vs_train_mean"] / output["train_mean_rmse"]
    )
    return output


def maturity_bucket(maturity_years: pd.Series) -> pd.Series:
    """Map maturities into front-end, belly, and long-end buckets."""
    return pd.cut(
        maturity_years,
        bins=[0.0, 2.0, 10.0, float("inf")],
        labels=["front_end", "belly", "long_end"],
        right=True,
    ).astype("string")
