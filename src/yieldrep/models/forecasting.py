from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from yieldrep.config import ProjectConfig
from yieldrep.evaluation.metrics import directional_accuracy, mae, rmse

GROUP_COLUMNS = ["country", "horizon_days", "split_method", "window_id"]
TARGET_COLUMN = "target_yield_change"


@dataclass(frozen=True)
class FeatureSet:
    representation: str
    columns: list[str]


def evaluate_supervised_forecasts(config: ProjectConfig) -> list[Path]:
    """Evaluate classical forecasting benchmarks from the canonical supervised table."""
    if not config.supervised_yield_change_path.exists():
        return []

    data = pd.read_parquet(config.supervised_yield_change_path)
    metrics = supervised_forecast_metrics(data, feature_sets=_feature_sets(config), config=config)

    config.evaluation_dir.mkdir(parents=True, exist_ok=True)
    config.tables_dir.mkdir(parents=True, exist_ok=True)

    metrics.to_parquet(config.supervised_forecast_metrics_path, index=False)
    summary = summarize_supervised_forecasts(metrics)
    summary.to_csv(config.supervised_forecast_summary_table_path, index=False)
    rank = rank_supervised_forecasts(metrics)
    rank.to_csv(config.supervised_forecast_rank_table_path, index=False)
    return [
        config.supervised_forecast_metrics_path,
        config.supervised_forecast_summary_table_path,
        config.supervised_forecast_rank_table_path,
    ]


def supervised_forecast_metrics(
    data: pd.DataFrame,
    feature_sets: list[FeatureSet],
    config: ProjectConfig,
) -> pd.DataFrame:
    """Compute train-mean and Ridge forecast metrics for each feature set."""
    rows: list[dict[str, object]] = []
    for feature_set in feature_sets:
        columns = [column for column in feature_set.columns if column in data.columns]
        if not columns:
            continue

        required = ["date", *GROUP_COLUMNS, "split", TARGET_COLUMN, *columns]
        sample = data.dropna(subset=required).loc[:, required]
        for group_values, group in sample.groupby(GROUP_COLUMNS, sort=True):
            train = group.loc[group["split"] == "train"]
            test = group.loc[group["split"] == "test"]
            if train.empty or test.empty:
                continue

            x_train = train[columns].to_numpy(dtype=float)
            y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
            x_test = test[columns].to_numpy(dtype=float)
            y_test = test[TARGET_COLUMN].to_numpy(dtype=float)

            rows.extend(
                _metric_rows(
                    group_values=group_values,
                    feature_set=feature_set,
                    feature_count=len(columns),
                    y_true=y_test,
                    predictions=_predictions(config, x_train, y_train, x_test),
                    train_rows=len(train),
                    test_rows=len(test),
                    train_dates=train["date"].nunique() if "date" in train else 0,
                    test_dates=test["date"].nunique() if "date" in test else 0,
                )
            )

    return pd.DataFrame(rows)


def summarize_supervised_forecasts(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate supervised forecast metrics by representation and model."""
    return (
        metrics.groupby(["representation", "model"], sort=True)
        .agg(
            rows=("rmse", "size"),
            countries=("country", "nunique"),
            horizons=("horizon_days", "nunique"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_directional_accuracy=("directional_accuracy", "mean"),
        )
        .reset_index()
        .sort_values(["mean_rmse", "mean_mae", "representation", "model"])
        .reset_index(drop=True)
    )


def rank_supervised_forecasts(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank supervised feature sets within each country and horizon."""
    summary = (
        metrics.groupby(["country", "horizon_days", "representation", "model"], sort=True)
        .agg(
            rows=("rmse", "size"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_directional_accuracy=("directional_accuracy", "mean"),
            mean_test_dates=("test_dates", "mean"),
        )
        .reset_index()
    )
    summary["rank"] = summary.groupby(["country", "horizon_days"])["mean_rmse"].rank(
        method="min",
        ascending=True,
    )
    best_rmse = summary.groupby(["country", "horizon_days"])["mean_rmse"].transform("min")
    summary["rmse_gap_to_best"] = summary["mean_rmse"] - best_rmse
    summary["pct_gap_to_best"] = summary["rmse_gap_to_best"] / best_rmse
    return summary.sort_values(
        ["country", "horizon_days", "rank", "mean_mae", "representation", "model"]
    ).reset_index(drop=True)


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


def _predictions(
    config: ProjectConfig,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
) -> list[tuple[str, NDArray[np.float64]]]:
    predictions = [("train_mean", np.full(x_test.shape[0], float(np.mean(y_train))))]
    model = make_pipeline(StandardScaler(), Ridge(alpha=config.evaluation.ridge_alpha))
    model.fit(x_train, y_train)
    predictions.append(("ridge", model.predict(x_test)))
    return predictions


def _metric_rows(
    group_values: tuple[object, ...],
    feature_set: FeatureSet,
    feature_count: int,
    y_true: NDArray[np.float64],
    predictions: list[tuple[str, NDArray[np.float64]]],
    train_rows: int,
    test_rows: int,
    train_dates: int,
    test_dates: int,
) -> list[dict[str, object]]:
    country, horizon_days, split_method, window_id = group_values
    return [
        {
            "representation": feature_set.representation,
            "model": model,
            "country": country,
            "horizon_days": int(str(horizon_days)),
            "split_method": split_method,
            "window_id": int(str(window_id)),
            "feature_count": feature_count,
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "directional_accuracy": directional_accuracy(y_true, y_pred),
            "train_rows": train_rows,
            "test_rows": test_rows,
            "train_dates": train_dates,
            "test_dates": test_dates,
        }
        for model, y_pred in predictions
    ]
