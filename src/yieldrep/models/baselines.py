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

GROUP_COLUMNS = ["country", "horizon_days"]
MATURITY_GROUP_COLUMNS = ["country", "horizon_days", "maturity_bucket"]
PCA_FEATURES = ["PC1", "PC2", "PC3", "PC4", "PC5"]
NELSON_SIEGEL_FEATURES = ["beta_level", "beta_slope", "beta_curvature", "rmse"]
CURVE_FEATURES = [
    "level",
    "slope_10y_2y",
    "curvature_2s5s10s",
    "front_slope_2y_1y",
    "long_slope_30y_10y",
]


@dataclass(frozen=True)
class EvaluationSpec:
    representation: str
    path: Path
    features: list[str]
    target: str
    target_column: str


@dataclass(frozen=True)
class SplitWindow:
    method: str
    window_id: int
    train: pd.DataFrame
    test: pd.DataFrame


def evaluate_baselines(config: ProjectConfig) -> Path:
    """Evaluate simple forecasting baselines on prepared modeling datasets."""
    specs = _evaluation_specs(config)

    rows: list[dict[str, object]] = []
    maturity_rows: list[dict[str, object]] = []
    for spec in specs:
        if spec.path.exists():
            rows.extend(_evaluate_representation(config, spec, group_columns=GROUP_COLUMNS))
            maturity_rows.extend(
                _evaluate_representation(
                    config,
                    spec,
                    group_columns=MATURITY_GROUP_COLUMNS,
                    include_maturity_bucket=True,
                )
            )

    config.evaluation_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(config.baseline_metrics_path, index=False)
    pd.DataFrame(maturity_rows).to_parquet(config.baseline_metrics_by_maturity_path, index=False)
    return config.baseline_metrics_path


def _evaluate_representation(
    config: ProjectConfig,
    spec: EvaluationSpec,
    group_columns: list[str],
    include_maturity_bucket: bool = False,
) -> list[dict[str, object]]:
    data = pd.read_parquet(spec.path)
    feature_columns = [column for column in spec.features if column in data.columns]
    if not feature_columns:
        return []
    if include_maturity_bucket:
        data = data.copy()
        data["maturity_bucket"] = maturity_bucket(data["maturity_years"])

    rows: list[dict[str, object]] = []
    for group_values, group in data.groupby(group_columns, sort=True):
        group_key = dict(zip(group_columns, _as_tuple(group_values), strict=True))
        group = group.sort_values("date").dropna(subset=[*feature_columns, spec.target_column])
        country = str(group_key["country"])
        horizon_days = int(str(group_key["horizon_days"]))
        maturity_bucket_value = group_key.get("maturity_bucket")

        for split in evaluation_splits(
            group,
            method=config.evaluation.method,
            test_fraction=config.evaluation.test_fraction,
            min_train_dates=config.evaluation.min_train_dates,
            test_window_dates=config.evaluation.test_window_dates,
            step_dates=config.evaluation.step_dates,
        ):
            if split.train.empty or split.test.empty:
                continue

            x_train = split.train[feature_columns].to_numpy(dtype=float)
            y_train = split.train[spec.target_column].to_numpy(dtype=float)
            x_test = split.test[feature_columns].to_numpy(dtype=float)
            y_test = split.test[spec.target_column].to_numpy(dtype=float)

            rows.append(
                _metric_row(
                    target=spec.target,
                    representation=spec.representation,
                    model="train_mean",
                    split_method=split.method,
                    window_id=split.window_id,
                    country=country,
                    horizon_days=horizon_days,
                    y_true=y_test,
                    y_pred=np.full_like(y_test, fill_value=float(np.mean(y_train))),
                    train_rows=len(y_train),
                    test_rows=len(y_test),
                    train_dates=split.train["date"].nunique(),
                    test_dates=split.test["date"].nunique(),
                    maturity_bucket=maturity_bucket_value,
                )
            )

            model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=config.evaluation.ridge_alpha),
            )
            model.fit(x_train, y_train)
            rows.append(
                _metric_row(
                    target=spec.target,
                    representation=spec.representation,
                    model="ridge",
                    split_method=split.method,
                    window_id=split.window_id,
                    country=country,
                    horizon_days=horizon_days,
                    y_true=y_test,
                    y_pred=model.predict(x_test),
                    train_rows=len(y_train),
                    test_rows=len(y_test),
                    train_dates=split.train["date"].nunique(),
                    test_dates=split.test["date"].nunique(),
                    maturity_bucket=maturity_bucket_value,
                )
            )

    return rows


def _evaluation_specs(config: ProjectConfig) -> list[EvaluationSpec]:
    specs: list[EvaluationSpec] = []
    for target, suffix, target_column in [
        ("yield_change", "", "target_yield_change"),
        ("residual_change", "_residual", "target_residual_change"),
    ]:
        specs.extend(
            [
                EvaluationSpec(
                    target=target,
                    target_column=target_column,
                    representation="pca",
                    path=config.modeling_dir / f"pca{suffix}_targets.parquet",
                    features=PCA_FEATURES,
                ),
                EvaluationSpec(
                    target=target,
                    target_column=target_column,
                    representation="nelson_siegel",
                    path=config.modeling_dir / f"nelson_siegel{suffix}_targets.parquet",
                    features=NELSON_SIEGEL_FEATURES,
                ),
                EvaluationSpec(
                    target=target,
                    target_column=target_column,
                    representation="lagged",
                    path=config.modeling_dir / f"lagged{suffix}_targets.parquet",
                    features=[f"lag_{lag}_change" for lag in config.evaluation.lag_days],
                ),
                EvaluationSpec(
                    target=target,
                    target_column=target_column,
                    representation="curve",
                    path=config.modeling_dir / f"curve{suffix}_targets.parquet",
                    features=CURVE_FEATURES,
                ),
            ]
        )
    return specs


def evaluation_splits(
    data: pd.DataFrame,
    method: str,
    test_fraction: float,
    min_train_dates: int,
    test_window_dates: int,
    step_dates: int,
) -> list[SplitWindow]:
    if method == "date_ordered":
        train, test = date_ordered_split(data, test_fraction=test_fraction)
        return [SplitWindow(method=method, window_id=0, train=train, test=test)]
    if method == "walk_forward":
        return walk_forward_splits(
            data,
            min_train_dates=min_train_dates,
            test_window_dates=test_window_dates,
            step_dates=step_dates,
        )
    raise ValueError(f"Unsupported evaluation method: {method}")


def date_ordered_split(
    data: pd.DataFrame,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by unique dates so all maturities for a date remain together."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    dates = pd.Index(sorted(pd.to_datetime(data["date"]).unique()))
    split_index = int(len(dates) * (1.0 - test_fraction))
    if split_index <= 0 or split_index >= len(dates):
        return data.iloc[0:0].copy(), data.iloc[0:0].copy()

    train_dates = set(dates[:split_index])
    test_dates = set(dates[split_index:])
    normalized_dates = pd.to_datetime(data["date"])
    train = data.loc[normalized_dates.isin(train_dates)].copy()
    test = data.loc[normalized_dates.isin(test_dates)].copy()
    return train, test


def walk_forward_splits(
    data: pd.DataFrame,
    min_train_dates: int,
    test_window_dates: int,
    step_dates: int,
) -> list[SplitWindow]:
    """Create expanding-window chronological train/test splits."""
    if min_train_dates <= 0:
        raise ValueError("min_train_dates must be positive")
    if test_window_dates <= 0:
        raise ValueError("test_window_dates must be positive")
    if step_dates <= 0:
        raise ValueError("step_dates must be positive")

    dates = pd.Index(sorted(pd.to_datetime(data["date"]).unique()))
    normalized_dates = pd.to_datetime(data["date"])
    splits: list[SplitWindow] = []
    test_start = min_train_dates
    window_id = 0
    while test_start < len(dates):
        test_end = min(test_start + test_window_dates, len(dates))
        train_dates = set(dates[:test_start])
        test_dates = set(dates[test_start:test_end])
        train = data.loc[normalized_dates.isin(train_dates)].copy()
        test = data.loc[normalized_dates.isin(test_dates)].copy()
        splits.append(
            SplitWindow(
                method="walk_forward",
                window_id=window_id,
                train=train,
                test=test,
            )
        )
        window_id += 1
        test_start += step_dates
    return splits


def maturity_bucket(maturity_years: pd.Series) -> pd.Series:
    """Map maturities into front-end, belly, and long-end buckets."""
    return pd.cut(
        maturity_years,
        bins=[0.0, 2.0, 10.0, float("inf")],
        labels=["front_end", "belly", "long_end"],
        right=True,
    ).astype("string")


def _metric_row(
    target: str,
    representation: str,
    model: str,
    split_method: str,
    window_id: int,
    country: str,
    horizon_days: int,
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    train_rows: int,
    test_rows: int,
    train_dates: int,
    test_dates: int,
    maturity_bucket: object | None = None,
) -> dict[str, object]:
    row = {
        "target": target,
        "representation": representation,
        "model": model,
        "split_method": split_method,
        "window_id": window_id,
        "country": country,
        "horizon_days": horizon_days,
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_dates": train_dates,
        "test_dates": test_dates,
    }
    if maturity_bucket is not None:
        row["maturity_bucket"] = str(maturity_bucket)
    return row


def _as_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)
