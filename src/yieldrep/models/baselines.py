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

TARGET_COLUMN = "target_yield_change"
GROUP_COLUMNS = ["country", "horizon_days"]
MATURITY_GROUP_COLUMNS = ["country", "horizon_days", "maturity_bucket"]
PCA_FEATURES = ["PC1", "PC2", "PC3", "PC4", "PC5"]
NELSON_SIEGEL_FEATURES = ["beta_level", "beta_slope", "beta_curvature", "rmse"]
SPLIT_METHOD = "date_ordered"


@dataclass(frozen=True)
class EvaluationSpec:
    representation: str
    path: Path
    features: list[str]


def evaluate_baselines(config: ProjectConfig) -> Path:
    """Evaluate simple forecasting baselines on prepared modeling datasets."""
    specs = [
        EvaluationSpec(
            representation="pca",
            path=config.modeling_dir / "pca_targets.parquet",
            features=PCA_FEATURES,
        ),
        EvaluationSpec(
            representation="nelson_siegel",
            path=config.modeling_dir / "nelson_siegel_targets.parquet",
            features=NELSON_SIEGEL_FEATURES,
        ),
        EvaluationSpec(
            representation="lagged",
            path=config.modeling_dir / "lagged_targets.parquet",
            features=[f"lag_{lag}_change" for lag in config.evaluation.lag_days],
        ),
    ]

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
        group = group.sort_values("date").dropna(subset=[*feature_columns, TARGET_COLUMN])
        train, test = date_ordered_split(group, test_fraction=config.evaluation.test_fraction)
        if train.empty or test.empty:
            continue
        country = str(group_key["country"])
        horizon_days = int(str(group_key["horizon_days"]))
        maturity_bucket_value = group_key.get("maturity_bucket")

        x_train = train[feature_columns].to_numpy(dtype=float)
        y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
        x_test = test[feature_columns].to_numpy(dtype=float)
        y_test = test[TARGET_COLUMN].to_numpy(dtype=float)

        rows.append(
            _metric_row(
                representation=spec.representation,
                model="train_mean",
                country=country,
                horizon_days=horizon_days,
                y_true=y_test,
                y_pred=np.full_like(y_test, fill_value=float(np.mean(y_train))),
                train_rows=len(y_train),
                test_rows=len(y_test),
                train_dates=train["date"].nunique(),
                test_dates=test["date"].nunique(),
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
                representation=spec.representation,
                model="ridge",
                country=country,
                horizon_days=horizon_days,
                y_true=y_test,
                y_pred=model.predict(x_test),
                train_rows=len(y_train),
                test_rows=len(y_test),
                train_dates=train["date"].nunique(),
                test_dates=test["date"].nunique(),
                maturity_bucket=maturity_bucket_value,
            )
        )

    return rows


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


def maturity_bucket(maturity_years: pd.Series) -> pd.Series:
    """Map maturities into front-end, belly, and long-end buckets."""
    return pd.cut(
        maturity_years,
        bins=[0.0, 2.0, 10.0, float("inf")],
        labels=["front_end", "belly", "long_end"],
        right=True,
    ).astype("string")


def _metric_row(
    representation: str,
    model: str,
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
        "representation": representation,
        "model": model,
        "split_method": SPLIT_METHOD,
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
