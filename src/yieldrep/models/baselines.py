from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from yieldrep.config import ProjectConfig
from yieldrep.evaluation.metrics import directional_accuracy, mae, rmse

GROUP_COLUMNS = ["country", "horizon_days"]
MATURITY_GROUP_COLUMNS = ["country", "horizon_days", "maturity_bucket"]
MATURITY_POINT_GROUP_COLUMNS = ["country", "horizon_days", "maturity_years"]
NELSON_SIEGEL_FEATURES = ["beta_level", "beta_slope", "beta_curvature", "rmse"]
CURVE_FEATURES = [
    "level",
    "slope_10y_2y",
    "curvature_2s5s10s",
    "front_slope_2y_1y",
    "long_slope_30y_10y",
]
RESIDUAL_DYNAMIC_FEATURES = [
    "residual",
    "residual_z_60",
    "residual_z_252",
    "residual_change_1",
    "residual_change_5",
    "residual_vol_20",
]
BASE_EVALUATION_COLUMNS = ["date", "country", "maturity_years", "horizon_days"]
VOL_REGIME_LABELS = ["low", "medium", "high"]


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


@dataclass(frozen=True)
class PreparedEvaluationData:
    data: pd.DataFrame
    feature_columns: list[str]


def evaluate_baselines(config: ProjectConfig) -> Path:
    """Evaluate simple forecasting baselines on prepared modeling datasets."""
    specs = _evaluation_specs(config)

    rows: list[dict[str, object]] = []
    maturity_rows: list[dict[str, object]] = []
    maturity_point_rows: list[dict[str, object]] = []
    for spec in specs:
        prepared = _prepare_evaluation_data(spec)
        if prepared is None:
            continue

        spec_rows, spec_maturity_rows, spec_maturity_point_rows = (
            _evaluate_regression_representation(config, spec, prepared)
        )
        rows.extend(spec_rows)
        maturity_rows.extend(spec_maturity_rows)
        maturity_point_rows.extend(spec_maturity_point_rows)

    config.evaluation_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(config.baseline_metrics_path, index=False)
    pd.DataFrame(maturity_rows).to_parquet(config.baseline_metrics_by_maturity_path, index=False)
    pd.DataFrame(maturity_point_rows).to_parquet(
        config.baseline_metrics_by_maturity_point_path,
        index=False,
    )

    classification_rows: list[dict[str, object]] = []
    for spec in _classification_specs(config):
        prepared = _prepare_evaluation_data(spec)
        if prepared is None:
            continue
        classification_rows.extend(_evaluate_classification_representation(config, spec, prepared))
    pd.DataFrame(classification_rows).to_parquet(
        config.baseline_classification_metrics_path,
        index=False,
    )
    return config.baseline_metrics_path


def _evaluate_regression_representation(
    config: ProjectConfig,
    spec: EvaluationSpec,
    prepared: PreparedEvaluationData,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    data = prepared.data.dropna(subset=[*prepared.feature_columns, spec.target_column])
    rows: list[dict[str, object]] = []
    maturity_rows: list[dict[str, object]] = []
    maturity_point_rows: list[dict[str, object]] = []
    for group_values, group in data.groupby(GROUP_COLUMNS, sort=True):
        group_key = dict(zip(GROUP_COLUMNS, _as_tuple(group_values), strict=True))
        group = group.sort_values("date")
        country = str(group_key["country"])
        horizon_days = int(str(group_key["horizon_days"]))

        for split in evaluation_splits(
            group,
            method=config.evaluation.method,
            test_fraction=config.evaluation.test_fraction,
            min_train_dates=config.evaluation.min_train_dates,
            test_window_dates=config.evaluation.test_window_dates,
            step_dates=config.evaluation.step_dates,
            horizon_days=horizon_days,
            non_overlapping_targets=config.evaluation.non_overlapping_targets,
        ):
            if split.train.empty or split.test.empty:
                continue

            x_train = split.train[prepared.feature_columns].to_numpy(dtype=float)
            y_train = split.train[spec.target_column].to_numpy(dtype=float)
            x_test = split.test[prepared.feature_columns].to_numpy(dtype=float)
            y_test = split.test[spec.target_column].to_numpy(dtype=float)

            prediction_rows = _regression_predictions(
                config=config,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
            )
            for model_name, y_pred in prediction_rows:
                model_rows, model_maturity_rows, model_maturity_point_rows = (
                    _regression_metric_rows(
                        spec=spec,
                        split=split,
                        country=country,
                        horizon_days=horizon_days,
                        model_name=model_name,
                        y_true=y_test,
                        y_pred=y_pred,
                        train_rows=len(y_train),
                    )
                )
                rows.extend(model_rows)
                maturity_rows.extend(model_maturity_rows)
                maturity_point_rows.extend(model_maturity_point_rows)

    return rows, maturity_rows, maturity_point_rows


def _prepare_evaluation_data(spec: EvaluationSpec) -> PreparedEvaluationData | None:
    if not spec.path.exists():
        return None

    available_columns = set(pq.read_schema(spec.path).names)
    if spec.target_column not in available_columns:
        return None

    feature_columns = [column for column in spec.features if column in available_columns]
    if not feature_columns:
        return None

    required_columns = [*BASE_EVALUATION_COLUMNS, spec.target_column]
    columns = _ordered_unique([*required_columns, *feature_columns])
    if any(column not in available_columns for column in required_columns):
        return None

    data = pd.read_parquet(spec.path, columns=columns)
    return PreparedEvaluationData(data=data, feature_columns=feature_columns)


def _regression_predictions(
    config: ProjectConfig,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
) -> list[tuple[str, NDArray[np.float64]]]:
    predictions = [("train_mean", np.full(x_test.shape[0], fill_value=float(np.mean(y_train))))]
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=config.evaluation.ridge_alpha),
    )
    model.fit(x_train, y_train)
    predictions.append(("ridge", model.predict(x_test)))
    return predictions


def _evaluate_classification_representation(
    config: ProjectConfig,
    spec: EvaluationSpec,
    prepared: PreparedEvaluationData,
) -> list[dict[str, object]]:
    data = prepared.data
    rows: list[dict[str, object]] = []
    for group_values, group in data.groupby(GROUP_COLUMNS, sort=True):
        group_key = dict(zip(GROUP_COLUMNS, _as_tuple(group_values), strict=True))
        group = group.sort_values("date").dropna(
            subset=[*prepared.feature_columns, spec.target_column]
        )
        country = str(group_key["country"])
        horizon_days = int(str(group_key["horizon_days"]))

        for split in evaluation_splits(
            group,
            method=config.evaluation.method,
            test_fraction=config.evaluation.test_fraction,
            min_train_dates=config.evaluation.min_train_dates,
            test_window_dates=config.evaluation.test_window_dates,
            step_dates=config.evaluation.step_dates,
            horizon_days=horizon_days,
            non_overlapping_targets=config.evaluation.non_overlapping_targets,
        ):
            if split.train.empty or split.test.empty:
                continue

            train = _limit_classification_training_rows(split.train, config)
            x_train = train[prepared.feature_columns].to_numpy(dtype=float)
            y_train = train[spec.target_column].astype(str).to_numpy()
            x_test = split.test[prepared.feature_columns].to_numpy(dtype=float)
            y_test = split.test[spec.target_column].astype(str).to_numpy()

            rows.extend(
                _evaluate_classification_split(
                    config=config,
                    spec=spec,
                    split=split,
                    country=country,
                    horizon_days=horizon_days,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    train_dates=train["date"].nunique(),
                )
            )
    return rows


def _evaluate_classification_split(
    config: ProjectConfig,
    spec: EvaluationSpec,
    split: SplitWindow,
    country: str,
    horizon_days: int,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.str_],
    x_test: NDArray[np.float64],
    y_test: NDArray[np.str_],
    train_dates: int,
) -> list[dict[str, object]]:
    common = {
        "target": spec.target,
        "representation": spec.representation,
        "split_method": split.method,
        "window_id": split.window_id,
        "country": country,
        "horizon_days": horizon_days,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "train_dates": train_dates,
        "test_dates": split.test["date"].nunique(),
    }

    rows = [
        _classification_metric_row(
            **common,
            model="train_mode",
            y_true=y_test,
            y_pred=_mode_predictions(y_train, len(y_test)),
        )
    ]

    if len(np.unique(y_train)) < 2:
        return rows

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config.evaluation.logistic_c,
            class_weight="balanced",
            max_iter=1000,
        ),
    )
    model.fit(x_train, y_train)
    rows.append(
        _classification_metric_row(
            **common,
            model="logistic_l2",
            y_true=y_test,
            y_pred=model.predict(x_test),
        )
    )

    return rows


def _limit_classification_training_rows(
    train: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    max_rows = config.evaluation.classification_max_train_rows
    if max_rows <= 0 or len(train) <= max_rows:
        return train
    return train.tail(max_rows)


def _mode_predictions(y_train: NDArray[np.str_], rows: int) -> NDArray[np.str_]:
    labels, counts = np.unique(y_train, return_counts=True)
    mode_label = labels[int(np.argmax(counts))]
    return np.full(rows, mode_label, dtype=labels.dtype)


def _evaluation_specs(config: ProjectConfig) -> list[EvaluationSpec]:
    specs: list[EvaluationSpec] = []
    for target, suffix, target_column in [
        ("yield_change", "", "target_yield_change"),
        (
            "standardized_yield_change",
            "_standardized",
            "target_standardized_yield_change",
        ),
        ("residual_change", "_residual", "target_residual_change"),
        ("vol_change", "_vol", "target_vol_change"),
    ]:
        specs.extend(
            [
                EvaluationSpec(
                    target=target,
                    target_column=target_column,
                    representation="pca",
                    path=config.modeling_dir / f"pca{suffix}_targets.parquet",
                    features=_pca_features(config),
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
                EvaluationSpec(
                    target=target,
                    target_column=target_column,
                    representation="residual_feature",
                    path=config.modeling_dir / f"residual_feature{suffix}_targets.parquet",
                    features=RESIDUAL_DYNAMIC_FEATURES,
                ),
            ]
        )
    return specs


def _classification_specs(config: ProjectConfig) -> list[EvaluationSpec]:
    return [
        EvaluationSpec(
            target="future_vol_regime",
            target_column="future_vol_regime",
            representation="pca",
            path=config.modeling_dir / "pca_vol_targets.parquet",
            features=_pca_features(config),
        ),
        EvaluationSpec(
            target="future_vol_regime",
            target_column="future_vol_regime",
            representation="nelson_siegel",
            path=config.modeling_dir / "nelson_siegel_vol_targets.parquet",
            features=NELSON_SIEGEL_FEATURES,
        ),
        EvaluationSpec(
            target="future_vol_regime",
            target_column="future_vol_regime",
            representation="lagged",
            path=config.modeling_dir / "lagged_vol_targets.parquet",
            features=[f"lag_{lag}_change" for lag in config.evaluation.lag_days],
        ),
        EvaluationSpec(
            target="future_vol_regime",
            target_column="future_vol_regime",
            representation="curve",
            path=config.modeling_dir / "curve_vol_targets.parquet",
            features=CURVE_FEATURES,
        ),
        EvaluationSpec(
            target="future_vol_regime",
            target_column="future_vol_regime",
            representation="residual_feature",
            path=config.modeling_dir / "residual_feature_vol_targets.parquet",
            features=RESIDUAL_DYNAMIC_FEATURES,
        ),
    ]


def evaluation_splits(
    data: pd.DataFrame,
    method: str,
    test_fraction: float,
    min_train_dates: int,
    test_window_dates: int,
    step_dates: int,
    horizon_days: int | None = None,
    non_overlapping_targets: bool = False,
) -> list[SplitWindow]:
    if method == "date_ordered":
        train, test = date_ordered_split(data, test_fraction=test_fraction)
        split = SplitWindow(method=method, window_id=0, train=train, test=test)
        return [_apply_non_overlapping_test_filter(split, horizon_days, non_overlapping_targets)]
    if method == "walk_forward":
        splits = walk_forward_splits(
            data,
            min_train_dates=min_train_dates,
            test_window_dates=test_window_dates,
            step_dates=step_dates,
        )
        return [
            _apply_non_overlapping_test_filter(split, horizon_days, non_overlapping_targets)
            for split in splits
        ]
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


def _apply_non_overlapping_test_filter(
    split: SplitWindow,
    horizon_days: int | None,
    non_overlapping_targets: bool,
) -> SplitWindow:
    if not non_overlapping_targets or horizon_days is None or horizon_days <= 1 or split.test.empty:
        return split

    dates = pd.Index(sorted(pd.to_datetime(split.test["date"]).unique()))
    keep_dates = set(dates[::horizon_days])
    normalized_dates = pd.to_datetime(split.test["date"])
    filtered_test = split.test.loc[normalized_dates.isin(keep_dates)].copy()
    return SplitWindow(
        method=split.method,
        window_id=split.window_id,
        train=split.train,
        test=filtered_test,
    )


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
    maturity_years: object | None = None,
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
    if maturity_years is not None:
        row["maturity_years"] = float(str(maturity_years))
    return row


def _regression_metric_rows(
    spec: EvaluationSpec,
    split: SplitWindow,
    country: str,
    horizon_days: int,
    model_name: str,
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    train_rows: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    eval_frame = pd.DataFrame(
        {
            "date": split.test["date"].to_numpy(),
            "maturity_years": split.test["maturity_years"].to_numpy(dtype=float),
            "squared_error": np.square(y_true - y_pred),
            "absolute_error": np.abs(y_true - y_pred),
            "direction_match": np.sign(y_true) == np.sign(y_pred),
        }
    )
    eval_frame["maturity_bucket"] = maturity_bucket(eval_frame["maturity_years"])

    common = {
        "target": spec.target,
        "representation": spec.representation,
        "model": model_name,
        "split_method": split.method,
        "window_id": split.window_id,
        "country": country,
        "horizon_days": horizon_days,
        "train_rows": train_rows,
        "train_dates": split.train["date"].nunique(),
    }
    return (
        _aggregate_regression_metrics(eval_frame, [], common),
        _aggregate_regression_metrics(eval_frame, ["maturity_bucket"], common),
        _aggregate_regression_metrics(eval_frame, ["maturity_years"], common),
    )


def _aggregate_regression_metrics(
    eval_frame: pd.DataFrame,
    group_columns: list[str],
    common: dict[str, object],
) -> list[dict[str, object]]:
    if group_columns:
        grouped = eval_frame.groupby(group_columns, sort=True, observed=True)
        summary = (
            grouped.agg(
                mse=("squared_error", "mean"),
                mae=("absolute_error", "mean"),
                directional_accuracy=("direction_match", "mean"),
                test_rows=("squared_error", "size"),
                test_dates=("date", "nunique"),
            )
            .reset_index()
            .assign(rmse=lambda data: np.sqrt(data["mse"]))
        )
    else:
        summary = pd.DataFrame(
            [
                {
                    "mse": eval_frame["squared_error"].mean(),
                    "mae": eval_frame["absolute_error"].mean(),
                    "directional_accuracy": eval_frame["direction_match"].mean(),
                    "test_rows": len(eval_frame),
                    "test_dates": eval_frame["date"].nunique(),
                }
            ]
        ).assign(rmse=lambda data: np.sqrt(data["mse"]))

    rows = summary.drop(columns=["mse"]).to_dict("records")
    return [{**common, **row} for row in rows]


def _classification_metric_row(
    target: str,
    representation: str,
    model: str,
    split_method: str,
    window_id: int,
    country: str,
    horizon_days: int,
    y_true: NDArray[np.str_],
    y_pred: NDArray[np.str_],
    train_rows: int,
    test_rows: int,
    train_dates: int,
    test_dates: int,
) -> dict[str, object]:
    accuracy, balanced_accuracy, macro_f1 = _classification_metrics(y_true, y_pred)
    return {
        "target": target,
        "representation": representation,
        "model": model,
        "split_method": split_method,
        "window_id": window_id,
        "country": country,
        "horizon_days": horizon_days,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_dates": train_dates,
        "test_dates": test_dates,
    }


def _classification_metrics(
    y_true: NDArray[np.str_],
    y_pred: NDArray[np.str_],
) -> tuple[float, float, float]:
    true_codes = _encode_vol_regimes(y_true)
    pred_codes = _encode_vol_regimes(y_pred)
    n_classes = len(VOL_REGIME_LABELS)
    confusion = np.bincount(
        true_codes * n_classes + pred_codes,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)

    total = confusion.sum()
    accuracy = float(np.trace(confusion) / total) if total else float("nan")

    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    diagonal = np.diag(confusion)
    recall = np.divide(diagonal, support, out=np.zeros_like(diagonal, dtype=float), where=support > 0)
    precision = np.divide(
        diagonal,
        predicted,
        out=np.zeros_like(diagonal, dtype=float),
        where=predicted > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(recall, dtype=float),
        where=(precision + recall) > 0,
    )

    present_true = support > 0
    present_either = (support + predicted) > 0
    balanced_accuracy = float(recall[present_true].mean()) if present_true.any() else float("nan")
    macro_f1 = float(f1[present_either].mean()) if present_either.any() else float("nan")
    return accuracy, balanced_accuracy, macro_f1


def _encode_vol_regimes(labels: NDArray[np.str_]) -> NDArray[np.int64]:
    categorical = pd.Categorical(labels, categories=VOL_REGIME_LABELS)
    codes = np.asarray(categorical.codes, dtype=np.int64)
    if (codes < 0).any():
        unknown = sorted(set(labels[codes < 0]))
        raise ValueError(f"Unknown volatility regime labels: {unknown}")
    return codes


def _as_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _pca_features(config: ProjectConfig) -> list[str]:
    return [f"PC{i}" for i in range(1, config.pca.n_components + 1)]
