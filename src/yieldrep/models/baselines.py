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
from yieldrep.evaluation.splits import (
    SplitWindow,
    date_ordered_split,
    evaluation_splits,
    walk_forward_splits,
)

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
CARRY_ROLL_FEATURES = [
    "carry_1m",
    "roll_down_1m",
    "carry_3m",
    "roll_down_3m",
    "carry_12m",
    "roll_down_12m",
]
RESIDUAL_DYNAMIC_FEATURES = [
    "residual",
    "residual_z_60",
    "residual_z_252",
    "residual_change_1",
    "residual_change_5",
    "residual_vol_20",
]
MATURITY_BASIS_FEATURES = ["maturity", "maturity_squared", "log_maturity"]
BASE_EVALUATION_COLUMNS = ["date", "country", "maturity_years", "horizon_days"]
VOL_REGIME_LABELS = ["low", "medium", "high"]

__all__ = [
    "date_ordered_split",
    "evaluate_baselines",
    "evaluate_baseline_frames",
    "maturity_bucket",
    "walk_forward_splits",
]


@dataclass(frozen=True)
class EvaluationSpec:
    representation: str
    path: Path
    features: list[str]
    target: str
    target_column: str
    required_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedEvaluationData:
    data: pd.DataFrame
    feature_columns: list[str]


@dataclass(frozen=True)
class BaselineEvaluationFrames:
    metrics: pd.DataFrame
    metrics_by_maturity: pd.DataFrame
    metrics_by_maturity_point: pd.DataFrame
    residual_rv_spread_metrics: pd.DataFrame
    classification_metrics: pd.DataFrame


def evaluate_baselines(config: ProjectConfig) -> Path:
    """Evaluate simple forecasting baselines on prepared modeling datasets."""
    frames = evaluate_baseline_frames(config)

    config.evaluation_dir.mkdir(parents=True, exist_ok=True)
    frames.metrics.to_parquet(config.baseline_metrics_path, index=False)
    frames.metrics_by_maturity.to_parquet(config.baseline_metrics_by_maturity_path, index=False)
    frames.metrics_by_maturity_point.to_parquet(
        config.baseline_metrics_by_maturity_point_path,
        index=False,
    )
    frames.residual_rv_spread_metrics.to_parquet(
        config.baseline_residual_rv_spread_path,
        index=False,
    )
    frames.classification_metrics.to_parquet(
        config.baseline_classification_metrics_path,
        index=False,
    )
    return config.baseline_metrics_path


def evaluate_baseline_frames(config: ProjectConfig) -> BaselineEvaluationFrames:
    """Evaluate baselines and return metric frames without writing outputs."""
    specs = _evaluation_specs(config)

    rows: list[dict[str, object]] = []
    maturity_rows: list[dict[str, object]] = []
    maturity_point_rows: list[dict[str, object]] = []
    residual_rv_spread_rows: list[dict[str, object]] = []
    for spec in specs:
        prepared = _prepare_evaluation_data(spec)
        if prepared is None:
            continue

        spec_rows, spec_maturity_rows, spec_maturity_point_rows, spec_residual_rv_spread_rows = (
            _evaluate_regression_representation(config, spec, prepared)
        )
        rows.extend(spec_rows)
        maturity_rows.extend(spec_maturity_rows)
        maturity_point_rows.extend(spec_maturity_point_rows)
        residual_rv_spread_rows.extend(spec_residual_rv_spread_rows)

    classification_rows: list[dict[str, object]] = []
    for spec in _classification_specs(config):
        prepared = _prepare_evaluation_data(spec)
        if prepared is None:
            continue
        classification_rows.extend(_evaluate_classification_representation(config, spec, prepared))
    return BaselineEvaluationFrames(
        metrics=pd.DataFrame(rows),
        metrics_by_maturity=pd.DataFrame(maturity_rows),
        metrics_by_maturity_point=pd.DataFrame(maturity_point_rows),
        residual_rv_spread_metrics=pd.DataFrame(residual_rv_spread_rows),
        classification_metrics=pd.DataFrame(classification_rows),
    )


def _evaluate_regression_representation(
    config: ProjectConfig,
    spec: EvaluationSpec,
    prepared: PreparedEvaluationData,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    data = prepared.data.dropna(subset=[*prepared.feature_columns, spec.target_column])
    rows: list[dict[str, object]] = []
    maturity_rows: list[dict[str, object]] = []
    maturity_point_rows: list[dict[str, object]] = []
    residual_rv_spread_rows: list[dict[str, object]] = []
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
            max_windows=config.evaluation.walk_forward_max_windows,
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
                residual_rv_spread_rows.extend(
                    _residual_rv_spread_rows(
                        spec=spec,
                        split=split,
                        country=country,
                        horizon_days=horizon_days,
                        model_name=model_name,
                        y_true=y_test,
                        y_pred=y_pred,
                    )
                )

    return rows, maturity_rows, maturity_point_rows, residual_rv_spread_rows


def _prepare_evaluation_data(spec: EvaluationSpec) -> PreparedEvaluationData | None:
    if not spec.path.exists():
        return None

    available_columns = set(pq.read_schema(spec.path).names)
    if spec.target_column not in available_columns:
        return None

    feature_columns = [column for column in spec.features if column in available_columns]
    if not feature_columns:
        return None
    if any(column not in available_columns for column in spec.required_features):
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
            max_windows=config.evaluation.walk_forward_max_windows,
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
                    representation="carry_roll",
                    path=config.modeling_dir / f"carry_roll{suffix}_targets.parquet",
                    features=CARRY_ROLL_FEATURES,
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
        if target == "residual_change":
            specs.extend(
                [
                    EvaluationSpec(
                        target=target,
                        target_column=target_column,
                        representation="pca_maturity",
                        path=config.modeling_dir / f"pca{suffix}_targets.parquet",
                        features=_state_maturity_features(_pca_features(config)),
                        required_features=tuple(MATURITY_BASIS_FEATURES),
                    ),
                    EvaluationSpec(
                        target=target,
                        target_column=target_column,
                        representation="nelson_siegel_maturity",
                        path=config.modeling_dir / f"nelson_siegel{suffix}_targets.parquet",
                        features=_state_maturity_features(NELSON_SIEGEL_FEATURES),
                        required_features=tuple(MATURITY_BASIS_FEATURES),
                    ),
                    EvaluationSpec(
                        target=target,
                        target_column=target_column,
                        representation="curve_maturity",
                        path=config.modeling_dir / f"curve{suffix}_targets.parquet",
                        features=_state_maturity_features(CURVE_FEATURES),
                        required_features=tuple(MATURITY_BASIS_FEATURES),
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
            representation="carry_roll",
            path=config.modeling_dir / "carry_roll_vol_targets.parquet",
            features=CARRY_ROLL_FEATURES,
        ),
        EvaluationSpec(
            target="future_vol_regime",
            target_column="future_vol_regime",
            representation="residual_feature",
            path=config.modeling_dir / "residual_feature_vol_targets.parquet",
            features=RESIDUAL_DYNAMIC_FEATURES,
        ),
    ]


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
            "y_true": y_true,
            "y_pred": y_pred,
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


def _residual_rv_spread_rows(
    spec: EvaluationSpec,
    split: SplitWindow,
    country: str,
    horizon_days: int,
    model_name: str,
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> list[dict[str, object]]:
    if spec.target != "residual_change":
        return []

    eval_frame = pd.DataFrame(
        {
            "date": split.test["date"].to_numpy(),
            "maturity_years": split.test["maturity_years"].to_numpy(dtype=float),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    spreads = _top_bottom_residual_spreads(eval_frame)
    if spreads.empty:
        return []

    spread_values = spreads["spread_score"].to_numpy(dtype=float)
    spread_std = float(np.std(spread_values, ddof=1)) if len(spread_values) > 1 else float("nan")
    spread_t_stat = (
        float(np.mean(spread_values) / spread_std * np.sqrt(len(spread_values)))
        if len(spread_values) > 1 and not np.isclose(spread_std, 0.0)
        else float("nan")
    )
    return [
        {
            "target": spec.target,
            "representation": spec.representation,
            "model": model_name,
            "split_method": split.method,
            "window_id": split.window_id,
            "country": country,
            "horizon_days": horizon_days,
            "dates": len(spreads),
            "mean_spread_score": float(np.mean(spread_values)),
            "spread_t_stat": spread_t_stat,
            "hit_rate": float(np.mean(spread_values > 0.0)),
            "mean_top_realized": float(spreads["top_realized"].mean()),
            "mean_bottom_realized": float(spreads["bottom_realized"].mean()),
            "mean_leg_size": float(spreads["leg_size"].mean()),
        }
    ]


def _top_bottom_residual_spreads(eval_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, group in eval_frame.groupby("date", sort=True):
        if group["y_pred"].nunique(dropna=True) < 2:
            continue

        leg_size = max(1, int(np.floor(len(group) * 0.2)))
        leg_size = min(leg_size, len(group) // 2)
        if leg_size < 1:
            continue

        ranked = group.sort_values(["y_pred", "maturity_years"])
        bottom = ranked.head(leg_size)
        top = ranked.tail(leg_size)
        top_realized = float(top["y_true"].mean())
        bottom_realized = float(bottom["y_true"].mean())
        rows.append(
            {
                "date": date,
                "leg_size": leg_size,
                "top_realized": top_realized,
                "bottom_realized": bottom_realized,
                "spread_score": top_realized - bottom_realized,
            }
        )
    return pd.DataFrame(rows)


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
        summary["mean_rank_ic"] = float("nan")
        summary["rank_ic_dates"] = 0
    else:
        rank_ic_summary = _rank_ic_summary(eval_frame)
        summary = pd.DataFrame(
            [
                {
                    "mse": eval_frame["squared_error"].mean(),
                    "mae": eval_frame["absolute_error"].mean(),
                    "directional_accuracy": eval_frame["direction_match"].mean(),
                    "test_rows": len(eval_frame),
                    "test_dates": eval_frame["date"].nunique(),
                    "mean_rank_ic": rank_ic_summary["mean_rank_ic"],
                    "rank_ic_dates": rank_ic_summary["rank_ic_dates"],
                }
            ]
        ).assign(rmse=lambda data: np.sqrt(data["mse"]))

    rows = summary.drop(columns=["mse"]).to_dict("records")
    return [{**common, **row} for row in rows]


def _rank_ic_summary(eval_frame: pd.DataFrame) -> pd.Series:
    if eval_frame.empty:
        return pd.Series({"mean_rank_ic": float("nan"), "rank_ic_dates": 0})

    ranks = pd.DataFrame(
        {
            "date": eval_frame["date"].to_numpy(),
            "true_rank": eval_frame.groupby("date", sort=False)["y_true"].rank(method="average"),
            "pred_rank": eval_frame.groupby("date", sort=False)["y_pred"].rank(method="average"),
        }
    )
    ranks["true_rank_squared"] = np.square(ranks["true_rank"])
    ranks["pred_rank_squared"] = np.square(ranks["pred_rank"])
    ranks["rank_product"] = ranks["true_rank"] * ranks["pred_rank"]
    summary = ranks.groupby("date", sort=False).agg(
        observations=("true_rank", "size"),
        true_sum=("true_rank", "sum"),
        pred_sum=("pred_rank", "sum"),
        true_squared_sum=("true_rank_squared", "sum"),
        pred_squared_sum=("pred_rank_squared", "sum"),
        product_sum=("rank_product", "sum"),
    )
    numerator = summary["observations"] * summary["product_sum"] - (
        summary["true_sum"] * summary["pred_sum"]
    )
    true_denominator = summary["observations"] * summary["true_squared_sum"] - np.square(
        summary["true_sum"]
    )
    pred_denominator = summary["observations"] * summary["pred_squared_sum"] - np.square(
        summary["pred_sum"]
    )
    denominator = np.sqrt(true_denominator * pred_denominator)
    rank_ics = numerator.loc[denominator > 0] / denominator.loc[denominator > 0]

    return pd.Series(
        {
            "mean_rank_ic": float(rank_ics.mean()) if not rank_ics.empty else float("nan"),
            "rank_ic_dates": len(rank_ics),
        }
    )


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


def _state_maturity_features(state_features: list[str]) -> list[str]:
    interaction_features = [
        f"{state_feature}_x_{maturity_feature}"
        for state_feature in state_features
        for maturity_feature in MATURITY_BASIS_FEATURES
    ]
    return _ordered_unique([*state_features, *MATURITY_BASIS_FEATURES, *interaction_features])
