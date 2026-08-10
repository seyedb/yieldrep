from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from yieldrep.config import ProjectConfig
from yieldrep.evaluation.metrics import mae, rmse
from yieldrep.evaluation.splits import date_ordered_split
from yieldrep.factors.residual import RESIDUAL_FEATURE_COLUMNS
from yieldrep.models.baselines import maturity_bucket

TARGET_COLUMN = "target_residual_change"
SUBPERIODS = [
    ("pre_2020", None, "2020-01-01"),
    ("covid_policy_period", "2020-01-01", "2022-01-01"),
    ("hiking_cycle_and_after", "2022-01-01", None),
]
RESULT_COLUMNS = [
    "country",
    "horizon_days",
    "sample",
    "rmse",
    "mae",
    "mean_rank_ic",
    "rank_ic_dates",
    "top_bottom_spread",
    "train_rows",
    "test_rows",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
]
MATURITY_BUCKET_COLUMNS = [
    "country",
    "horizon_days",
    "maturity_bucket",
    "rmse",
    "mae",
    "mean_rank_ic",
    "rank_ic_dates",
    "top_bottom_spread",
    "test_rows",
]
FEATURE_IMPORTANCE_COLUMNS = [
    "country",
    "horizon_days",
    "feature",
    "coefficient",
    "abs_coefficient",
    "rank",
    "train_rows",
    "test_rows",
]


@dataclass(frozen=True)
class ResidualRVFit:
    country: str
    horizon_days: int
    sample: str
    train: pd.DataFrame
    test: pd.DataFrame
    predictions: pd.DataFrame
    model: Pipeline


def build_residual_rv_validation_report(config: ProjectConfig) -> list[Path]:
    """Write residual RV robustness and feature-importance validation tables."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    data = _read_residual_dataset(config)

    subperiod = residual_rv_subperiod_results(data, config)
    subperiod.to_csv(config.residual_rv_subperiod_results_table_path, index=False)

    maturity = residual_rv_maturity_bucket_results(data, config)
    maturity.to_csv(config.residual_rv_maturity_bucket_results_table_path, index=False)

    importance = residual_rv_feature_importance(data, config)
    importance.to_csv(config.residual_rv_feature_importance_table_path, index=False)

    return [
        config.residual_rv_subperiod_results_table_path,
        config.residual_rv_maturity_bucket_results_table_path,
        config.residual_rv_feature_importance_table_path,
    ]


def residual_rv_subperiod_results(
    data: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    rows = []
    for (country, horizon_days), group in _country_horizon_groups(data):
        for sample, start, end in SUBPERIODS:
            period = _subperiod(group, start=start, end=end)
            fit = _fit_residual_ridge(
                period,
                country=country,
                horizon_days=horizon_days,
                sample=sample,
                test_fraction=config.evaluation.test_fraction,
            )
            if fit is not None:
                rows.append(_result_row(fit, sample_column="sample"))
    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.DataFrame(rows).loc[:, RESULT_COLUMNS].sort_values(
        ["country", "horizon_days", "sample"]
    )


def residual_rv_maturity_bucket_results(
    data: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    rows = []
    for (country, horizon_days), group in _country_horizon_groups(data):
        fit = _fit_residual_ridge(
            group,
            country=country,
            horizon_days=horizon_days,
            sample="full_sample",
            test_fraction=config.evaluation.test_fraction,
        )
        if fit is None:
            continue
        predictions = fit.predictions.copy()
        predictions["maturity_bucket"] = maturity_bucket(predictions["maturity_years"])
        for bucket, bucket_predictions in predictions.groupby("maturity_bucket", sort=True):
            rows.append(
                {
                    "country": country,
                    "horizon_days": horizon_days,
                    "maturity_bucket": bucket,
                    **_prediction_metrics(bucket_predictions),
                    "test_rows": len(bucket_predictions),
                }
            )
    if not rows:
        return pd.DataFrame(columns=MATURITY_BUCKET_COLUMNS)
    return pd.DataFrame(rows).loc[:, MATURITY_BUCKET_COLUMNS].sort_values(
        ["country", "horizon_days", "maturity_bucket"]
    )


def residual_rv_feature_importance(
    data: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    rows = []
    for (country, horizon_days), group in _country_horizon_groups(data):
        fit = _fit_residual_ridge(
            group,
            country=country,
            horizon_days=horizon_days,
            sample="full_sample",
            test_fraction=config.evaluation.test_fraction,
        )
        if fit is None:
            continue
        coefficients = _ridge_coefficients(fit.model)
        for feature, coefficient in zip(RESIDUAL_FEATURE_COLUMNS, coefficients, strict=True):
            rows.append(
                {
                    "country": country,
                    "horizon_days": horizon_days,
                    "feature": feature,
                    "coefficient": coefficient,
                    "abs_coefficient": abs(coefficient),
                    "train_rows": len(fit.train),
                    "test_rows": len(fit.test),
                }
            )
    if not rows:
        return pd.DataFrame(columns=FEATURE_IMPORTANCE_COLUMNS)
    result = pd.DataFrame(rows)
    result["rank"] = result.groupby(["country", "horizon_days"])["abs_coefficient"].rank(
        method="first",
        ascending=False,
    )
    return result.loc[:, FEATURE_IMPORTANCE_COLUMNS].sort_values(
        ["country", "horizon_days", "rank"]
    )


def _read_residual_dataset(config: ProjectConfig) -> pd.DataFrame:
    path = config.modeling_dir / "residual_residual_targets.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Residual RV modeling dataset not found: {path}")
    data = pd.read_parquet(path)
    if "residual" not in data.columns and "residual_x" in data.columns:
        data["residual"] = data["residual_x"]
    required_columns = {
        "date",
        "country",
        "maturity_years",
        "horizon_days",
        TARGET_COLUMN,
        *RESIDUAL_FEATURE_COLUMNS,
    }
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Residual RV modeling dataset is missing columns: {sorted(missing)}")
    data = data.dropna(subset=[TARGET_COLUMN, *RESIDUAL_FEATURE_COLUMNS]).copy()
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values(["country", "horizon_days", "date", "maturity_years"])


def _country_horizon_groups(data: pd.DataFrame) -> Iterator[tuple[tuple[str, int], pd.DataFrame]]:
    yield from data.groupby(["country", "horizon_days"], sort=True)


def _subperiod(data: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    period = data
    if start is not None:
        period = period.loc[period["date"] >= pd.Timestamp(start)]
    if end is not None:
        period = period.loc[period["date"] < pd.Timestamp(end)]
    return period.copy()


def _fit_residual_ridge(
    data: pd.DataFrame,
    country: str,
    horizon_days: int,
    sample: str,
    test_fraction: float,
) -> ResidualRVFit | None:
    if data.empty or data["date"].nunique() < 40:
        return None
    train, test = date_ordered_split(data, test_fraction=test_fraction)
    if train.empty or test.empty:
        return None
    if train["date"].nunique() < 20 or test["date"].nunique() < 5:
        return None

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[RESIDUAL_FEATURE_COLUMNS], train[TARGET_COLUMN])
    predictions = test.loc[:, ["date", "country", "maturity_years", "horizon_days"]].copy()
    predictions["target"] = test[TARGET_COLUMN].to_numpy(dtype=float)
    predictions["prediction"] = model.predict(test[RESIDUAL_FEATURE_COLUMNS])
    return ResidualRVFit(
        country=country,
        horizon_days=horizon_days,
        sample=sample,
        train=train,
        test=test,
        predictions=predictions,
        model=model,
    )


def _result_row(fit: ResidualRVFit, sample_column: str) -> dict[str, object]:
    return {
        "country": fit.country,
        "horizon_days": fit.horizon_days,
        sample_column: fit.sample,
        **_prediction_metrics(fit.predictions),
        "train_rows": len(fit.train),
        "test_rows": len(fit.test),
        "train_start": fit.train["date"].min().date().isoformat(),
        "train_end": fit.train["date"].max().date().isoformat(),
        "test_start": fit.test["date"].min().date().isoformat(),
        "test_end": fit.test["date"].max().date().isoformat(),
    }


def _prediction_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    rank_ics = _daily_rank_ics(predictions)
    return {
        "rmse": rmse(predictions["target"], predictions["prediction"]),
        "mae": mae(predictions["target"], predictions["prediction"]),
        "mean_rank_ic": float(rank_ics.mean()) if not rank_ics.empty else float("nan"),
        "rank_ic_dates": int(rank_ics.notna().sum()),
        "top_bottom_spread": _top_bottom_spread(predictions),
    }


def _daily_rank_ics(predictions: pd.DataFrame) -> pd.Series:
    values = []
    for _, group in predictions.groupby("date", sort=True):
        if len(group) < 3:
            continue
        predicted_rank = group["prediction"].rank(method="average")
        realized_rank = group["target"].rank(method="average")
        values.append(predicted_rank.corr(realized_rank))
    return pd.Series(values, dtype=float).dropna()


def _top_bottom_spread(predictions: pd.DataFrame) -> float:
    spreads = []
    for _, group in predictions.groupby("date", sort=True):
        if len(group) < 3:
            continue
        ordered = group.sort_values("prediction")
        n_tail = max(1, len(ordered) // 5)
        bottom = ordered.head(n_tail)["target"].mean()
        top = ordered.tail(n_tail)["target"].mean()
        spreads.append(top - bottom)
    if not spreads:
        return float("nan")
    return float(np.mean(spreads))


def _ridge_coefficients(model: Pipeline) -> NDArray[np.float64]:
    ridge = model.named_steps["ridge"]
    return np.asarray(ridge.coef_, dtype=float)
