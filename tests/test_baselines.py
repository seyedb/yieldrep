import pytest
from pathlib import Path

import pandas as pd

from yieldrep.config import EvaluationConfig, ProjectConfig, SourceConfig
from yieldrep.models.baselines import date_ordered_split, evaluate_baselines, maturity_bucket


def test_evaluate_baselines_writes_metrics(tmp_path: Path) -> None:
    modeling_dir = tmp_path / "data" / "processed" / "modeling"
    modeling_dir.mkdir(parents=True)
    _sample_modeling_data(feature_prefix="pca").to_parquet(
        modeling_dir / "pca_targets.parquet",
        index=False,
    )
    _sample_modeling_data(feature_prefix="ns").to_parquet(
        modeling_dir / "nelson_siegel_targets.parquet",
        index=False,
    )
    _sample_modeling_data(feature_prefix="lagged").to_parquet(
        modeling_dir / "lagged_targets.parquet",
        index=False,
    )
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
        evaluation=EvaluationConfig(test_fraction=0.25, ridge_alpha=1.0),
    )

    output_path = evaluate_baselines(config)
    metrics = pd.read_parquet(output_path)
    maturity_metrics = pd.read_parquet(
        tmp_path / "data" / "processed" / "evaluation" / "baseline_metrics_by_maturity.parquet"
    )

    assert output_path == tmp_path / "data" / "processed" / "evaluation" / "baseline_metrics.parquet"
    assert set(metrics["representation"]) == {"pca", "nelson_siegel", "lagged"}
    assert set(metrics["model"]) == {"train_mean", "ridge"}
    assert set(metrics["split_method"]) == {"date_ordered"}
    assert set(metrics["horizon_days"]) == {1, 5}
    assert {
        "rmse",
        "mae",
        "directional_accuracy",
        "train_rows",
        "test_rows",
        "train_dates",
        "test_dates",
    }.issubset(metrics.columns)
    assert set(metrics["train_dates"]) == {9}
    assert set(metrics["test_dates"]) == {3}
    assert set(maturity_metrics["maturity_bucket"]) == {"front_end", "belly", "long_end"}
    assert set(maturity_metrics["representation"]) == {"pca", "nelson_siegel", "lagged"}


def test_date_ordered_split_keeps_dates_disjoint() -> None:
    data = _sample_modeling_data(feature_prefix="pca")

    train, test = date_ordered_split(data, test_fraction=0.25)

    train_dates = set(train["date"])
    test_dates = set(test["date"])
    assert train_dates.isdisjoint(test_dates)
    assert len(train_dates) == 9
    assert len(test_dates) == 3
    assert len(train) == 54
    assert len(test) == 18


def test_date_ordered_split_rejects_invalid_fraction() -> None:
    data = _sample_modeling_data(feature_prefix="pca")

    with pytest.raises(ValueError, match="between 0 and 1"):
        date_ordered_split(data, test_fraction=0.0)


def test_maturity_bucket_maps_curve_segments() -> None:
    buckets = maturity_bucket(pd.Series([0.25, 2.0, 5.0, 10.0, 30.0]))

    assert buckets.tolist() == ["front_end", "front_end", "belly", "belly", "long_end"]


def _sample_modeling_data(feature_prefix: str) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=12)
    rows = []
    for horizon in [1, 5]:
        for index, date in enumerate(dates):
            for maturity in [1.0, 5.0, 30.0]:
                row = {
                    "date": date,
                    "country": "US",
                    "maturity_years": maturity,
                    "horizon_days": horizon,
                    "yield": 4.0,
                    "future_yield": 4.0 + index * 0.01 + maturity * 0.001,
                    "target_yield_change": index * 0.01 + maturity * 0.001,
                }
                if feature_prefix == "pca":
                    row.update({"PC1": float(index), "PC2": float(horizon)})
                elif feature_prefix == "ns":
                    row.update(
                        {
                            "beta_level": float(index),
                            "beta_slope": float(horizon),
                            "beta_curvature": float(index + horizon),
                            "rmse": 0.01,
                        }
                    )
                else:
                    row.update(
                        {
                            "lag_1_change": index * 0.001,
                            "lag_5_change": index * 0.002,
                            "lag_20_change": index * 0.003,
                        }
                    )
                rows.append(row)
    return pd.DataFrame(rows)
