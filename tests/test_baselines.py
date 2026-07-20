from pathlib import Path

import pandas as pd
import pytest

from yieldrep.config import EvaluationConfig, PCAConfig, ProjectConfig, SourceConfig
from yieldrep.models.baselines import (
    _pca_features,
    date_ordered_split,
    evaluate_baselines,
    maturity_bucket,
    walk_forward_splits,
)
from yieldrep.models.forecasting import evaluate_supervised_forecasts


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
    _sample_modeling_data(feature_prefix="curve").to_parquet(
        modeling_dir / "curve_targets.parquet",
        index=False,
    )
    _sample_modeling_data(feature_prefix="carry_roll").to_parquet(
        modeling_dir / "carry_roll_targets.parquet",
        index=False,
    )
    _sample_modeling_data(feature_prefix="residual_feature").to_parquet(
        modeling_dir / "residual_feature_targets.parquet",
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
    maturity_point_metrics = pd.read_parquet(
        tmp_path
        / "data"
        / "processed"
        / "evaluation"
        / "baseline_metrics_by_maturity_point.parquet"
    )

    assert output_path == tmp_path / "data" / "processed" / "evaluation" / "baseline_metrics.parquet"
    assert set(metrics["target"]) == {"yield_change"}
    expected_representations = {
        "pca",
        "nelson_siegel",
        "lagged",
        "curve",
        "carry_roll",
        "residual_feature",
    }
    assert set(metrics["representation"]) == expected_representations
    assert set(metrics["model"]) == {"train_mean", "ridge"}
    assert set(metrics["split_method"]) == {"date_ordered"}
    assert set(metrics["horizon_days"]) == {1, 5}
    assert {
        "rmse",
        "mae",
        "directional_accuracy",
        "mean_rank_ic",
        "rank_ic_dates",
        "train_rows",
        "test_rows",
        "train_dates",
        "test_dates",
    }.issubset(metrics.columns)
    assert set(metrics["train_dates"]) == {9}
    assert set(metrics["test_dates"]) == {1, 3}
    assert set(maturity_metrics["maturity_bucket"]) == {"front_end", "belly", "long_end"}
    assert set(maturity_metrics["representation"]) == expected_representations
    assert set(maturity_point_metrics["maturity_years"]) == {1.0, 5.0, 30.0}
    assert set(maturity_point_metrics["representation"]) == expected_representations


def test_evaluate_baselines_supports_residual_targets(tmp_path: Path) -> None:
    modeling_dir = tmp_path / "data" / "processed" / "modeling"
    modeling_dir.mkdir(parents=True)
    _sample_modeling_data(feature_prefix="pca", target_column="target_residual_change").to_parquet(
        modeling_dir / "pca_residual_targets.parquet",
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

    assert set(metrics["target"]) == {"residual_change"}
    assert set(metrics["representation"]) == {"pca"}
    assert set(metrics["horizon_days"]) == {1, 5}


def test_evaluate_baselines_supports_standardized_targets(tmp_path: Path) -> None:
    modeling_dir = tmp_path / "data" / "processed" / "modeling"
    modeling_dir.mkdir(parents=True)
    _sample_modeling_data(
        feature_prefix="pca",
        target_column="target_standardized_yield_change",
    ).to_parquet(
        modeling_dir / "pca_standardized_targets.parquet",
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

    assert set(metrics["target"]) == {"standardized_yield_change"}
    assert set(metrics["representation"]) == {"pca"}


def test_evaluate_baselines_supports_vol_targets(tmp_path: Path) -> None:
    modeling_dir = tmp_path / "data" / "processed" / "modeling"
    modeling_dir.mkdir(parents=True)
    _sample_modeling_data(feature_prefix="pca", target_column="target_vol_change").to_parquet(
        modeling_dir / "pca_vol_targets.parquet",
        index=False,
    )
    _sample_curve_level_data(feature_prefix="pca").to_parquet(
        modeling_dir / "pca_curve_vol_regime_targets.parquet",
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
    classification_metrics = pd.read_parquet(config.baseline_classification_metrics_path)

    assert set(metrics["target"]) == {"vol_change"}
    assert set(metrics["representation"]) == {"pca"}
    assert set(classification_metrics["target"]) == {"curve_vol_regime"}
    assert set(classification_metrics["representation"]) == {"pca"}
    assert set(classification_metrics["model"]) == {"train_mode", "logistic_l2"}
    assert {"accuracy", "balanced_accuracy", "macro_f1"}.issubset(classification_metrics.columns)


def test_evaluate_supervised_forecasts_writes_metrics_and_tables(tmp_path: Path) -> None:
    modeling_dir = tmp_path / "data" / "processed" / "modeling"
    modeling_dir.mkdir(parents=True)
    data = _sample_modeling_data(feature_prefix="pca").assign(
        split=lambda frame: [
            "train" if date < frame["date"].iloc[-3] else "test" for date in frame["date"]
        ],
        split_method="date_ordered",
        window_id=0,
    )
    data.to_parquet(modeling_dir / "supervised_yield_change.parquet", index=False)
    residual_data = data.rename(
        columns={"target_yield_change": "target_residual_change"}
    ).assign(residual=0.01)
    residual_data.to_parquet(modeling_dir / "supervised_residual_change.parquet", index=False)
    vol_data = data.rename(columns={"target_yield_change": "target_vol_change"}).assign(
        future_vol_regime="medium"
    )
    vol_data.to_parquet(modeling_dir / "supervised_vol_change.parquet", index=False)
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
        evaluation=EvaluationConfig(test_fraction=0.25, ridge_alpha=1.0),
    )

    output_paths = evaluate_supervised_forecasts(config)
    metrics = pd.read_parquet(config.supervised_forecast_metrics_path)
    bucket_metrics = pd.read_parquet(config.supervised_forecast_by_maturity_bucket_path)
    coefficients = pd.read_parquet(config.supervised_forecast_coefficients_path)
    summary = pd.read_csv(config.supervised_forecast_summary_table_path)

    assert output_paths == [
        config.supervised_forecast_metrics_path,
        config.supervised_forecast_by_maturity_bucket_path,
        config.supervised_forecast_coefficients_path,
        config.supervised_forecast_summary_table_path,
        config.supervised_forecast_rank_table_path,
        config.supervised_forecast_by_maturity_bucket_table_path,
        config.supervised_forecast_coefficients_table_path,
    ]
    assert set(metrics["representation"]) == {"pca", "residual_feature"}
    assert set(metrics["target"]) == {"yield_change", "residual_change", "vol_change"}
    assert set(metrics["model"]) == {"train_mean", "ridge", "elastic_net"}
    assert {
        "rmse",
        "mae",
        "directional_accuracy",
        "rmse_improvement_vs_train_mean",
        "pct_improvement_vs_train_mean",
    }.issubset(metrics.columns)
    assert set(bucket_metrics["maturity_bucket"]) == {"front_end", "belly", "long_end"}
    assert set(coefficients["model"]) == {"ridge", "elastic_net"}
    assert set(coefficients["target"]) == {"yield_change", "residual_change", "vol_change"}
    assert {"feature", "coefficient", "abs_coefficient"}.issubset(coefficients.columns)
    assert set(summary["representation"]) == {"pca", "residual_feature"}


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


def test_evaluate_baselines_supports_walk_forward(tmp_path: Path) -> None:
    modeling_dir = tmp_path / "data" / "processed" / "modeling"
    modeling_dir.mkdir(parents=True)
    _sample_modeling_data(feature_prefix="pca").to_parquet(
        modeling_dir / "pca_targets.parquet",
        index=False,
    )
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
        evaluation=EvaluationConfig(
            method="walk_forward",
            min_train_dates=6,
            test_window_dates=3,
            step_dates=3,
        ),
    )

    output_path = evaluate_baselines(config)
    metrics = pd.read_parquet(output_path)

    assert set(metrics["split_method"]) == {"walk_forward"}
    assert set(metrics["window_id"]) == {0, 1}
    assert set(metrics["train_dates"]) == {6, 9}
    assert set(metrics["test_dates"]) == {1, 3}


def test_evaluate_baselines_supports_non_overlapping_targets(tmp_path: Path) -> None:
    modeling_dir = tmp_path / "data" / "processed" / "modeling"
    modeling_dir.mkdir(parents=True)
    _sample_modeling_data(feature_prefix="pca").to_parquet(
        modeling_dir / "pca_targets.parquet",
        index=False,
    )
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
        evaluation=EvaluationConfig(test_fraction=0.5, non_overlapping_targets=True),
    )

    output_path = evaluate_baselines(config)
    metrics = pd.read_parquet(output_path)
    horizon_1 = metrics.loc[metrics["horizon_days"] == 1]
    horizon_5 = metrics.loc[metrics["horizon_days"] == 5]

    assert set(horizon_1["test_dates"]) == {6}
    assert set(horizon_5["test_dates"]) == {2}


def test_walk_forward_splits_use_expanding_training_window() -> None:
    data = _sample_modeling_data(feature_prefix="pca")

    splits = walk_forward_splits(
        data,
        min_train_dates=6,
        test_window_dates=3,
        step_dates=3,
    )

    assert [split.window_id for split in splits] == [0, 1]
    assert [split.train["date"].nunique() for split in splits] == [6, 9]
    assert [split.test["date"].nunique() for split in splits] == [3, 3]
    assert set(splits[0].train["date"]).isdisjoint(set(splits[0].test["date"]))


def test_walk_forward_splits_can_keep_latest_windows() -> None:
    data = _sample_modeling_data(feature_prefix="pca")

    splits = walk_forward_splits(
        data,
        min_train_dates=3,
        test_window_dates=2,
        step_dates=2,
        max_windows=2,
    )

    assert [split.window_id for split in splits] == [0, 1]
    assert [split.train["date"].nunique() for split in splits] == [9, 11]


def test_walk_forward_splits_reject_invalid_windows() -> None:
    data = _sample_modeling_data(feature_prefix="pca")

    with pytest.raises(ValueError, match="min_train_dates"):
        walk_forward_splits(
            data,
            min_train_dates=0,
            test_window_dates=3,
            step_dates=3,
        )


def test_maturity_bucket_maps_curve_segments() -> None:
    buckets = maturity_bucket(pd.Series([0.25, 2.0, 5.0, 10.0, 30.0]))

    assert buckets.tolist() == ["front_end", "front_end", "belly", "belly", "long_end"]


def test_pca_features_follow_configured_component_count(tmp_path: Path) -> None:
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
        pca=PCAConfig(n_components=7),
    )

    assert _pca_features(config) == ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"]


def _sample_modeling_data(
    feature_prefix: str,
    target_column: str = "target_yield_change",
) -> pd.DataFrame:
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
                    target_column: index * 0.01 + maturity * 0.001,
                }
                if target_column == "target_vol_change":
                    row["future_vol_regime"] = ["low", "medium", "high"][index % 3]
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
                elif feature_prefix == "lagged":
                    row.update(
                        {
                            "lag_1_change": index * 0.001,
                            "lag_5_change": index * 0.002,
                            "lag_20_change": index * 0.003,
                        }
                    )
                elif feature_prefix == "curve":
                    row.update(
                        {
                            "level": 4.0 + index * 0.01,
                            "slope_10y_2y": maturity * 0.001,
                            "curvature_2s5s10s": horizon * 0.001,
                            "front_slope_2y_1y": 0.1,
                            "long_slope_30y_10y": 0.5,
                        }
                    )
                elif feature_prefix == "carry_roll":
                    row.update(
                        {
                            "carry_1m": maturity * 0.01,
                            "roll_down_1m": -maturity * 0.001,
                            "carry_3m": maturity * 0.03,
                            "roll_down_3m": -maturity * 0.003,
                            "carry_12m": maturity * 0.12,
                            "roll_down_12m": -maturity * 0.012,
                        }
                    )
                else:
                    row.update(
                        {
                            "residual": index * 0.001,
                            "residual_z_60": index * 0.01,
                            "residual_z_252": index * 0.005,
                            "residual_change_1": 0.001,
                            "residual_change_5": 0.005,
                            "residual_vol_20": 0.002,
                        }
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def _sample_curve_level_data(feature_prefix: str) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=12)
    rows = []
    for horizon in [1, 5]:
        for index, date in enumerate(dates):
            row = {
                "date": date,
                "country": "US",
                "horizon_days": horizon,
                "realized_curve_vol": 0.01 + index * 0.001,
                "future_curve_move_rms": 0.02 + index * 0.002 + horizon * 0.0001,
                "available_maturities": 3,
            }
            if feature_prefix == "pca":
                row.update({"PC1": float(index), "PC2": float(horizon)})
            elif feature_prefix == "curve":
                row.update(
                    {
                        "level": 4.0 + index * 0.01,
                        "slope_10y_2y": index * 0.001,
                        "curvature_2s5s10s": horizon * 0.001,
                        "front_slope_2y_1y": 0.1,
                        "long_slope_30y_10y": 0.5,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)
