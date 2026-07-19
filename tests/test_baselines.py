from pathlib import Path

import pandas as pd
import pytest

from yieldrep.config import EvaluationConfig, PCAConfig, ProjectConfig, SourceConfig
from yieldrep.models.baselines import (
    _maturity_aware_features,
    _pca_features,
    date_ordered_split,
    evaluate_baselines,
    maturity_bucket,
    walk_forward_splits,
)


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
        "pca_maturity",
        "nelson_siegel",
        "nelson_siegel_maturity",
        "lagged",
        "curve",
        "curve_maturity",
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
    assert metrics.loc[
        metrics["representation"].eq("pca_maturity"),
        "rank_ic_dates",
    ].gt(0).any()
    assert set(metrics["train_dates"]) == {9}
    assert set(metrics["test_dates"]) == {3}
    assert set(maturity_metrics["maturity_bucket"]) == {"front_end", "belly", "long_end"}
    assert set(maturity_metrics["representation"]) == expected_representations
    assert set(maturity_point_metrics["maturity_years"]) == {1.0, 5.0, 30.0}
    assert set(maturity_point_metrics["representation"]) == expected_representations


def test_maturity_aware_features_add_maturity_interactions() -> None:
    assert _maturity_aware_features(["PC1", "PC2"]) == [
        "maturity_years",
        "PC1",
        "PC2",
        "PC1_x_maturity",
        "PC2_x_maturity",
    ]


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
    assert set(metrics["representation"]) == {"pca", "pca_maturity"}
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
    assert set(metrics["representation"]) == {"pca", "pca_maturity"}


def test_evaluate_baselines_supports_vol_targets(tmp_path: Path) -> None:
    modeling_dir = tmp_path / "data" / "processed" / "modeling"
    modeling_dir.mkdir(parents=True)
    _sample_modeling_data(feature_prefix="pca", target_column="target_vol_change").to_parquet(
        modeling_dir / "pca_vol_targets.parquet",
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
    assert set(metrics["representation"]) == {"pca", "pca_maturity"}
    assert set(classification_metrics["target"]) == {"future_vol_regime"}
    assert set(classification_metrics["model"]) == {"train_mode", "logistic_l2"}
    assert {"accuracy", "balanced_accuracy", "macro_f1"}.issubset(classification_metrics.columns)


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
    assert set(metrics["test_dates"]) == {3}


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
