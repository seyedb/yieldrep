from pathlib import Path

import pandas as pd
import pytest

from yieldrep.config import ProjectConfig, SourceConfig
from yieldrep.evaluation.reports import (
    baseline_winners,
    overlap_sensitivity_table,
    supervised_walk_forward_comparison,
    summarize_baselines,
    top_maturity_point_metrics,
)


def test_summarize_baselines_writes_csv_tables(tmp_path: Path) -> None:
    evaluation_dir = tmp_path / "data" / "processed" / "evaluation"
    evaluation_dir.mkdir(parents=True)
    _sample_metrics().to_parquet(evaluation_dir / "baseline_metrics.parquet", index=False)
    _sample_bucket_metrics().to_parquet(
        evaluation_dir / "baseline_metrics_by_maturity.parquet",
        index=False,
    )
    _sample_point_metrics().to_parquet(
        evaluation_dir / "baseline_metrics_by_maturity_point.parquet",
        index=False,
    )
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
    )

    output_paths = summarize_baselines(config, top_n=2)

    assert output_paths == [
        tmp_path / "reports" / "tables" / "baseline_summary.csv",
        tmp_path / "reports" / "tables" / "baseline_rank.csv",
        tmp_path / "reports" / "tables" / "baseline_winners.csv",
        tmp_path / "reports" / "tables" / "baseline_by_maturity_bucket.csv",
        tmp_path / "reports" / "tables" / "residual_relative_value.csv",
        tmp_path / "reports" / "tables" / "baseline_by_maturity_point_top.csv",
    ]
    summary = pd.read_csv(output_paths[0])
    rank_table = pd.read_csv(output_paths[1])
    winners = pd.read_csv(output_paths[2])
    bucket_summary = pd.read_csv(output_paths[3])
    residual_rv = pd.read_csv(output_paths[4])
    point_top = pd.read_csv(output_paths[5])
    assert {"target", "representation", "model", "mean_rmse"}.issubset(summary.columns)
    assert {"rank", "rmse_gap_to_best", "pct_gap_to_best", "mean_rank_ic"}.issubset(
        rank_table.columns
    )
    assert winners.loc[0, "best_representation"] == "pca"
    assert winners.loc[0, "lagged_rmse_gap_to_best"] == pytest.approx(0.02)
    assert "maturity_bucket" in bucket_summary.columns
    assert set(residual_rv["maturity_bucket"]) == {"front_end"}
    assert residual_rv.loc[0, "representation"] == "residual_feature"
    assert len(point_top) == 2
    assert point_top["rmse"].tolist() == sorted(point_top["rmse"].tolist())


def test_top_maturity_point_metrics_rejects_invalid_top_n() -> None:
    with pytest.raises(ValueError, match="top_n"):
        top_maturity_point_metrics(_sample_point_metrics(), top_n=0)


def test_baseline_winners_handles_missing_reference_representations() -> None:
    rank_table = pd.DataFrame(
        {
            "target": ["yield_change"],
            "country": ["US"],
            "horizon_days": [1],
            "representation": ["curve"],
            "model": ["ridge"],
            "mean_rmse": [0.1],
            "mean_mae": [0.05],
            "rank": [1.0],
            "rmse_gap_to_best": [0.0],
            "pct_gap_to_best": [0.0],
        }
    )

    winners = baseline_winners(rank_table)

    assert winners.loc[0, "best_representation"] == "curve"
    assert pd.isna(winners.loc[0, "pca_rank"])


def test_overlap_sensitivity_table_compares_target_windows() -> None:
    overlapping = pd.DataFrame(
        [
            _metric_row(representation="lagged", rmse=0.10, maturity_years=None),
            _metric_row(representation="pca", rmse=0.12, maturity_years=None),
        ]
    ).drop(columns=["maturity_years"])
    non_overlapping = pd.DataFrame(
        [
            _metric_row(representation="lagged", rmse=0.16, maturity_years=None),
            _metric_row(representation="pca", rmse=0.14, maturity_years=None),
        ]
    ).drop(columns=["maturity_years"])

    sensitivity = overlap_sensitivity_table(overlapping, non_overlapping)

    lagged = sensitivity.loc[sensitivity["representation"] == "lagged"].iloc[0]
    assert lagged["overlapping_rank"] == 1.0
    assert lagged["non_overlapping_rank"] == 2.0
    assert lagged["rmse_change_non_overlapping_minus_overlapping"] == pytest.approx(0.06)
    assert lagged["rank_change_non_overlapping_minus_overlapping"] == pytest.approx(1.0)


def test_supervised_walk_forward_comparison_compares_split_methods() -> None:
    date_ordered = pd.DataFrame(
        [
            _supervised_metric_row(representation="pca", rmse=0.10),
            _supervised_metric_row(representation="curve", rmse=0.12),
        ]
    )
    walk_forward = pd.DataFrame(
        [
            _supervised_metric_row(representation="pca", rmse=0.14, window_id=1),
            _supervised_metric_row(representation="curve", rmse=0.11, window_id=1),
        ]
    )

    comparison = supervised_walk_forward_comparison(date_ordered, walk_forward)

    pca = comparison.loc[comparison["representation"] == "pca"].iloc[0]
    assert pca["date_ordered_rank"] == 1.0
    assert pca["walk_forward_rank"] == 2.0
    assert pca["rmse_change_walk_forward_minus_date_ordered"] == pytest.approx(0.04)
    assert pca["rank_change_walk_forward_minus_date_ordered"] == pytest.approx(1.0)


def _sample_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _metric_row(representation="pca", rmse=0.10, maturity_years=None),
            _metric_row(representation="pca", rmse=0.10, maturity_years=None),
            _metric_row(representation="lagged", rmse=0.12, maturity_years=None),
            _metric_row(representation="curve", rmse=0.15, maturity_years=None),
        ]
    ).drop(columns=["maturity_years"])


def _sample_bucket_metrics() -> pd.DataFrame:
    rows = [
        _metric_row(representation="pca", rmse=0.10, maturity_years=None),
        _metric_row(representation="curve", rmse=0.15, maturity_years=None),
        _metric_row(
            representation="residual_feature",
            rmse=0.08,
            maturity_years=None,
            target="residual_change",
        ),
        _metric_row(
            representation="pca",
            rmse=0.10,
            maturity_years=None,
            target="residual_change",
        ),
    ]
    for row, bucket in zip(rows, ["front_end", "belly", "front_end", "front_end"], strict=True):
        row["maturity_bucket"] = bucket
    return pd.DataFrame(rows).drop(columns=["maturity_years"])


def _sample_point_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _metric_row(representation="pca", rmse=0.10, maturity_years=2.0),
            _metric_row(representation="curve", rmse=0.08, maturity_years=5.0),
            _metric_row(representation="lagged", rmse=0.12, maturity_years=10.0),
        ]
    )


def _metric_row(
    representation: str,
    rmse: float,
    maturity_years: float | None,
    target: str = "yield_change",
) -> dict[str, object]:
    return {
        "target": target,
        "representation": representation,
        "model": "ridge",
        "split_method": "date_ordered",
        "window_id": 0,
        "country": "US",
        "horizon_days": 1,
        "maturity_years": maturity_years,
        "rmse": rmse,
        "mae": rmse / 2.0,
        "directional_accuracy": 0.5,
        "mean_rank_ic": 0.1,
        "rank_ic_dates": 3,
        "train_rows": 10,
        "test_rows": 3,
        "train_dates": 10,
        "test_dates": 3,
    }


def _supervised_metric_row(
    representation: str,
    rmse: float,
    window_id: int = 0,
) -> dict[str, object]:
    return {
        "target": "yield_change",
        "representation": representation,
        "model": "ridge",
        "country": "US",
        "horizon_days": 1,
        "split_method": "date_ordered" if window_id == 0 else "walk_forward",
        "window_id": window_id,
        "feature_count": 2,
        "rmse": rmse,
        "mae": rmse / 2.0,
        "directional_accuracy": 0.5,
        "train_rows": 10,
        "test_rows": 3,
        "train_dates": 10,
        "test_dates": 3,
        "train_mean_rmse": 0.15,
        "rmse_improvement_vs_train_mean": 0.15 - rmse,
        "pct_improvement_vs_train_mean": (0.15 - rmse) / 0.15,
    }
