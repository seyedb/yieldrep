from pathlib import Path

import pandas as pd
import pytest

from yieldrep.config import ProjectConfig, SourceConfig
from yieldrep.evaluation.reports import baseline_winners, summarize_baselines, top_maturity_point_metrics


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
        tmp_path / "reports" / "tables" / "baseline_by_maturity_point_top.csv",
    ]
    summary = pd.read_csv(output_paths[0])
    rank_table = pd.read_csv(output_paths[1])
    winners = pd.read_csv(output_paths[2])
    bucket_summary = pd.read_csv(output_paths[3])
    point_top = pd.read_csv(output_paths[4])
    assert {"target", "representation", "model", "mean_rmse"}.issubset(summary.columns)
    assert {"rank", "rmse_gap_to_best", "pct_gap_to_best"}.issubset(rank_table.columns)
    assert winners.loc[0, "best_representation"] == "pca"
    assert winners.loc[0, "lagged_rmse_gap_to_best"] == pytest.approx(0.02)
    assert "maturity_bucket" in bucket_summary.columns
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
    ]
    for row, bucket in zip(rows, ["front_end", "belly"], strict=True):
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
) -> dict[str, object]:
    return {
        "target": "yield_change",
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
        "train_rows": 10,
        "test_rows": 3,
        "train_dates": 10,
        "test_dates": 3,
    }
