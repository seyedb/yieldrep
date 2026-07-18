from pathlib import Path

import pandas as pd

from yieldrep.config import ProjectConfig, SourceConfig
from yieldrep.visualization.plotly_baselines import plot_baseline_metrics


def test_plot_baseline_metrics_writes_html_outputs(tmp_path: Path) -> None:
    evaluation_dir = tmp_path / "data" / "processed" / "evaluation"
    evaluation_dir.mkdir(parents=True)
    metrics = _sample_metrics()
    metrics.to_parquet(evaluation_dir / "baseline_metrics.parquet", index=False)
    metrics.assign(maturity_bucket=["front_end", "belly", "front_end", "belly"]).to_parquet(
        evaluation_dir / "baseline_metrics_by_maturity.parquet",
        index=False,
    )
    metrics.assign(maturity_years=[2.0, 10.0, 2.0, 10.0]).to_parquet(
        evaluation_dir / "baseline_metrics_by_maturity_point.parquet",
        index=False,
    )
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
    )

    output_paths = plot_baseline_metrics(config)

    assert output_paths == [
        tmp_path / "reports" / "figures" / "baseline_rmse_summary.html",
        tmp_path / "reports" / "figures" / "baseline_rmse_by_maturity_bucket.html",
        tmp_path / "reports" / "figures" / "baseline_rmse_by_maturity_point.html",
    ]
    assert all(path.exists() for path in output_paths)


def _sample_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": ["yield_change", "yield_change", "residual_change", "residual_change"],
            "representation": ["pca", "curve", "pca", "curve"],
            "model": ["ridge", "ridge", "train_mean", "ridge"],
            "split_method": ["date_ordered"] * 4,
            "window_id": [0] * 4,
            "country": ["US"] * 4,
            "horizon_days": [1] * 4,
            "rmse": [0.10, 0.12, 0.02, 0.03],
            "mae": [0.05, 0.06, 0.01, 0.015],
            "directional_accuracy": [0.5, 0.6, 0.55, 0.58],
            "train_rows": [10] * 4,
            "test_rows": [3] * 4,
            "train_dates": [10] * 4,
            "test_dates": [3] * 4,
        }
    )
