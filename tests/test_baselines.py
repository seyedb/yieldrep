from pathlib import Path

import pandas as pd

from yieldrep.config import EvaluationConfig, ProjectConfig, SourceConfig
from yieldrep.models.baselines import evaluate_baselines


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
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
        evaluation=EvaluationConfig(test_fraction=0.25, ridge_alpha=1.0),
    )

    output_path = evaluate_baselines(config)
    metrics = pd.read_parquet(output_path)

    assert output_path == tmp_path / "data" / "processed" / "evaluation" / "baseline_metrics.parquet"
    assert set(metrics["representation"]) == {"pca", "nelson_siegel"}
    assert set(metrics["model"]) == {"train_mean", "ridge"}
    assert set(metrics["horizon_days"]) == {1, 5}
    assert {"rmse", "mae", "directional_accuracy", "train_rows", "test_rows"}.issubset(
        metrics.columns
    )


def _sample_modeling_data(feature_prefix: str) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=12)
    rows = []
    for horizon in [1, 5]:
        for index, date in enumerate(dates):
            row = {
                "date": date,
                "country": "US",
                "maturity_years": 2.0,
                "horizon_days": horizon,
                "yield": 4.0,
                "future_yield": 4.0 + index * 0.01,
                "target_yield_change": index * 0.01,
            }
            if feature_prefix == "pca":
                row.update({"PC1": float(index), "PC2": float(horizon)})
            else:
                row.update(
                    {
                        "beta_level": float(index),
                        "beta_slope": float(horizon),
                        "beta_curvature": float(index + horizon),
                        "rmse": 0.01,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)
