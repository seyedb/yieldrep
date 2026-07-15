from pathlib import Path

import pandas as pd

from yieldrep.config import ProjectConfig, SourceConfig
from yieldrep.evaluation.datasets import build_modeling_datasets


def test_build_modeling_datasets_joins_features_to_targets(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    pca_dir = processed_dir / "pca"
    ns_dir = processed_dir / "nelson_siegel"
    pca_dir.mkdir(parents=True)
    ns_dir.mkdir(parents=True)

    dates = pd.date_range("2024-01-01", periods=2)
    pd.DataFrame(
        {
            "date": dates,
            "country": ["US", "US"],
            "maturity_years": [2.0, 2.0],
            "horizon_days": [1, 1],
            "yield": [4.0, 4.1],
            "future_yield": [4.1, 4.2],
            "target_yield_change": [0.1, 0.1],
        }
    ).to_parquet(processed_dir / "targets.parquet", index=False)
    pd.DataFrame({"date": dates, "PC1": [1.0, 1.1], "PC2": [0.1, 0.2]}).to_parquet(
        pca_dir / "us_scores.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "date": dates,
            "country": ["US", "US"],
            "beta_level": [4.0, 4.1],
            "beta_slope": [-1.0, -0.9],
            "beta_curvature": [0.5, 0.4],
            "tau": [1.5, 1.5],
            "rmse": [0.01, 0.02],
        }
    ).to_parquet(ns_dir / "us_factors.parquet", index=False)
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
    )

    output_paths = build_modeling_datasets(config)

    assert output_paths == [
        processed_dir / "modeling" / "pca_targets.parquet",
        processed_dir / "modeling" / "nelson_siegel_targets.parquet",
    ]
    pca_targets = pd.read_parquet(processed_dir / "modeling" / "pca_targets.parquet")
    ns_targets = pd.read_parquet(processed_dir / "modeling" / "nelson_siegel_targets.parquet")
    assert {"PC1", "PC2", "target_yield_change"}.issubset(pca_targets.columns)
    assert {"beta_level", "beta_slope", "beta_curvature", "target_yield_change"}.issubset(
        ns_targets.columns
    )
    assert len(pca_targets) == 2
    assert len(ns_targets) == 2
