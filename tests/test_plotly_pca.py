from pathlib import Path

import pandas as pd

from yieldrep.config import ProjectConfig, SourceConfig
from yieldrep.visualization.plotly_pca import plot_pca


def test_plot_pca_writes_html_files(tmp_path: Path) -> None:
    pca_dir = tmp_path / "data" / "processed" / "pca"
    pca_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "PC1": [1.0, 0.5, -0.5],
            "PC2": [0.0, 0.2, 0.1],
            "PC3": [0.1, -0.1, 0.0],
        }
    ).to_parquet(pca_dir / "us_scores.parquet", index=False)
    pd.DataFrame(
        {
            "component": ["PC1", "PC2", "PC3"],
            "explained_variance_ratio": [0.8, 0.15, 0.05],
        }
    ).to_parquet(pca_dir / "us_variance.parquet", index=False)
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
    )

    output_paths = plot_pca(config)

    assert output_paths == [
        tmp_path / "reports" / "figures" / "us_pca_explained_variance.html",
        tmp_path / "reports" / "figures" / "us_pca_scores.html",
    ]
    for output_path in output_paths:
        assert output_path.exists()
        assert "<html>" in output_path.read_text(encoding="utf-8").lower()
