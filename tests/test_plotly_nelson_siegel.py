from pathlib import Path

import pandas as pd

from yieldrep.config import ProjectConfig, SourceConfig
from yieldrep.visualization.plotly_nelson_siegel import plot_nelson_siegel


def test_plot_nelson_siegel_writes_html_files(tmp_path: Path) -> None:
    ns_dir = tmp_path / "data" / "processed" / "nelson_siegel"
    ns_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "country": ["US", "US", "US"],
            "beta_level": [4.0, 4.1, 4.2],
            "beta_slope": [-1.0, -0.9, -0.8],
            "beta_curvature": [0.5, 0.4, 0.3],
            "tau": [1.5, 1.5, 1.5],
            "rmse": [0.02, 0.03, 0.01],
        }
    ).to_parquet(ns_dir / "us_factors.parquet", index=False)
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
    )

    output_paths = plot_nelson_siegel(config)

    assert output_paths == [
        tmp_path / "reports" / "figures" / "us_nelson_siegel_factors.html",
        tmp_path / "reports" / "figures" / "us_nelson_siegel_rmse.html",
    ]
    for output_path in output_paths:
        assert output_path.exists()
        assert "<html>" in output_path.read_text(encoding="utf-8").lower()
