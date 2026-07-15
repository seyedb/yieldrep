from pathlib import Path

import pandas as pd

from yieldrep.config import ProjectConfig, SourceConfig
from yieldrep.visualization.plotly_curves import plot_curves


def test_plot_curves_writes_html_files(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    _sample_curves().to_parquet(processed_dir / "curves.parquet", index=False)
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
    )

    output_paths = plot_curves(config)

    assert output_paths == [
        tmp_path / "reports" / "figures" / "us_selected_maturities.html",
        tmp_path / "reports" / "figures" / "us_curve_heatmap.html",
    ]
    for output_path in output_paths:
        assert output_path.exists()
        assert "<html>" in output_path.read_text(encoding="utf-8").lower()


def _sample_curves() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=3)
    return pd.DataFrame(
        [
            {
                "date": date,
                "country": "US",
                "maturity_years": maturity,
                "yield": 3.0 + date_index * 0.1 + maturity * 0.02,
                "source": "test",
            }
            for date_index, date in enumerate(dates)
            for maturity in [1.0, 2.0, 10.0]
        ]
    )
