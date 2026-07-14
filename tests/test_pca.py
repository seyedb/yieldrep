from pathlib import Path

import pandas as pd

from yieldrep.config import PCAConfig, ProjectConfig, SourceConfig
from yieldrep.features.curve import curve_panel
from yieldrep.features.pca import build_pca


def test_curve_panel_pivots_one_country() -> None:
    curves = _sample_curves()

    panel = curve_panel(curves, "US")

    assert panel.shape == (4, 3)
    assert panel.columns.tolist() == [1.0, 2.0, 10.0]


def test_build_pca_writes_expected_outputs(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    _sample_curves().to_parquet(processed_dir / "curves.parquet", index=False)
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
        pca=PCAConfig(n_components=2, min_maturities=3),
    )

    output_paths = build_pca(config)

    assert output_paths == [
        processed_dir / "pca" / "us_scores.parquet",
        processed_dir / "pca" / "us_loadings.parquet",
        processed_dir / "pca" / "us_variance.parquet",
    ]
    scores = pd.read_parquet(processed_dir / "pca" / "us_scores.parquet")
    loadings = pd.read_parquet(processed_dir / "pca" / "us_loadings.parquet")
    variance = pd.read_parquet(processed_dir / "pca" / "us_variance.parquet")

    assert scores.shape == (4, 3)
    assert loadings.shape == (3, 3)
    assert variance.shape == (2, 2)
    assert variance["component"].tolist() == ["PC1", "PC2"]


def _sample_curves() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4)
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
