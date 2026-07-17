from pathlib import Path

import pandas as pd
import pytest

from yieldrep.config import ProjectConfig, SourceConfig
from yieldrep.features.curve import build_curve_features, make_curve_features


def test_make_curve_features_builds_shape_features() -> None:
    features = make_curve_features(_sample_curves())

    row = features.iloc[0]
    assert row["country"] == "US"
    assert row["level"] == pytest.approx(4.48)
    assert row["slope_10y_2y"] == pytest.approx(0.4)
    assert row["curvature_2s5s10s"] == pytest.approx(0.0)
    assert row["front_slope_2y_1y"] == pytest.approx(0.1)
    assert row["long_slope_30y_10y"] == pytest.approx(1.0)


def test_build_curve_features_writes_parquet(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    _sample_curves().to_parquet(processed_dir / "curves.parquet", index=False)
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
    )

    output_path = build_curve_features(config)
    features = pd.read_parquet(output_path)

    assert output_path == processed_dir / "curve_features.parquet"
    assert {"level", "slope_10y_2y", "curvature_2s5s10s"}.issubset(features.columns)


def _sample_curves() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "country": "US",
                "maturity_years": maturity,
                "yield": yield_value,
                "source": "test",
            }
            for maturity, yield_value in [
                (1.0, 4.0),
                (2.0, 4.1),
                (5.0, 4.3),
                (10.0, 4.5),
                (30.0, 5.5),
            ]
        ]
    )
