from pathlib import Path

import pandas as pd

from yieldrep.config import ProjectConfig, SourceConfig
from yieldrep.factors.residual import build_residual_features, make_residual_features


def test_make_residual_features_builds_dynamic_features() -> None:
    features = make_residual_features(_sample_fitted_curves())

    assert not features.empty
    assert {
        "residual_z_60",
        "residual_z_252",
        "residual_change_1",
        "residual_change_5",
        "residual_vol_20",
    }.issubset(features.columns)
    assert features["residual_z_252"].notna().all()


def test_build_residual_features_writes_parquet(tmp_path: Path) -> None:
    ns_dir = tmp_path / "data" / "processed" / "nelson_siegel"
    ns_dir.mkdir(parents=True)
    _sample_fitted_curves().to_parquet(ns_dir / "us_fitted.parquet", index=False)
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
    )

    output_path = build_residual_features(config)
    features = pd.read_parquet(output_path)

    assert output_path == tmp_path / "data" / "processed" / "residual_features.parquet"
    assert not features.empty


def _sample_fitted_curves() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=260)
    return pd.DataFrame(
        [
            {
                "date": date,
                "country": "US",
                "maturity_years": 2.0,
                "fitted_yield": 4.0,
                "residual": 0.01 * date_index + 0.002 * (date_index % 5),
                "tau": 1.5,
            }
            for date_index, date in enumerate(dates)
        ]
    )
