from pathlib import Path

import pandas as pd
import pytest

from yieldrep.config import ProjectConfig, SourceConfig, TargetConfig
from yieldrep.features.targets import (
    build_residual_targets,
    build_targets,
    make_forward_residual_change_targets,
    make_forward_yield_change_targets,
)


def test_make_forward_yield_change_targets() -> None:
    curves = _sample_curves()

    targets = make_forward_yield_change_targets(curves, horizons_days=[1, 2])

    one_day = targets.loc[
        (targets["horizon_days"] == 1)
        & (targets["country"] == "US")
        & (targets["maturity_years"] == 2.0)
    ]
    assert one_day["target_yield_change"].tolist() == pytest.approx([0.1, 0.1, 0.1])
    assert set(targets["horizon_days"]) == {1, 2}


def test_make_forward_residual_change_targets() -> None:
    fitted = _sample_fitted_curves()

    targets = make_forward_residual_change_targets(fitted, horizons_days=[1, 2])

    one_day = targets.loc[
        (targets["horizon_days"] == 1)
        & (targets["country"] == "US")
        & (targets["maturity_years"] == 2.0)
    ]
    assert one_day["target_residual_change"].tolist() == pytest.approx([0.02, 0.02, 0.02])
    assert set(targets["horizon_days"]) == {1, 2}


def test_build_targets_writes_parquet(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    _sample_curves().to_parquet(processed_dir / "curves.parquet", index=False)
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
        targets=TargetConfig(horizons_days=[1]),
    )

    output_path = build_targets(config)
    targets = pd.read_parquet(output_path)

    assert output_path == processed_dir / "targets.parquet"
    assert len(targets) == 6
    assert targets["horizon_days"].unique().tolist() == [1]


def test_build_residual_targets_writes_parquet(tmp_path: Path) -> None:
    ns_dir = tmp_path / "data" / "processed" / "nelson_siegel"
    ns_dir.mkdir(parents=True)
    _sample_fitted_curves().to_parquet(ns_dir / "us_fitted.parquet", index=False)
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
        targets=TargetConfig(horizons_days=[1]),
    )

    output_path = build_residual_targets(config)
    targets = pd.read_parquet(output_path)

    assert output_path == tmp_path / "data" / "processed" / "residual_targets.parquet"
    assert len(targets) == 6
    assert {"residual", "future_residual", "target_residual_change"}.issubset(targets.columns)


def test_make_forward_yield_change_targets_rejects_invalid_horizons() -> None:
    with pytest.raises(ValueError, match="At least one"):
        make_forward_yield_change_targets(_sample_curves(), horizons_days=[])
    with pytest.raises(ValueError, match="positive"):
        make_forward_yield_change_targets(_sample_curves(), horizons_days=[0])


def _sample_curves() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4)
    return pd.DataFrame(
        [
            {
                "date": date,
                "country": "US",
                "maturity_years": maturity,
                "yield": 3.0 + date_index * 0.1 + maturity * 0.01,
                "source": "test",
            }
            for date_index, date in enumerate(dates)
            for maturity in [2.0, 10.0]
        ]
    )


def _sample_fitted_curves() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4)
    return pd.DataFrame(
        [
            {
                "date": date,
                "country": "US",
                "maturity_years": maturity,
                "fitted_yield": 3.0 + maturity * 0.01,
                "residual": date_index * 0.02 + maturity * 0.001,
                "tau": 1.5,
            }
            for date_index, date in enumerate(dates)
            for maturity in [2.0, 10.0]
        ]
    )
