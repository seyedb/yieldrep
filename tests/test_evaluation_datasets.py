from pathlib import Path

import pandas as pd

import pytest

from yieldrep.config import EvaluationConfig, ProjectConfig, SourceConfig
from yieldrep.evaluation.datasets import build_modeling_datasets, make_lagged_yield_change_features


def test_build_modeling_datasets_joins_features_to_targets(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    dates = pd.date_range("2024-01-01", periods=2)
    _sample_curves().to_parquet(processed_dir / "curves.parquet", index=False)
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
    pd.DataFrame(
        {
            "date": dates,
            "country": ["US", "US"],
            "maturity_years": [2.0, 2.0],
            "horizon_days": [1, 1],
            "yield": [4.0, 4.1],
            "future_yield": [4.1, 4.2],
            "realized_vol": [0.05, 0.05],
            "target_yield_change": [0.1, 0.1],
            "target_standardized_yield_change": [2.0, 2.0],
        }
    ).to_parquet(processed_dir / "standardized_targets.parquet", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "country": ["US", "US"],
            "maturity_years": [2.0, 2.0],
            "horizon_days": [1, 1],
            "residual": [0.01, 0.03],
            "future_residual": [0.03, 0.02],
            "target_residual_change": [0.02, -0.01],
            "fitted_yield": [3.99, 4.07],
        }
    ).to_parquet(processed_dir / "residual_targets.parquet", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "country": ["US", "US"],
            "maturity_years": [2.0, 2.0],
            "horizon_days": [1, 1],
            "realized_vol": [0.01, 0.02],
            "future_realized_vol": [0.02, 0.03],
            "target_vol_change": [0.01, 0.01],
            "future_vol_regime": ["medium", "high"],
        }
    ).to_parquet(processed_dir / "vol_targets.parquet", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "country": ["US", "US"],
            "horizon_days": [1, 1],
            "realized_curve_vol": [0.01, 0.02],
            "future_curve_move_rms": [0.03, 0.04],
            "available_maturities": [2, 2],
        }
    ).to_parquet(processed_dir / "curve_vol_regime_targets.parquet", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "country": ["US", "US"],
            "level": [4.0, 4.1],
            "slope_10y_2y": [0.2, 0.3],
            "curvature_2s5s10s": [0.0, 0.1],
            "front_slope_2y_1y": [0.1, 0.1],
            "long_slope_30y_10y": [0.4, 0.5],
        }
    ).to_parquet(processed_dir / "curve_features.parquet", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "country": ["US", "US"],
            "maturity_years": [2.0, 2.0],
            "carry_1m": [0.33, 0.34],
            "roll_down_1m": [-0.01, -0.02],
            "carry_3m": [1.0, 1.025],
            "roll_down_3m": [-0.03, -0.04],
            "carry_12m": [4.0, 4.1],
            "roll_down_12m": [-0.2, -0.3],
        }
    ).to_parquet(processed_dir / "carry_roll_features.parquet", index=False)
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
        evaluation=EvaluationConfig(lag_days=[1]),
    )

    output_paths = build_modeling_datasets(config)

    assert set(output_paths).issuperset(
        {
            processed_dir / "modeling" / "supervised_yield_change.parquet",
            processed_dir / "modeling" / "supervised_residual_change.parquet",
            processed_dir / "modeling" / "supervised_vol_change.parquet",
            processed_dir / "modeling" / "lagged_targets.parquet",
            processed_dir / "modeling" / "curve_targets.parquet",
            processed_dir / "modeling" / "carry_roll_targets.parquet",
            processed_dir / "modeling" / "curve_vol_curve_vol_regime_targets.parquet",
        }
    )
    supervised = pd.read_parquet(processed_dir / "modeling" / "supervised_yield_change.parquet")
    supervised_residual = pd.read_parquet(
        processed_dir / "modeling" / "supervised_residual_change.parquet"
    )
    supervised_vol = pd.read_parquet(processed_dir / "modeling" / "supervised_vol_change.parquet")
    lagged_targets = pd.read_parquet(processed_dir / "modeling" / "lagged_targets.parquet")
    curve_targets = pd.read_parquet(processed_dir / "modeling" / "curve_targets.parquet")
    carry_roll_targets = pd.read_parquet(processed_dir / "modeling" / "carry_roll_targets.parquet")
    curve_vol_regime_targets = pd.read_parquet(
        processed_dir / "modeling" / "curve_vol_curve_vol_regime_targets.parquet"
    )
    assert {
        "split",
        "split_method",
        "window_id",
        "level",
        "carry_3m",
        "roll_down_3m",
        "lag_1_change",
        "target_yield_change",
    }.issubset(supervised.columns)
    assert set(supervised["split"]) == {"train", "test"}
    assert {"target_residual_change", "level", "carry_3m"}.issubset(supervised_residual.columns)
    assert set(supervised_residual["split"]) == {"train", "test"}
    assert {"target_vol_change", "future_vol_regime", "level", "carry_3m"}.issubset(
        supervised_vol.columns
    )
    assert set(supervised_vol["split"]) == {"train", "test"}
    assert {"lag_1_change", "target_yield_change"}.issubset(lagged_targets.columns)
    assert len(lagged_targets) == 1
    assert {"level", "slope_10y_2y", "target_yield_change"}.issubset(curve_targets.columns)
    assert len(curve_targets) == 2
    assert {"carry_3m", "roll_down_3m", "target_yield_change"}.issubset(
        carry_roll_targets.columns
    )
    assert len(carry_roll_targets) == 2
    assert {"realized_curve_vol", "future_curve_move_rms"}.issubset(
        curve_vol_regime_targets.columns
    )
    generated_names = {path.name for path in output_paths}
    assert not any(
        name.startswith(("pca_", "autoencoder_", "nelson_siegel_", "residual_feature_"))
        for name in generated_names
    )


def test_make_lagged_yield_change_features() -> None:
    features = make_lagged_yield_change_features(_sample_curves(), lag_days=[1, 2])

    assert features["lag_1_change"].tolist() == pytest.approx([0.1, 0.1])
    assert features["lag_2_change"].tolist() == pytest.approx([0.2, 0.2])


def _sample_curves() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4)
    return pd.DataFrame(
        {
            "date": dates,
            "country": ["US"] * 4,
            "maturity_years": [2.0] * 4,
            "yield": [4.0, 4.1, 4.2, 4.3],
            "source": ["test"] * 4,
        }
    )
