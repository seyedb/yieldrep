from pathlib import Path

import pandas as pd

import pytest

from yieldrep.config import EvaluationConfig, ProjectConfig, SourceConfig
from yieldrep.evaluation.datasets import build_modeling_datasets, make_lagged_yield_change_features


def test_build_modeling_datasets_joins_features_to_targets(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    pca_dir = processed_dir / "pca"
    ns_dir = processed_dir / "nelson_siegel"
    pca_dir.mkdir(parents=True)
    ns_dir.mkdir(parents=True)

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
            "horizon_days": [1, 1],
            "future_PC1": [1.1, 1.2],
            "future_PC2": [0.2, 0.3],
            "future_PC3": [-0.1, 0.0],
        }
    ).to_parquet(processed_dir / "curve_state_targets.parquet", index=False)
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
    pd.DataFrame(
        {
            "date": dates,
            "country": ["US", "US"],
            "maturity_years": [2.0, 2.0],
            "residual": [0.01, 0.03],
            "residual_z_60": [0.5, 1.0],
            "residual_z_252": [0.2, 0.4],
            "residual_change_1": [0.01, 0.02],
            "residual_change_5": [0.03, 0.04],
            "residual_vol_20": [0.01, 0.02],
        }
    ).to_parquet(processed_dir / "residual_features.parquet", index=False)
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
            processed_dir / "modeling" / "pca_targets.parquet",
            processed_dir / "modeling" / "nelson_siegel_targets.parquet",
            processed_dir / "modeling" / "lagged_targets.parquet",
            processed_dir / "modeling" / "curve_targets.parquet",
            processed_dir / "modeling" / "carry_roll_targets.parquet",
            processed_dir / "modeling" / "residual_feature_targets.parquet",
            processed_dir / "modeling" / "pca_standardized_targets.parquet",
            processed_dir / "modeling" / "pca_residual_targets.parquet",
            processed_dir / "modeling" / "residual_feature_residual_targets.parquet",
            processed_dir / "modeling" / "pca_vol_targets.parquet",
            processed_dir / "modeling" / "residual_feature_vol_targets.parquet",
            processed_dir / "modeling" / "pca_curve_vol_regime_targets.parquet",
            processed_dir / "modeling" / "curve_vol_curve_vol_regime_targets.parquet",
            processed_dir / "modeling" / "pca_curve_state_targets.parquet",
            processed_dir / "modeling" / "nelson_siegel_curve_state_targets.parquet",
            processed_dir / "modeling" / "curve_curve_state_targets.parquet",
        }
    )
    pca_targets = pd.read_parquet(processed_dir / "modeling" / "pca_targets.parquet")
    supervised = pd.read_parquet(processed_dir / "modeling" / "supervised_yield_change.parquet")
    supervised_residual = pd.read_parquet(
        processed_dir / "modeling" / "supervised_residual_change.parquet"
    )
    supervised_vol = pd.read_parquet(processed_dir / "modeling" / "supervised_vol_change.parquet")
    ns_targets = pd.read_parquet(processed_dir / "modeling" / "nelson_siegel_targets.parquet")
    lagged_targets = pd.read_parquet(processed_dir / "modeling" / "lagged_targets.parquet")
    curve_targets = pd.read_parquet(processed_dir / "modeling" / "curve_targets.parquet")
    carry_roll_targets = pd.read_parquet(processed_dir / "modeling" / "carry_roll_targets.parquet")
    pca_residual_targets = pd.read_parquet(
        processed_dir / "modeling" / "pca_residual_targets.parquet"
    )
    pca_standardized_targets = pd.read_parquet(
        processed_dir / "modeling" / "pca_standardized_targets.parquet"
    )
    pca_vol_targets = pd.read_parquet(processed_dir / "modeling" / "pca_vol_targets.parquet")
    pca_curve_vol_regime_targets = pd.read_parquet(
        processed_dir / "modeling" / "pca_curve_vol_regime_targets.parquet"
    )
    curve_vol_regime_targets = pd.read_parquet(
        processed_dir / "modeling" / "curve_vol_curve_vol_regime_targets.parquet"
    )
    pca_curve_state_targets = pd.read_parquet(
        processed_dir / "modeling" / "pca_curve_state_targets.parquet"
    )
    residual_feature_targets = pd.read_parquet(
        processed_dir / "modeling" / "residual_feature_targets.parquet"
    )
    assert {"PC1", "PC2", "target_yield_change"}.issubset(pca_targets.columns)
    assert {
        "split",
        "split_method",
        "window_id",
        "PC1",
        "beta_level",
        "level",
        "carry_3m",
        "roll_down_3m",
        "lag_1_change",
        "residual_z_60",
        "target_yield_change",
    }.issubset(supervised.columns)
    assert set(supervised["split"]) == {"train", "test"}
    assert {"target_residual_change", "PC1", "residual_z_60"}.issubset(
        supervised_residual.columns
    )
    assert set(supervised_residual["split"]) == {"train", "test"}
    assert {"target_vol_change", "future_vol_regime", "PC1", "residual_z_60"}.issubset(
        supervised_vol.columns
    )
    assert set(supervised_vol["split"]) == {"train", "test"}
    assert {"beta_level", "beta_slope", "beta_curvature", "target_yield_change"}.issubset(
        ns_targets.columns
    )
    assert len(pca_targets) == 2
    assert len(ns_targets) == 2
    assert {"lag_1_change", "target_yield_change"}.issubset(lagged_targets.columns)
    assert len(lagged_targets) == 1
    assert {"level", "slope_10y_2y", "target_yield_change"}.issubset(curve_targets.columns)
    assert len(curve_targets) == 2
    assert {"carry_3m", "roll_down_3m", "target_yield_change"}.issubset(
        carry_roll_targets.columns
    )
    assert len(carry_roll_targets) == 2
    assert {"PC1", "target_residual_change"}.issubset(pca_residual_targets.columns)
    assert len(pca_residual_targets) == 2
    assert {"PC1", "target_standardized_yield_change"}.issubset(
        pca_standardized_targets.columns
    )
    assert len(pca_standardized_targets) == 2
    assert {"PC1", "target_vol_change"}.issubset(pca_vol_targets.columns)
    assert len(pca_vol_targets) == 2
    assert {"PC1", "future_curve_move_rms"}.issubset(pca_curve_vol_regime_targets.columns)
    assert {"realized_curve_vol", "future_curve_move_rms"}.issubset(
        curve_vol_regime_targets.columns
    )
    assert {"PC1", "future_PC1", "future_PC3"}.issubset(pca_curve_state_targets.columns)
    assert {"residual_z_60", "residual_change_5", "target_yield_change"}.issubset(
        residual_feature_targets.columns
    )
    assert len(residual_feature_targets) == 2


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
