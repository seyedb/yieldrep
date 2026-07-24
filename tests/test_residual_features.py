import pandas as pd

from yieldrep.evaluation.residual_rv import residual_mean_reversion_summary
from yieldrep.factors.residual import make_residual_features


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


def test_residual_mean_reversion_summary_measures_convergence() -> None:
    features = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3).repeat(2),
            "country": ["US"] * 6,
            "maturity_years": [2.0, 10.0] * 3,
            "residual": [1.0, -1.0, 0.8, -0.8, 0.6, -0.6],
            "residual_z_252": [1.2, -1.2, 1.1, -1.1, 1.0, -1.0],
        }
    )
    targets = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3).repeat(2),
            "country": ["US"] * 6,
            "maturity_years": [2.0, 10.0] * 3,
            "horizon_days": [5] * 6,
            "target_residual_change": [-0.2, 0.2, -0.1, 0.1, -0.05, 0.05],
        }
    )

    summary = residual_mean_reversion_summary(features, targets)

    all_residual = summary.loc[
        (summary["sample"] == "all") & (summary["signal"] == "residual")
    ]
    assert not all_residual.empty
    assert all_residual["convergence_hit_rate"].eq(1.0).all()
    assert all_residual["mean_convergence_score"].gt(0.0).all()


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
