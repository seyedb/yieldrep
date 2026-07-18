import pandas as pd

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
