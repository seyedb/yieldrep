import pandas as pd
import pytest

from yieldrep.factors.curve import make_curve_features


def test_make_curve_features_builds_shape_features() -> None:
    features = make_curve_features(_sample_curves())

    row = features.iloc[0]
    assert row["country"] == "US"
    assert row["level"] == pytest.approx(4.48)
    assert row["slope_10y_2y"] == pytest.approx(0.4)
    assert row["curvature_2s5s10s"] == pytest.approx(0.0)
    assert row["front_slope_2y_1y"] == pytest.approx(0.1)
    assert row["long_slope_30y_10y"] == pytest.approx(1.0)

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
