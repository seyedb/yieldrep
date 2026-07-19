import pandas as pd
import pytest

from yieldrep.factors.carry import make_carry_roll_features


def test_make_carry_roll_features_builds_maturity_specific_proxies() -> None:
    features = make_carry_roll_features(_sample_curves())

    row = features.loc[features["maturity_years"] == 2.0].iloc[0]
    assert row["country"] == "US"
    assert row["carry_3m"] == pytest.approx(4.2 * 0.25)
    assert row["roll_down_3m"] == pytest.approx(4.15 - 4.2)
    assert row["carry_12m"] == pytest.approx(4.2)
    assert row["roll_down_12m"] == pytest.approx(4.0 - 4.2)


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
                (2.0, 4.2),
                (5.0, 4.5),
            ]
        ]
    )
