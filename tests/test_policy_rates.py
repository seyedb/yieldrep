from pathlib import Path

import pandas as pd
import pytest

from yieldrep.config import SourceConfig
from yieldrep.data.policy_rates import normalize_policy_rate_source
from yieldrep.factors.policy import make_policy_features


def test_normalize_policy_rate_sources() -> None:
    fred = pd.DataFrame({"observation_date": ["2024-01-01"], "DFF": [5.33]})
    boc = pd.DataFrame({"date": ["2024-01-01"], "V39079": [5.0]})
    ecb = pd.DataFrame({"TIME_PERIOD": ["2024-01-01"], "OBS_VALUE": [4.0]})

    frames = [
        normalize_policy_rate_source(
            fred,
            SourceConfig(country="US", source="fred_dff", raw_file=Path("fed.csv")),
        ),
        normalize_policy_rate_source(
            boc,
            SourceConfig(country="CA", source="bank_of_canada_policy_rate", raw_file=Path("boc.csv")),
        ),
        normalize_policy_rate_source(
            ecb,
            SourceConfig(country="EA", source="ecb_deposit_facility", raw_file=Path("ecb.csv")),
        ),
    ]

    normalized = pd.concat(frames, ignore_index=True)

    assert set(normalized["country"]) == {"US", "CA", "EA"}
    assert normalized["policy_rate"].tolist() == pytest.approx([5.33, 5.0, 4.0])


def test_make_policy_features_aligns_to_curve_dates() -> None:
    dates = pd.date_range("2024-01-01", periods=70)
    curves = pd.DataFrame(
        [
            {
                "date": date,
                "country": "US",
                "maturity_years": maturity,
                "yield": 4.0 + index * 0.01 + maturity * 0.1,
                "source": "test",
            }
            for index, date in enumerate(dates)
            for maturity in [2.0, 10.0]
        ]
    )
    policy_rates = pd.DataFrame(
        {
            "date": dates[::5],
            "country": ["US"] * len(dates[::5]),
            "policy_rate": [5.0 + index * 0.05 for index in range(len(dates[::5]))],
            "source": ["test"] * len(dates[::5]),
        }
    )

    features = make_policy_features(policy_rates, curves)

    assert {"policy_rate", "policy_change_21d", "policy_change_63d", "policy_2y_spread"}.issubset(
        features.columns
    )
    assert features["policy_rate"].notna().all()
    assert features["policy_2y_spread"].notna().all()
    assert set(features["policy_regime_63d"].dropna()).issubset({"easing", "stable", "hiking"})
