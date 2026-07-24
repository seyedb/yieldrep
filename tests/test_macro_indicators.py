import pandas as pd
import pytest

from yieldrep.config import SourceConfig
from yieldrep.data.macro_indicators import (
    make_macro_regimes,
    normalize_macro_indicator_source,
)


def test_normalize_unemployment_macro_indicator() -> None:
    raw = pd.DataFrame(
        {
            "observation_date": ["2024-01-01", "2024-02-01"],
            "UNRATE": ["4.0", "4.1"],
        }
    )
    source = SourceConfig(
        country="US",
        source="fred_unemployment_rate",
        raw_file="unrate.csv",
    )

    normalized = normalize_macro_indicator_source(raw, source, indicator="us_unemployment")

    assert normalized["country"].tolist() == ["US", "US"]
    assert normalized["indicator"].tolist() == ["unemployment", "unemployment"]
    assert normalized["value"].tolist() == [4.0, 4.1]


def test_normalize_cpi_index_to_yoy_inflation() -> None:
    raw = pd.DataFrame(
        {
            "observation_date": pd.date_range("2023-01-01", periods=13, freq="MS"),
            "CPIAUCSL": [100.0] * 12 + [103.0],
        }
    )
    source = SourceConfig(
        country="US",
        source="fred_cpi_index_yoy",
        raw_file="cpi.csv",
    )

    normalized = normalize_macro_indicator_source(raw, source, indicator="us_inflation")

    assert normalized["indicator"].tolist() == ["inflation"]
    assert normalized["value"].tolist() == pytest.approx([3.0])


def test_make_macro_regimes_uses_expanding_history() -> None:
    indicators = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=8, freq="MS"),
            "country": ["US"] * 8,
            "indicator": ["inflation"] * 8,
            "value": [1.0, 1.5, 2.0, 0.5, 2.5, 0.2, 3.0, 1.8],
            "source": ["test"] * 8,
        }
    )

    regimes = make_macro_regimes(indicators, min_history=3)

    assert set(regimes["macro_regime"]).issubset({"low", "medium", "high"})
    assert regimes["date"].min() > indicators["date"].min()
