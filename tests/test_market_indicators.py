import pandas as pd

from yieldrep.config import SourceConfig
from yieldrep.data.market_indicators import (
    make_market_regimes,
    normalize_market_indicator_source,
)


def test_normalize_fred_market_indicator_source() -> None:
    raw = pd.DataFrame(
        {
            "observation_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "VIXCLS": ["13.2", ".", "14.1"],
        }
    )
    source = SourceConfig(
        country="GLOBAL",
        source="fred_vixcls",
        raw_file="vix.csv",
    )

    normalized = normalize_market_indicator_source(raw, source, indicator="vix")

    assert normalized["indicator"].tolist() == ["VIX", "VIX"]
    assert normalized["value"].tolist() == [13.2, 14.1]


def test_normalize_yahoo_move_source() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": [1_704_067_200, 1_704_153_600],
            "close": [110.5, None],
        }
    )
    source = SourceConfig(
        country="GLOBAL",
        source="yahoo_move",
        raw_file="move.json",
    )

    normalized = normalize_market_indicator_source(raw, source, indicator="move")

    assert normalized["indicator"].tolist() == ["MOVE"]
    assert normalized["value"].tolist() == [110.5]


def test_make_market_regimes_uses_expanding_history() -> None:
    indicators = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8),
            "indicator": ["MOVE"] * 8,
            "value": [80.0, 90.0, 100.0, 70.0, 110.0, 60.0, 120.0, 95.0],
            "source": ["fred_move"] * 8,
        }
    )

    regimes = make_market_regimes(indicators, min_history=3)

    assert set(regimes["market_vol_regime"]).issubset({"low", "medium", "high"})
    assert regimes["date"].min() > indicators["date"].min()
