import pandas as pd
import pytest

from yieldrep.data.normalize import (
    normalize_bank_of_canada,
    normalize_ecb_yield_curve,
    normalize_fed_gsw,
)
from yieldrep.data.schema import CURVE_COLUMNS


def test_normalize_fed_gsw_returns_common_schema() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "BETA0": [1.0, 1.1],
            "SVENY01": [4.0, 4.1],
            "SVENY10": [4.2, 4.3],
        }
    )

    curves = normalize_fed_gsw(raw)

    assert tuple(curves.columns) == CURVE_COLUMNS
    assert len(curves) == 4
    assert set(curves["country"]) == {"US"}
    assert set(curves["source"]) == {"fed_gsw"}
    assert set(curves["maturity_years"]) == {1.0, 10.0}
    assert curves["yield"].tolist() == [4.0, 4.2, 4.1, 4.3]


def test_normalize_bank_of_canada_scales_decimal_yields_to_percent() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-02"],
            "ZC025YR": [0.0400],
            "ZC100YR": [0.0425],
        }
    )

    curves = normalize_bank_of_canada(raw)

    assert tuple(curves.columns) == CURVE_COLUMNS
    assert set(curves["country"]) == {"CA"}
    assert set(curves["source"]) == {"bank_of_canada"}
    assert curves["maturity_years"].tolist() == [0.25, 1.0]
    assert curves["yield"].tolist() == [4.0, 4.25]


def test_normalize_ecb_yield_curve_from_sdmx_long_csv() -> None:
    raw = pd.DataFrame(
        {
            "TIME_PERIOD": ["2024-01-02", "2024-01-02", "2024-01-02"],
            "DATA_TYPE_FM": ["SR_3M", "SR_10Y", "IF_10Y"],
            "OBS_VALUE": [3.0, 3.25, 3.5],
        }
    )

    curves = normalize_ecb_yield_curve(raw)

    assert tuple(curves.columns) == CURVE_COLUMNS
    assert set(curves["country"]) == {"EA"}
    assert set(curves["source"]) == {"ecb_yield_curve"}
    assert curves["maturity_years"].tolist() == [0.25, 10.0]
    assert curves["yield"].tolist() == [3.0, 3.25]


def test_normalizers_reject_missing_maturity_columns() -> None:
    raw = pd.DataFrame({"Date": ["2024-01-02"], "value": [1.0]})

    with pytest.raises(ValueError, match="SVENY maturity columns"):
        normalize_fed_gsw(raw)
    with pytest.raises(ValueError, match="ZC maturity columns"):
        normalize_bank_of_canada(raw)
    with pytest.raises(ValueError, match="SR_\\* spot-rate"):
        normalize_ecb_yield_curve(raw)
