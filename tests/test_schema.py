import pandas as pd
import pytest

from yieldrep.data.schema import CURVE_COLUMNS, validate_curve_frame


def test_validate_curve_frame_returns_ordered_schema() -> None:
    frame = pd.DataFrame(
        {
            "source": ["test"],
            "yield": ["4.25"],
            "maturity_years": ["10"],
            "country": ["US"],
            "date": ["2024-01-02"],
            "extra": ["ignored"],
        }
    )

    result = validate_curve_frame(frame)

    assert tuple(result.columns) == CURVE_COLUMNS
    assert result.loc[0, "date"] == pd.Timestamp("2024-01-02")
    assert result.loc[0, "maturity_years"] == 10.0
    assert result.loc[0, "yield"] == 4.25


def test_validate_curve_frame_requires_schema_columns() -> None:
    frame = pd.DataFrame({"date": ["2024-01-02"]})

    with pytest.raises(ValueError, match="Missing required curve columns"):
        validate_curve_frame(frame)


def test_validate_curve_frame_rejects_null_values() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "country": ["US"],
            "maturity_years": [10.0],
            "yield": [None],
            "source": ["test"],
        }
    )

    with pytest.raises(ValueError, match="null values"):
        validate_curve_frame(frame)


def test_validate_curve_frame_rejects_non_positive_maturities() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "country": ["US"],
            "maturity_years": [0.0],
            "yield": [4.25],
            "source": ["test"],
        }
    )

    with pytest.raises(ValueError, match="positive"):
        validate_curve_frame(frame)
