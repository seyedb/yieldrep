from pathlib import Path

import pandas as pd
import pytest

from yieldrep.data.sources.bank_of_canada import (
    bank_of_canada_date_column,
    bank_of_canada_maturity_columns,
    load_bank_of_canada_raw,
)
from yieldrep.data.sources.fed_gsw import (
    fed_gsw_date_column,
    fed_gsw_maturity_columns,
    load_fed_gsw_raw,
)


def test_load_fed_gsw_raw_skips_preamble(tmp_path: Path) -> None:
    raw_path = tmp_path / "fed_gsw.csv"
    raw_path.write_text(
        "\n".join(
            [
                "Note: research data",
                "",
                "Series,Compounding Convention,Mnemonic(s)",
                "Date,BETA0,SVENY01,SVENY10",
                "2024-01-02,1.0,4.10,4.25",
            ]
        ),
        encoding="utf-8",
    )

    frame = load_fed_gsw_raw(raw_path)

    assert fed_gsw_date_column(frame) == "Date"
    assert fed_gsw_maturity_columns(frame) == {"SVENY01": 1.0, "SVENY10": 10.0}
    assert frame.loc[0, "SVENY10"] == 4.25


def test_load_bank_of_canada_raw_strips_column_whitespace(tmp_path: Path) -> None:
    raw_path = tmp_path / "boc.csv"
    raw_path.write_text(
        "\n".join(
            [
                "Date, ZC025YR, ZC100YR,",
                "2024-01-02, 0.0400, 0.0425,",
            ]
        ),
        encoding="utf-8",
    )

    frame = load_bank_of_canada_raw(raw_path)

    assert bank_of_canada_date_column(frame) == "Date"
    assert bank_of_canada_maturity_columns(frame) == {"ZC025YR": 0.25, "ZC100YR": 1.0}
    assert "Unnamed: 3" not in frame.columns


def test_source_date_helpers_reject_missing_date_column() -> None:
    frame = pd.DataFrame({"value": [1.0]})

    with pytest.raises(ValueError, match="Date column"):
        fed_gsw_date_column(frame)
    with pytest.raises(ValueError, match="Date column"):
        bank_of_canada_date_column(frame)
