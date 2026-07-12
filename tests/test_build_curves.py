from pathlib import Path

import pandas as pd
import pytest

from yieldrep.config import ProjectConfig, SourceConfig
from yieldrep.data.normalize import build_curves_parquet
from yieldrep.data.schema import CURVE_COLUMNS


def test_build_curves_parquet_from_configured_raw_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    fed_raw = raw_dir / "fed_gsw.csv"
    fed_raw.write_text(
        "\n".join(
            [
                "Note: research data",
                "Date,BETA0,SVENY01,SVENY10",
                "2024-01-02,1.0,4.00,4.20",
            ]
        ),
        encoding="utf-8",
    )

    boc_raw = raw_dir / "boc_zero_coupon.csv"
    boc_raw.write_text(
        "\n".join(
            [
                "Date, ZC025YR, ZC100YR,",
                "2024-01-02, 0.0400, 0.0425,",
            ]
        ),
        encoding="utf-8",
    )

    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={
            "fed_gsw": SourceConfig(country="US", source="fed_gsw", raw_file=fed_raw),
            "bank_of_canada": SourceConfig(
                country="CA",
                source="bank_of_canada",
                raw_file=boc_raw,
            ),
        },
    )

    output_path = build_curves_parquet(config)
    curves = pd.read_parquet(output_path)

    assert output_path == tmp_path / "data" / "processed" / "curves.parquet"
    assert tuple(curves.columns) == CURVE_COLUMNS
    assert len(curves) == 4
    assert set(curves["country"]) == {"US", "CA"}
    assert set(curves["source"]) == {"fed_gsw", "bank_of_canada"}


def test_build_curves_parquet_rejects_unsupported_source(tmp_path: Path) -> None:
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={
            "unknown": SourceConfig(
                country="XX",
                source="unknown",
                raw_file=tmp_path / "missing.csv",
            )
        },
    )

    with pytest.raises(ValueError, match="Unsupported source"):
        build_curves_parquet(config)
