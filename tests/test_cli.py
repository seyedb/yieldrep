from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from yieldrep.cli import app


def test_ingest_command_downloads_raw_file(tmp_path: Path) -> None:
    source_file = tmp_path / "source.csv"
    source_file.write_text("Date,Value\n2024-01-02,1.0\n", encoding="utf-8")
    raw_file = tmp_path / "data" / "raw" / "source.csv"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_dir: {tmp_path / 'data'}",
                f"reports_dir: {tmp_path / 'reports'}",
                "sources:",
                "  test:",
                "    country: US",
                "    source: test",
                f"    raw_file: {raw_file}",
                f"    url: {source_file.as_uri()}",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["ingest", "--config", str(config_path)])

    assert result.exit_code == 0
    assert raw_file.read_text(encoding="utf-8") == source_file.read_text(encoding="utf-8")
    assert str(raw_file) in result.stdout


def test_normalize_command_writes_curves_parquet(tmp_path: Path) -> None:
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

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_dir: {tmp_path / 'data'}",
                f"reports_dir: {tmp_path / 'reports'}",
                "sources:",
                "  fed_gsw:",
                "    country: US",
                "    source: fed_gsw",
                f"    raw_file: {fed_raw}",
                "  bank_of_canada:",
                "    country: CA",
                "    source: bank_of_canada",
                f"    raw_file: {boc_raw}",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["normalize", "--config", str(config_path)])

    assert result.exit_code == 0
    assert (tmp_path / "data" / "processed" / "curves.parquet").exists()
    assert "curves.parquet" in result.stdout


def test_build_pca_command_writes_outputs(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    dates = pd.date_range("2024-01-01", periods=4)
    curves = pd.DataFrame(
        [
            {
                "date": date,
                "country": "US",
                "maturity_years": maturity,
                "yield": 3.0 + date_index * 0.1 + maturity * 0.02,
                "source": "test",
            }
            for date_index, date in enumerate(dates)
            for maturity in [1.0, 2.0, 10.0]
        ]
    )
    curves.to_parquet(processed_dir / "curves.parquet", index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_dir: {tmp_path / 'data'}",
                f"reports_dir: {tmp_path / 'reports'}",
                "pca:",
                "  n_components: 2",
                "  min_maturities: 3",
                "sources:",
                "  test:",
                "    country: US",
                "    source: test",
                f"    raw_file: {tmp_path / 'raw.csv'}",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["build-pca", "--config", str(config_path)])

    assert result.exit_code == 0
    assert (processed_dir / "pca" / "us_scores.parquet").exists()
    assert (processed_dir / "pca" / "us_loadings.parquet").exists()
    assert (processed_dir / "pca" / "us_variance.parquet").exists()
    assert "us_scores.parquet" in result.stdout
