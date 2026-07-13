from pathlib import Path

from typer.testing import CliRunner

from yieldrep.cli import app


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
