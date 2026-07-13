from pathlib import Path

import pytest

from yieldrep.config import ProjectConfig, SourceConfig
from yieldrep.data.ingest import ingest_sources


def test_ingest_sources_downloads_configured_files(tmp_path: Path) -> None:
    source_file = tmp_path / "source.csv"
    source_file.write_text("Date,Value\n2024-01-02,1.0\n", encoding="utf-8")
    raw_file = tmp_path / "data" / "raw" / "source.csv"
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={
            "test": SourceConfig(
                country="US",
                source="test",
                raw_file=raw_file,
                url=source_file.as_uri(),
            )
        },
    )

    raw_paths = ingest_sources(config)

    assert raw_paths == [raw_file]
    assert raw_file.read_text(encoding="utf-8") == source_file.read_text(encoding="utf-8")


def test_ingest_sources_skips_existing_file_without_overwrite(tmp_path: Path) -> None:
    source_file = tmp_path / "source.csv"
    source_file.write_text("new\n", encoding="utf-8")
    raw_file = tmp_path / "data" / "raw" / "source.csv"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_text("existing\n", encoding="utf-8")
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={
            "test": SourceConfig(
                country="US",
                source="test",
                raw_file=raw_file,
                url=source_file.as_uri(),
            )
        },
    )

    ingest_sources(config)

    assert raw_file.read_text(encoding="utf-8") == "existing\n"


def test_ingest_sources_overwrites_existing_file_when_requested(tmp_path: Path) -> None:
    source_file = tmp_path / "source.csv"
    source_file.write_text("new\n", encoding="utf-8")
    raw_file = tmp_path / "data" / "raw" / "source.csv"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_text("existing\n", encoding="utf-8")
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={
            "test": SourceConfig(
                country="US",
                source="test",
                raw_file=raw_file,
                url=source_file.as_uri(),
            )
        },
    )

    ingest_sources(config, overwrite=True)

    assert raw_file.read_text(encoding="utf-8") == "new\n"


def test_ingest_sources_requires_url(tmp_path: Path) -> None:
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={
            "test": SourceConfig(
                country="US",
                source="test",
                raw_file=tmp_path / "data" / "raw" / "source.csv",
            )
        },
    )

    with pytest.raises(ValueError, match="no URL"):
        ingest_sources(config)
