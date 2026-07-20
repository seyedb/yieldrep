from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

from yieldrep.config import ProjectConfig, SourceConfig

_USER_AGENT = "yieldrep/0.1"


def ingest_sources(config: ProjectConfig, overwrite: bool = False) -> list[Path]:
    """Download configured source files into local raw data paths."""
    return _ingest_configured_sources(config.sources, overwrite=overwrite)


def ingest_policy_rates(config: ProjectConfig, overwrite: bool = False) -> list[Path]:
    """Download configured policy-rate source files into local raw data paths."""
    return _ingest_configured_sources(config.policy_rates, overwrite=overwrite)


def _ingest_configured_sources(
    sources: dict[str, SourceConfig],
    overwrite: bool,
) -> list[Path]:
    raw_paths: list[Path] = []
    for name, source_config in sources.items():
        raw_paths.append(_ingest_source(name, source_config, overwrite=overwrite))
    return raw_paths


def _ingest_source(name: str, source_config: SourceConfig, overwrite: bool) -> Path:
    if source_config.url is None:
        raise ValueError(f"Source has no URL configured: {name}")

    source_config.raw_file.parent.mkdir(parents=True, exist_ok=True)
    if source_config.raw_file.exists() and not overwrite:
        return source_config.raw_file

    request = Request(source_config.url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=60) as response:
        source_config.raw_file.write_bytes(response.read())

    return source_config.raw_file
