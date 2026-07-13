from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class SourceConfig(BaseModel):
    country: str
    source: str
    raw_file: Path
    url: str | None = None


class ProjectConfig(BaseModel):
    data_dir: Path
    reports_dir: Path
    sources: dict[str, SourceConfig]

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def curves_path(self) -> Path:
        return self.processed_dir / "curves.parquet"


def load_config(path: Path) -> ProjectConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload: Any = yaml.safe_load(handle)
    return ProjectConfig.model_validate(payload)
