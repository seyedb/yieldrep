from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class SourceConfig(BaseModel):
    country: str
    source: str
    raw_file: Path
    url: str | None = None


class PCAConfig(BaseModel):
    n_components: int = 5
    min_maturities: int = 3


class NelsonSiegelConfig(BaseModel):
    tau: float = 1.5
    min_maturities: int = 3


class TargetConfig(BaseModel):
    horizons_days: list[int] = Field(default_factory=lambda: [1, 5, 20])


class EvaluationConfig(BaseModel):
    method: Literal["date_ordered", "walk_forward"] = "date_ordered"
    test_fraction: float = 0.2
    min_train_dates: int = 252
    test_window_dates: int = 63
    step_dates: int = 63
    ridge_alpha: float = 1.0
    lag_days: list[int] = Field(default_factory=lambda: [1, 5, 20])


class PlotConfig(BaseModel):
    selected_maturities: list[float] = Field(
        default_factory=lambda: [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
    )


class ProjectConfig(BaseModel):
    data_dir: Path
    reports_dir: Path
    sources: dict[str, SourceConfig]
    pca: PCAConfig = Field(default_factory=PCAConfig)
    nelson_siegel: NelsonSiegelConfig = Field(default_factory=NelsonSiegelConfig)
    targets: TargetConfig = Field(default_factory=TargetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    plots: PlotConfig = Field(default_factory=PlotConfig)

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
    def tables_dir(self) -> Path:
        return self.reports_dir / "tables"

    @property
    def curves_path(self) -> Path:
        return self.processed_dir / "curves.parquet"

    @property
    def pca_dir(self) -> Path:
        return self.processed_dir / "pca"

    @property
    def nelson_siegel_dir(self) -> Path:
        return self.processed_dir / "nelson_siegel"

    @property
    def targets_path(self) -> Path:
        return self.processed_dir / "targets.parquet"

    @property
    def residual_targets_path(self) -> Path:
        return self.processed_dir / "residual_targets.parquet"

    @property
    def curve_features_path(self) -> Path:
        return self.processed_dir / "curve_features.parquet"

    @property
    def modeling_dir(self) -> Path:
        return self.processed_dir / "modeling"

    @property
    def evaluation_dir(self) -> Path:
        return self.processed_dir / "evaluation"

    @property
    def baseline_metrics_path(self) -> Path:
        return self.evaluation_dir / "baseline_metrics.parquet"

    @property
    def baseline_metrics_by_maturity_path(self) -> Path:
        return self.evaluation_dir / "baseline_metrics_by_maturity.parquet"

    @property
    def baseline_metrics_by_maturity_point_path(self) -> Path:
        return self.evaluation_dir / "baseline_metrics_by_maturity_point.parquet"

    @property
    def baseline_summary_table_path(self) -> Path:
        return self.tables_dir / "baseline_summary.csv"

    @property
    def baseline_by_maturity_bucket_table_path(self) -> Path:
        return self.tables_dir / "baseline_by_maturity_bucket.csv"

    @property
    def baseline_by_maturity_point_top_table_path(self) -> Path:
        return self.tables_dir / "baseline_by_maturity_point_top.csv"


def load_config(path: Path) -> ProjectConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload: Any = yaml.safe_load(handle)
    return ProjectConfig.model_validate(payload)
