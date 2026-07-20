from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
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
    realized_vol_window: int = 20


class EvaluationConfig(BaseModel):
    method: Literal["date_ordered", "walk_forward"] = "date_ordered"
    test_fraction: float = 0.2
    min_train_dates: int = 252
    test_window_dates: int = 63
    step_dates: int = 63
    walk_forward_max_windows: int = 4
    ridge_alpha: float = 1.0
    elastic_net_alpha: float = 0.01
    elastic_net_l1_ratio: float = 0.5
    logistic_c: float = 1.0
    classification_max_train_rows: int = 2_000
    non_overlapping_targets: bool = True
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
    def standardized_targets_path(self) -> Path:
        return self.processed_dir / "standardized_targets.parquet"

    @property
    def residual_targets_path(self) -> Path:
        return self.processed_dir / "residual_targets.parquet"

    @property
    def vol_targets_path(self) -> Path:
        return self.processed_dir / "vol_targets.parquet"

    @property
    def residual_features_path(self) -> Path:
        return self.processed_dir / "residual_features.parquet"

    @property
    def curve_features_path(self) -> Path:
        return self.processed_dir / "curve_features.parquet"

    @property
    def carry_roll_features_path(self) -> Path:
        return self.processed_dir / "carry_roll_features.parquet"

    @property
    def modeling_dir(self) -> Path:
        return self.processed_dir / "modeling"

    @property
    def supervised_yield_change_path(self) -> Path:
        return self.modeling_dir / "supervised_yield_change.parquet"

    @property
    def supervised_residual_change_path(self) -> Path:
        return self.modeling_dir / "supervised_residual_change.parquet"

    @property
    def supervised_vol_change_path(self) -> Path:
        return self.modeling_dir / "supervised_vol_change.parquet"

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
    def baseline_classification_metrics_path(self) -> Path:
        return self.evaluation_dir / "baseline_classification_metrics.parquet"

    @property
    def baseline_residual_rv_spread_path(self) -> Path:
        return self.evaluation_dir / "residual_rv_spread.parquet"

    @property
    def supervised_forecast_metrics_path(self) -> Path:
        return self.evaluation_dir / "supervised_forecast_metrics.parquet"

    @property
    def supervised_forecast_by_maturity_bucket_path(self) -> Path:
        return self.evaluation_dir / "supervised_forecast_by_maturity_bucket.parquet"

    @property
    def supervised_forecast_coefficients_path(self) -> Path:
        return self.evaluation_dir / "supervised_forecast_coefficients.parquet"

    @property
    def supervised_forecast_summary_table_path(self) -> Path:
        return self.tables_dir / "supervised_forecast_summary.csv"

    @property
    def supervised_forecast_rank_table_path(self) -> Path:
        return self.tables_dir / "supervised_forecast_rank.csv"

    @property
    def supervised_forecast_by_maturity_bucket_table_path(self) -> Path:
        return self.tables_dir / "supervised_forecast_by_maturity_bucket.csv"

    @property
    def supervised_forecast_coefficients_table_path(self) -> Path:
        return self.tables_dir / "supervised_forecast_coefficients.csv"

    @property
    def lagged_diagnostics_path(self) -> Path:
        return self.evaluation_dir / "lagged_diagnostics.parquet"

    @property
    def lagged_diagnostics_table_path(self) -> Path:
        return self.tables_dir / "lagged_diagnostics.csv"

    @property
    def baseline_summary_table_path(self) -> Path:
        return self.tables_dir / "baseline_summary.csv"

    @property
    def baseline_by_maturity_bucket_table_path(self) -> Path:
        return self.tables_dir / "baseline_by_maturity_bucket.csv"

    @property
    def residual_relative_value_table_path(self) -> Path:
        return self.tables_dir / "residual_relative_value.csv"

    @property
    def residual_relative_value_rank_ic_table_path(self) -> Path:
        return self.tables_dir / "residual_relative_value_rank_ic.csv"

    @property
    def residual_relative_value_rank_ic_coverage_table_path(self) -> Path:
        return self.tables_dir / "residual_relative_value_rank_ic_coverage.csv"

    @property
    def residual_relative_value_spread_table_path(self) -> Path:
        return self.tables_dir / "residual_relative_value_spread.csv"

    @property
    def baseline_by_maturity_point_top_table_path(self) -> Path:
        return self.tables_dir / "baseline_by_maturity_point_top.csv"

    @property
    def baseline_rank_table_path(self) -> Path:
        return self.tables_dir / "baseline_rank.csv"

    @property
    def baseline_winners_table_path(self) -> Path:
        return self.tables_dir / "baseline_winners.csv"

    @property
    def overlap_sensitivity_table_path(self) -> Path:
        return self.tables_dir / "overlap_sensitivity.csv"

    @property
    def supervised_walk_forward_summary_table_path(self) -> Path:
        return self.tables_dir / "supervised_walk_forward_summary.csv"

    @property
    def supervised_walk_forward_rank_table_path(self) -> Path:
        return self.tables_dir / "supervised_walk_forward_rank.csv"

    @property
    def supervised_walk_forward_comparison_table_path(self) -> Path:
        return self.tables_dir / "supervised_walk_forward_comparison.csv"

    @property
    def reconstruction_summary_table_path(self) -> Path:
        return self.tables_dir / "reconstruction_summary.csv"

    @property
    def reconstruction_by_maturity_table_path(self) -> Path:
        return self.tables_dir / "reconstruction_by_maturity.csv"

    @property
    def reconstruction_worst_maturities_table_path(self) -> Path:
        return self.tables_dir / "reconstruction_worst_maturities.csv"

    @property
    def reconstruction_oos_summary_table_path(self) -> Path:
        return self.tables_dir / "reconstruction_oos_summary.csv"

    @property
    def reconstruction_oos_by_maturity_table_path(self) -> Path:
        return self.tables_dir / "reconstruction_oos_by_maturity.csv"


def load_config(path: Path) -> ProjectConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload: Any = yaml.safe_load(handle)
    return ProjectConfig.model_validate(payload)
