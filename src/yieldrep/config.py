from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

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


class AutoencoderConfig(BaseModel):
    latent_dim: int = 5
    hidden_dim: int = 128
    depth: int = 2
    dropout: float = 0.0
    epochs: int = 2000
    batch_size: int = 1024
    learning_rate: float = 0.003
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    mask_probability: float = 0.15
    clean_loss_weight: float = 1.0
    early_stopping_patience: int = 150
    min_delta: float = 1e-5
    random_seed: int = 42
    min_train_dates: int = 252


class TransformerConfig(BaseModel):
    latent_dim: int = 5
    model_dim: int = 24
    n_heads: int = 4
    n_layers: int = 1
    feedforward_dim: int = 48
    dropout: float = 0.05
    epochs: int = 40
    batch_size: int = 1024
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    mask_probability: float = 0.15
    clean_loss_weight: float = 1.0
    early_stopping_patience: int = 8
    min_delta: float = 1e-5
    random_seed: int = 42
    min_train_dates: int = 252
    max_train_dates: int | None = 500


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


class GraphConfig(BaseModel):
    realized_vol_window: int = 20
    correlation_min_observations: int = 252
    correlation_top_k: int = 3


class GNNConfig(BaseModel):
    edge_mode: Literal["adjacent", "adjacent_correlation"] = "adjacent"
    latent_dim: int = 5
    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.05
    epochs: int = 80
    batch_size: int = 1024
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    mask_probability: float = 0.15
    clean_loss_weight: float = 1.0
    early_stopping_patience: int = 10
    min_delta: float = 1e-5
    random_seed: int = 42
    min_train_dates: int = 252
    max_train_dates: int | None = 1500


class ProjectConfig(BaseModel):
    data_dir: Path
    reports_dir: Path
    sources: dict[str, SourceConfig]
    policy_rates: dict[str, SourceConfig] = Field(default_factory=dict)
    market_indicators: dict[str, SourceConfig] = Field(default_factory=dict)
    macro_indicators: dict[str, SourceConfig] = Field(default_factory=dict)
    pca: PCAConfig = Field(default_factory=PCAConfig)
    nelson_siegel: NelsonSiegelConfig = Field(default_factory=NelsonSiegelConfig)
    autoencoder: AutoencoderConfig = Field(default_factory=AutoencoderConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)
    targets: TargetConfig = Field(default_factory=TargetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    plots: PlotConfig = Field(default_factory=PlotConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    gnn: GNNConfig = Field(default_factory=GNNConfig)

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
    def policy_rates_path(self) -> Path:
        return self.processed_dir / "policy_rates.parquet"

    @property
    def policy_features_path(self) -> Path:
        return self.processed_dir / "policy_features.parquet"

    @property
    def market_indicators_path(self) -> Path:
        return self.processed_dir / "market_indicators.parquet"

    @property
    def market_regimes_path(self) -> Path:
        return self.processed_dir / "market_regimes.parquet"

    @property
    def macro_indicators_path(self) -> Path:
        return self.processed_dir / "macro_indicators.parquet"

    @property
    def macro_regimes_path(self) -> Path:
        return self.processed_dir / "macro_regimes.parquet"

    @property
    def pca_dir(self) -> Path:
        return self.processed_dir / "pca"

    @property
    def nelson_siegel_dir(self) -> Path:
        return self.processed_dir / "nelson_siegel"

    @property
    def autoencoder_dir(self) -> Path:
        return self.processed_dir / "autoencoder"

    @property
    def transformer_dir(self) -> Path:
        return self.processed_dir / "transformer"

    @property
    def gnn_dir(self) -> Path:
        return self.processed_dir / "gnn"

    @property
    def learned_states_dir(self) -> Path:
        return self.processed_dir / "learned_states"

    @property
    def graph_dir(self) -> Path:
        return self.processed_dir / "graph"

    @property
    def graph_nodes_path(self) -> Path:
        return self.graph_dir / "maturity_graph_nodes.parquet"

    @property
    def graph_edges_path(self) -> Path:
        return self.graph_dir / "maturity_graph_edges.parquet"

    @property
    def learned_state_regimes_path(self) -> Path:
        return self.learned_states_dir / "regime_states.parquet"

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
    def curve_vol_regime_targets_path(self) -> Path:
        return self.processed_dir / "curve_vol_regime_targets.parquet"

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
    def baseline_classification_coefficients_path(self) -> Path:
        return self.evaluation_dir / "baseline_classification_coefficients.parquet"

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
    def residual_relative_value_benchmark_table_path(self) -> Path:
        return self.tables_dir / "residual_relative_value_benchmark.csv"

    @property
    def residual_relative_value_overview_table_path(self) -> Path:
        return self.tables_dir / "residual_relative_value_overview.csv"

    @property
    def residual_relative_value_scorecard_table_path(self) -> Path:
        return self.tables_dir / "residual_relative_value_scorecard.csv"

    @property
    def residual_rv_by_market_regime_table_path(self) -> Path:
        return self.tables_dir / "residual_rv_by_market_regime.csv"

    @property
    def market_regime_rv_summary_table_path(self) -> Path:
        return self.tables_dir / "market_regime_rv_summary.csv"

    @property
    def residual_rv_by_macro_regime_table_path(self) -> Path:
        return self.tables_dir / "residual_rv_by_macro_regime.csv"

    @property
    def residual_rv_regime_scorecard_table_path(self) -> Path:
        return self.tables_dir / "residual_rv_regime_scorecard.csv"

    @property
    def residual_mean_reversion_table_path(self) -> Path:
        return self.tables_dir / "residual_mean_reversion.csv"

    @property
    def residual_zscores_figure_path(self) -> Path:
        return self.figures_dir / "residual_zscores.html"

    @property
    def residual_rv_regime_heatmap_figure_path(self) -> Path:
        return self.figures_dir / "residual_rv_regime_heatmap.html"

    @property
    def learned_state_regime_summary_table_path(self) -> Path:
        return self.tables_dir / "learned_state_regime_summary.csv"

    @property
    def learned_state_regime_means_table_path(self) -> Path:
        return self.tables_dir / "learned_state_regime_means.csv"

    @property
    def learned_state_regime_heatmap_figure_path(self) -> Path:
        return self.figures_dir / "learned_state_regime_heatmap.html"

    @property
    def learned_state_space_figure_path(self) -> Path:
        return self.figures_dir / "learned_state_space_regimes.html"

    @property
    def representation_comparison_table_path(self) -> Path:
        return self.tables_dir / "representation_comparison.csv"

    @property
    def macro_conditioned_representation_summary_table_path(self) -> Path:
        return self.tables_dir / "macro_conditioned_representation_summary.csv"

    @property
    def learned_model_comparison_table_path(self) -> Path:
        return self.tables_dir / "learned_model_comparison.csv"

    @property
    def learned_reconstruction_leaderboard_table_path(self) -> Path:
        return self.tables_dir / "learned_reconstruction_leaderboard.csv"

    @property
    def learned_model_findings_table_path(self) -> Path:
        return self.tables_dir / "learned_model_findings.csv"

    @property
    def gnn_edge_ablation_table_path(self) -> Path:
        return self.tables_dir / "gnn_edge_ablation.csv"

    @property
    def representation_task_scorecard_table_path(self) -> Path:
        return self.tables_dir / "representation_task_scorecard.csv"

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
    def benchmark_conclusions_table_path(self) -> Path:
        return self.tables_dir / "benchmark_conclusions.csv"

    @property
    def scenario_method_table_path(self) -> Path:
        return self.tables_dir / "scenario_method_comparison.csv"

    @property
    def baseline_audit_table_path(self) -> Path:
        return self.tables_dir / "baseline_audit.csv"

    @property
    def volatility_regime_table_path(self) -> Path:
        return self.tables_dir / "volatility_regime.csv"

    @property
    def volatility_regime_benchmark_table_path(self) -> Path:
        return self.tables_dir / "volatility_regime_benchmark.csv"

    @property
    def ae_classical_factor_correlations_table_path(self) -> Path:
        return self.tables_dir / "ae_classical_factor_correlations.csv"

    @property
    def cross_market_summary_table_path(self) -> Path:
        return self.tables_dir / "cross_market_summary.csv"

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

    @property
    def reconstruction_oos_by_maturity_bucket_table_path(self) -> Path:
        return self.tables_dir / "reconstruction_oos_by_maturity_bucket.csv"

    @property
    def reconstruction_oos_comparison_table_path(self) -> Path:
        return self.tables_dir / "reconstruction_oos_comparison.csv"

    @property
    def masked_reconstruction_by_maturity_table_path(self) -> Path:
        return self.tables_dir / "masked_reconstruction_by_maturity.csv"

    @property
    def masked_reconstruction_by_maturity_bucket_table_path(self) -> Path:
        return self.tables_dir / "masked_reconstruction_by_maturity_bucket.csv"

    @property
    def masked_reconstruction_hardest_maturities_table_path(self) -> Path:
        return self.tables_dir / "masked_reconstruction_hardest_maturities.csv"


def load_config(path: Path) -> ProjectConfig:
    payload = _load_config_payload(path)
    return ProjectConfig.model_validate(payload)


def _load_config_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = cast(dict[str, Any], yaml.safe_load(handle))

    base_path = payload.pop("extends", None)
    if base_path is None:
        return payload

    inherited = _load_config_payload(path.parent / Path(str(base_path)))
    return _deep_merge(inherited, payload)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged
