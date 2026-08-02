from pathlib import Path

from yieldrep.config import load_config


def test_load_config_reads_project_paths() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert config.data_dir == Path("data")
    assert config.raw_dir == Path("data/raw")
    assert config.processed_dir == Path("data/processed")
    assert config.curves_path == Path("data/processed/curves.parquet")
    assert config.policy_rates_path == Path("data/processed/policy_rates.parquet")
    assert config.policy_features_path == Path("data/processed/policy_features.parquet")
    assert config.market_indicators_path == Path("data/processed/market_indicators.parquet")
    assert config.market_regimes_path == Path("data/processed/market_regimes.parquet")
    assert config.macro_indicators_path == Path("data/processed/macro_indicators.parquet")
    assert config.macro_regimes_path == Path("data/processed/macro_regimes.parquet")
    assert config.pca_dir == Path("data/processed/pca")
    assert config.nelson_siegel_dir == Path("data/processed/nelson_siegel")
    assert config.autoencoder_dir == Path("data/processed/autoencoder")
    assert config.transformer_dir == Path("data/processed/transformer")
    assert config.learned_states_dir == Path("data/processed/learned_states")
    assert config.learned_state_regimes_path == Path(
        "data/processed/learned_states/regime_states.parquet"
    )
    assert config.targets_path == Path("data/processed/targets.parquet")
    assert config.standardized_targets_path == Path("data/processed/standardized_targets.parquet")
    assert config.residual_targets_path == Path("data/processed/residual_targets.parquet")
    assert config.vol_targets_path == Path("data/processed/vol_targets.parquet")
    assert config.curve_vol_regime_targets_path == Path(
        "data/processed/curve_vol_regime_targets.parquet"
    )
    assert config.residual_features_path == Path("data/processed/residual_features.parquet")
    assert config.carry_roll_features_path == Path("data/processed/carry_roll_features.parquet")
    assert config.modeling_dir == Path("data/processed/modeling")
    assert config.supervised_yield_change_path == Path(
        "data/processed/modeling/supervised_yield_change.parquet"
    )
    assert config.supervised_residual_change_path == Path(
        "data/processed/modeling/supervised_residual_change.parquet"
    )
    assert config.supervised_vol_change_path == Path(
        "data/processed/modeling/supervised_vol_change.parquet"
    )
    assert config.evaluation_dir == Path("data/processed/evaluation")
    assert config.baseline_metrics_path == Path("data/processed/evaluation/baseline_metrics.parquet")
    assert config.baseline_classification_metrics_path == Path(
        "data/processed/evaluation/baseline_classification_metrics.parquet"
    )
    assert config.baseline_classification_coefficients_path == Path(
        "data/processed/evaluation/baseline_classification_coefficients.parquet"
    )
    assert config.volatility_regime_table_path == Path("reports/tables/volatility_regime.csv")
    assert config.volatility_regime_benchmark_table_path == Path(
        "reports/tables/volatility_regime_benchmark.csv"
    )
    assert config.ae_classical_factor_correlations_table_path == Path(
        "reports/tables/ae_classical_factor_correlations.csv"
    )
    assert config.cross_market_summary_table_path == Path("reports/tables/cross_market_summary.csv")
    assert config.baseline_residual_rv_spread_path == Path(
        "data/processed/evaluation/residual_rv_spread.parquet"
    )
    assert config.supervised_forecast_metrics_path == Path(
        "data/processed/evaluation/supervised_forecast_metrics.parquet"
    )
    assert config.supervised_forecast_by_maturity_bucket_path == Path(
        "data/processed/evaluation/supervised_forecast_by_maturity_bucket.parquet"
    )
    assert config.supervised_forecast_coefficients_path == Path(
        "data/processed/evaluation/supervised_forecast_coefficients.parquet"
    )
    assert config.supervised_forecast_summary_table_path == Path(
        "reports/tables/supervised_forecast_summary.csv"
    )
    assert config.supervised_forecast_rank_table_path == Path(
        "reports/tables/supervised_forecast_rank.csv"
    )
    assert config.supervised_forecast_by_maturity_bucket_table_path == Path(
        "reports/tables/supervised_forecast_by_maturity_bucket.csv"
    )
    assert config.supervised_forecast_coefficients_table_path == Path(
        "reports/tables/supervised_forecast_coefficients.csv"
    )
    assert config.lagged_diagnostics_path == Path(
        "data/processed/evaluation/lagged_diagnostics.parquet"
    )
    assert config.lagged_diagnostics_table_path == Path("reports/tables/lagged_diagnostics.csv")
    assert config.figures_dir == Path("reports/figures")
    assert config.tables_dir == Path("reports/tables")
    assert config.baseline_summary_table_path == Path("reports/tables/baseline_summary.csv")
    assert config.baseline_rank_table_path == Path("reports/tables/baseline_rank.csv")
    assert config.baseline_winners_table_path == Path("reports/tables/baseline_winners.csv")
    assert config.residual_relative_value_table_path == Path(
        "reports/tables/residual_relative_value.csv"
    )
    assert config.residual_relative_value_rank_ic_table_path == Path(
        "reports/tables/residual_relative_value_rank_ic.csv"
    )
    assert config.residual_relative_value_rank_ic_coverage_table_path == Path(
        "reports/tables/residual_relative_value_rank_ic_coverage.csv"
    )
    assert config.residual_relative_value_spread_table_path == Path(
        "reports/tables/residual_relative_value_spread.csv"
    )
    assert config.residual_relative_value_benchmark_table_path == Path(
        "reports/tables/residual_relative_value_benchmark.csv"
    )
    assert config.residual_relative_value_overview_table_path == Path(
        "reports/tables/residual_relative_value_overview.csv"
    )
    assert config.residual_relative_value_scorecard_table_path == Path(
        "reports/tables/residual_relative_value_scorecard.csv"
    )
    assert config.residual_rv_by_market_regime_table_path == Path(
        "reports/tables/residual_rv_by_market_regime.csv"
    )
    assert config.market_regime_rv_summary_table_path == Path(
        "reports/tables/market_regime_rv_summary.csv"
    )
    assert config.residual_rv_by_macro_regime_table_path == Path(
        "reports/tables/residual_rv_by_macro_regime.csv"
    )
    assert config.residual_rv_regime_scorecard_table_path == Path(
        "reports/tables/residual_rv_regime_scorecard.csv"
    )
    assert config.residual_mean_reversion_table_path == Path(
        "reports/tables/residual_mean_reversion.csv"
    )
    assert config.residual_zscores_figure_path == Path("reports/figures/residual_zscores.html")
    assert config.residual_rv_regime_heatmap_figure_path == Path(
        "reports/figures/residual_rv_regime_heatmap.html"
    )
    assert config.learned_state_regime_summary_table_path == Path(
        "reports/tables/learned_state_regime_summary.csv"
    )
    assert config.learned_state_regime_means_table_path == Path(
        "reports/tables/learned_state_regime_means.csv"
    )
    assert config.learned_state_regime_heatmap_figure_path == Path(
        "reports/figures/learned_state_regime_heatmap.html"
    )
    assert config.learned_state_space_figure_path == Path(
        "reports/figures/learned_state_space_regimes.html"
    )
    assert config.representation_comparison_table_path == Path(
        "reports/tables/representation_comparison.csv"
    )
    assert config.macro_conditioned_representation_summary_table_path == Path(
        "reports/tables/macro_conditioned_representation_summary.csv"
    )
    assert config.learned_model_comparison_table_path == Path(
        "reports/tables/learned_model_comparison.csv"
    )
    assert config.representation_task_scorecard_table_path == Path(
        "reports/tables/representation_task_scorecard.csv"
    )
    assert config.autoencoder_edge_analysis_table_path == Path(
        "reports/tables/autoencoder_edge_analysis.csv"
    )
    assert config.overlap_sensitivity_table_path == Path("reports/tables/overlap_sensitivity.csv")
    assert config.benchmark_conclusions_table_path == Path(
        "reports/tables/benchmark_conclusions.csv"
    )
    assert config.scenario_method_table_path == Path(
        "reports/tables/scenario_method_comparison.csv"
    )
    assert config.baseline_audit_table_path == Path("reports/tables/baseline_audit.csv")
    assert config.supervised_walk_forward_summary_table_path == Path(
        "reports/tables/supervised_walk_forward_summary.csv"
    )
    assert config.supervised_walk_forward_rank_table_path == Path(
        "reports/tables/supervised_walk_forward_rank.csv"
    )
    assert config.supervised_walk_forward_comparison_table_path == Path(
        "reports/tables/supervised_walk_forward_comparison.csv"
    )
    assert config.reconstruction_summary_table_path == Path(
        "reports/tables/reconstruction_summary.csv"
    )
    assert config.reconstruction_by_maturity_table_path == Path(
        "reports/tables/reconstruction_by_maturity.csv"
    )
    assert config.reconstruction_worst_maturities_table_path == Path(
        "reports/tables/reconstruction_worst_maturities.csv"
    )
    assert config.reconstruction_oos_summary_table_path == Path(
        "reports/tables/reconstruction_oos_summary.csv"
    )
    assert config.reconstruction_oos_by_maturity_table_path == Path(
        "reports/tables/reconstruction_oos_by_maturity.csv"
    )
    assert config.reconstruction_oos_by_maturity_bucket_table_path == Path(
        "reports/tables/reconstruction_oos_by_maturity_bucket.csv"
    )
    assert config.reconstruction_oos_comparison_table_path == Path(
        "reports/tables/reconstruction_oos_comparison.csv"
    )
    assert config.masked_reconstruction_by_maturity_table_path == Path(
        "reports/tables/masked_reconstruction_by_maturity.csv"
    )
    assert config.masked_reconstruction_by_maturity_bucket_table_path == Path(
        "reports/tables/masked_reconstruction_by_maturity_bucket.csv"
    )
    assert config.masked_reconstruction_hardest_maturities_table_path == Path(
        "reports/tables/masked_reconstruction_hardest_maturities.csv"
    )


def test_load_config_reads_source_metadata() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert set(config.sources) == {"fed_gsw", "bank_of_canada", "ecb_yield_curve"}
    assert config.sources["fed_gsw"].country == "US"
    assert config.sources["bank_of_canada"].raw_file == Path("data/raw/boc_zero_coupon.csv")
    assert config.sources["ecb_yield_curve"].country == "EA"
    assert config.sources["ecb_yield_curve"].raw_file == Path("data/raw/ecb_yield_curve.csv")
    assert config.sources["fed_gsw"].url is not None
    assert config.sources["bank_of_canada"].url is not None
    assert config.sources["ecb_yield_curve"].url is not None
    assert set(config.policy_rates) == {
        "fed_funds",
        "boc_target_overnight",
        "ecb_deposit_facility",
    }
    assert config.policy_rates["fed_funds"].country == "US"
    assert config.policy_rates["boc_target_overnight"].country == "CA"
    assert config.policy_rates["ecb_deposit_facility"].country == "EA"
    assert set(config.market_indicators) == {"vix", "move"}
    assert config.market_indicators["vix"].source == "fred_vixcls"
    assert config.market_indicators["move"].source == "yahoo_move"
    assert set(config.macro_indicators) == {
        "us_inflation",
        "us_unemployment",
        "ca_inflation",
        "ca_unemployment",
        "ea_inflation",
    }
    assert config.macro_indicators["us_inflation"].source == "fred_cpi_index_yoy"
    assert config.macro_indicators["ea_inflation"].source == "fred_hicp_index_yoy"
    assert config.pca.n_components == 5
    assert config.pca.min_maturities == 3
    assert config.nelson_siegel.tau == 1.5
    assert config.nelson_siegel.min_maturities == 3
    assert config.autoencoder.latent_dim == 5
    assert config.autoencoder.hidden_dim == 128
    assert config.autoencoder.depth == 2
    assert config.autoencoder.dropout == 0.0
    assert config.autoencoder.epochs == 2000
    assert config.autoencoder.batch_size == 1024
    assert config.autoencoder.weight_decay == 0.00001
    assert config.autoencoder.validation_fraction == 0.2
    assert config.autoencoder.mask_probability == 0.15
    assert config.autoencoder.clean_loss_weight == 1.0
    assert config.autoencoder.early_stopping_patience == 150
    assert config.transformer.latent_dim == 5
    assert config.transformer.model_dim == 24
    assert config.transformer.n_heads == 4
    assert config.transformer.n_layers == 1
    assert config.transformer.max_train_dates == 500
    assert config.targets.horizons_days == [1, 5, 20]
    assert config.targets.realized_vol_window == 20
    assert config.evaluation.method == "date_ordered"
    assert config.evaluation.test_fraction == 0.2
    assert config.evaluation.ridge_alpha == 1.0
    assert config.evaluation.walk_forward_max_windows == 4
    assert config.evaluation.elastic_net_alpha == 0.01
    assert config.evaluation.elastic_net_l1_ratio == 0.5
    assert config.evaluation.logistic_c == 1.0
    assert config.evaluation.classification_max_train_rows == 2_000
    assert config.evaluation.non_overlapping_targets is True
    assert config.evaluation.lag_days == [1, 5, 20]
    assert config.plots.selected_maturities == [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]


def test_load_config_supports_inherited_overrides() -> None:
    config = load_config(Path("configs/learned_heavy.yaml"))

    assert config.sources["fed_gsw"].country == "US"
    assert config.transformer.model_dim == 32
    assert config.transformer.max_train_dates == 1500
