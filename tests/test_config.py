from pathlib import Path

from yieldrep.config import load_config


def test_load_config_reads_project_paths() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert config.data_dir == Path("data")
    assert config.raw_dir == Path("data/raw")
    assert config.processed_dir == Path("data/processed")
    assert config.curves_path == Path("data/processed/curves.parquet")
    assert config.pca_dir == Path("data/processed/pca")
    assert config.nelson_siegel_dir == Path("data/processed/nelson_siegel")
    assert config.targets_path == Path("data/processed/targets.parquet")
    assert config.standardized_targets_path == Path("data/processed/standardized_targets.parquet")
    assert config.residual_targets_path == Path("data/processed/residual_targets.parquet")
    assert config.vol_targets_path == Path("data/processed/vol_targets.parquet")
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
    assert config.overlap_sensitivity_table_path == Path("reports/tables/overlap_sensitivity.csv")
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


def test_load_config_reads_source_metadata() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert set(config.sources) == {"fed_gsw", "bank_of_canada"}
    assert config.sources["fed_gsw"].country == "US"
    assert config.sources["bank_of_canada"].raw_file == Path("data/raw/boc_zero_coupon.csv")
    assert config.sources["fed_gsw"].url is not None
    assert config.sources["bank_of_canada"].url is not None
    assert config.pca.n_components == 5
    assert config.pca.min_maturities == 3
    assert config.nelson_siegel.tau == 1.5
    assert config.nelson_siegel.min_maturities == 3
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
