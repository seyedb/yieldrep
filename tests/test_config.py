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
    assert config.modeling_dir == Path("data/processed/modeling")
    assert config.evaluation_dir == Path("data/processed/evaluation")
    assert config.baseline_metrics_path == Path("data/processed/evaluation/baseline_metrics.parquet")
    assert config.baseline_classification_metrics_path == Path(
        "data/processed/evaluation/baseline_classification_metrics.parquet"
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
    assert config.overlap_sensitivity_table_path == Path("reports/tables/overlap_sensitivity.csv")


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
    assert config.evaluation.logistic_c == 1.0
    assert config.evaluation.classification_max_train_rows == 2_000
    assert config.evaluation.non_overlapping_targets is False
    assert config.evaluation.lag_days == [1, 5, 20]
    assert config.plots.selected_maturities == [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
