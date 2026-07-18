from pathlib import Path

from yieldrep.config import load_config


def test_load_config_reads_project_paths() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert config.data_dir == Path("data")
    assert config.raw_dir == Path("data/raw")
    assert config.processed_dir == Path("data/processed")
    assert config.curves_path == Path("data/processed/curves.parquet")
    assert config.tables_dir == Path("reports/tables")
    assert config.pca_dir == Path("data/processed/pca")
    assert config.nelson_siegel_dir == Path("data/processed/nelson_siegel")
    assert config.targets_path == Path("data/processed/targets.parquet")
    assert config.residual_targets_path == Path("data/processed/residual_targets.parquet")
    assert config.curve_features_path == Path("data/processed/curve_features.parquet")
    assert config.modeling_dir == Path("data/processed/modeling")
    assert config.evaluation_dir == Path("data/processed/evaluation")
    assert config.baseline_metrics_path == Path("data/processed/evaluation/baseline_metrics.parquet")
    assert config.baseline_metrics_by_maturity_path == Path(
        "data/processed/evaluation/baseline_metrics_by_maturity.parquet"
    )
    assert config.baseline_metrics_by_maturity_point_path == Path(
        "data/processed/evaluation/baseline_metrics_by_maturity_point.parquet"
    )
    assert config.baseline_summary_table_path == Path("reports/tables/baseline_summary.csv")
    assert config.baseline_by_maturity_bucket_table_path == Path(
        "reports/tables/baseline_by_maturity_bucket.csv"
    )
    assert config.baseline_by_maturity_point_top_table_path == Path(
        "reports/tables/baseline_by_maturity_point_top.csv"
    )
    assert config.figures_dir == Path("reports/figures")


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
    assert config.evaluation.method == "date_ordered"
    assert config.evaluation.test_fraction == 0.2
    assert config.evaluation.min_train_dates == 252
    assert config.evaluation.test_window_dates == 63
    assert config.evaluation.step_dates == 63
    assert config.evaluation.ridge_alpha == 1.0
    assert config.evaluation.lag_days == [1, 5, 20]
    assert config.plots.selected_maturities == [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
