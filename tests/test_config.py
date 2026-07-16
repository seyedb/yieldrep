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
    assert config.modeling_dir == Path("data/processed/modeling")
    assert config.evaluation_dir == Path("data/processed/evaluation")
    assert config.baseline_metrics_path == Path("data/processed/evaluation/baseline_metrics.parquet")
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
    assert config.evaluation.test_fraction == 0.2
    assert config.evaluation.ridge_alpha == 1.0
    assert config.plots.selected_maturities == [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
