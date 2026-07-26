from pathlib import Path

import pandas as pd

from yieldrep.config import PCAConfig, ProjectConfig, SourceConfig
from yieldrep.evaluation.reconstruction import evaluate_reconstruction
from yieldrep.models.autoencoder import fit_autoencoder_panel


def test_evaluate_reconstruction_writes_summary_tables(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    ns_dir = processed_dir / "nelson_siegel"
    ns_dir.mkdir(parents=True)
    curves = _sample_curves()
    curves.to_parquet(processed_dir / "curves.parquet", index=False)
    _sample_nelson_siegel_fitted(curves).to_parquet(ns_dir / "us_fitted.parquet", index=False)
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
        pca=PCAConfig(n_components=2, min_maturities=3),
    )

    output_paths = evaluate_reconstruction(config)
    summary = pd.read_csv(output_paths[0])
    by_maturity = pd.read_csv(output_paths[1])
    worst_maturities = pd.read_csv(output_paths[2])
    oos_summary = pd.read_csv(output_paths[3])
    oos_by_maturity = pd.read_csv(output_paths[4])

    assert output_paths == [
        tmp_path / "reports" / "tables" / "reconstruction_summary.csv",
        tmp_path / "reports" / "tables" / "reconstruction_by_maturity.csv",
        tmp_path / "reports" / "tables" / "reconstruction_worst_maturities.csv",
        tmp_path / "reports" / "tables" / "reconstruction_oos_summary.csv",
        tmp_path / "reports" / "tables" / "reconstruction_oos_by_maturity.csv",
        tmp_path / "reports" / "tables" / "reconstruction_oos_by_maturity_bucket.csv",
        tmp_path / "reports" / "tables" / "reconstruction_oos_comparison.csv",
        tmp_path / "reports" / "tables" / "masked_reconstruction_by_maturity.csv",
        tmp_path / "reports" / "tables" / "masked_reconstruction_by_maturity_bucket.csv",
        tmp_path / "reports" / "tables" / "masked_reconstruction_hardest_maturities.csv",
    ]
    assert set(summary["representation"]) == {"pca", "nelson_siegel"}
    assert set(summary.loc[summary["representation"].eq("pca"), "n_components"]) == {1, 2}
    assert {
        "reconstruction_task",
        "observations",
        "dates",
        "rmse",
        "mae",
        "mean_error",
    }.issubset(summary.columns)
    assert {"maturity_years", "maturity_bucket"}.issubset(by_maturity.columns)
    assert {"abs_mean_error", "rmse_rank"}.issubset(worst_maturities.columns)
    assert worst_maturities["rmse_rank"].min() == 1
    assert set(oos_summary["representation"]) == {"pca", "nelson_siegel"}
    assert set(oos_summary["reconstruction_task"]) == {"clean_reconstruction"}
    assert {"maturity_years", "maturity_bucket"}.issubset(oos_by_maturity.columns)
    oos_by_bucket = pd.read_csv(output_paths[5])
    oos_comparison = pd.read_csv(output_paths[6])
    assert {"maturity_bucket", "rmse"}.issubset(oos_by_bucket.columns)
    assert {"rmse_rank", "rmse_gap_to_best", "pct_rmse_gap_to_best"}.issubset(
        oos_comparison.columns
    )
    masked_hardest = pd.read_csv(output_paths[9])
    assert {"maturity_years", "maturity_bucket", "hardness_rank", "rmse"}.issubset(
        masked_hardest.columns
    )


def test_fit_autoencoder_panel_returns_embeddings_and_reconstruction() -> None:
    panel = _sample_curves().pivot(index="date", columns="maturity_years", values="yield")
    panel.attrs["country"] = "US"

    result = fit_autoencoder_panel(
        panel=panel,
        test_fraction=0.5,
        latent_dim=2,
        hidden_dim=4,
        depth=1,
        dropout=0.0,
        epochs=20,
        batch_size=2,
        learning_rate=0.01,
        weight_decay=0.0,
        validation_fraction=0.5,
        mask_probability=0.4,
        clean_loss_weight=0.2,
        early_stopping_patience=3,
        min_delta=0.0,
        random_seed=42,
    )

    assert {"date", "country", "split", "AE1", "AE2"}.issubset(result.embeddings.columns)
    assert {
        "date",
        "country",
        "maturity_years",
        "yield",
        "fitted_yield",
        "split",
    }.issubset(result.reconstruction.columns)
    assert {
        "split",
        "rmse",
        "mae",
        "depth",
        "dropout",
        "batch_size",
        "weight_decay",
        "epochs_trained",
        "best_validation_loss",
    }.issubset(result.metrics.columns)
    assert {"epoch", "train_loss", "validation_loss", "is_best_epoch"}.issubset(
        result.training_history.columns
    )
    assert set(result.reconstruction["split"]) == {"train", "validation", "test"}
    assert set(result.masked_reconstruction["split"]) == {"train", "validation", "test"}
    assert set(result.metrics["metric_scope"]) == {"clean", "masked"}
    assert set(result.metrics["split"]) == {"train", "validation", "test"}
    assert len(result.reconstruction) == panel.size
    assert len(result.masked_reconstruction) < panel.size
    assert result.training_history["is_best_epoch"].sum() == 1


def _sample_curves() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=6)
    rows = []
    for index, date in enumerate(dates):
        for maturity in [1.0, 5.0, 10.0]:
            rows.append(
                {
                    "date": date,
                    "country": "US",
                    "maturity_years": maturity,
                    "yield": 4.0 + 0.01 * index + 0.02 * maturity,
                    "source": "test",
                }
            )
    return pd.DataFrame(rows)


def _sample_nelson_siegel_fitted(curves: pd.DataFrame) -> pd.DataFrame:
    fitted = curves.loc[:, ["date", "country", "maturity_years", "yield"]].copy()
    fitted["fitted_yield"] = fitted["yield"] - 0.001
    fitted["residual"] = 0.001
    fitted["tau"] = 1.5
    return fitted.drop(columns=["yield"])
