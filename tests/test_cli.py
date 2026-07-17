from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from yieldrep.cli import app


def test_ingest_command_downloads_raw_file(tmp_path: Path) -> None:
    source_file = tmp_path / "source.csv"
    source_file.write_text("Date,Value\n2024-01-02,1.0\n", encoding="utf-8")
    raw_file = tmp_path / "data" / "raw" / "source.csv"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_dir: {tmp_path / 'data'}",
                f"reports_dir: {tmp_path / 'reports'}",
                "sources:",
                "  test:",
                "    country: US",
                "    source: test",
                f"    raw_file: {raw_file}",
                f"    url: {source_file.as_uri()}",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["ingest", "--config", str(config_path)])

    assert result.exit_code == 0
    assert raw_file.read_text(encoding="utf-8") == source_file.read_text(encoding="utf-8")
    assert str(raw_file) in result.stdout


def test_normalize_command_writes_curves_parquet(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    fed_raw = raw_dir / "fed_gsw.csv"
    fed_raw.write_text(
        "\n".join(
            [
                "Note: research data",
                "Date,BETA0,SVENY01,SVENY10",
                "2024-01-02,1.0,4.00,4.20",
            ]
        ),
        encoding="utf-8",
    )

    boc_raw = raw_dir / "boc_zero_coupon.csv"
    boc_raw.write_text(
        "\n".join(
            [
                "Date, ZC025YR, ZC100YR,",
                "2024-01-02, 0.0400, 0.0425,",
            ]
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_dir: {tmp_path / 'data'}",
                f"reports_dir: {tmp_path / 'reports'}",
                "sources:",
                "  fed_gsw:",
                "    country: US",
                "    source: fed_gsw",
                f"    raw_file: {fed_raw}",
                "  bank_of_canada:",
                "    country: CA",
                "    source: bank_of_canada",
                f"    raw_file: {boc_raw}",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["normalize", "--config", str(config_path)])

    assert result.exit_code == 0
    assert (tmp_path / "data" / "processed" / "curves.parquet").exists()
    assert "curves.parquet" in result.stdout


def test_build_pca_command_writes_outputs(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    dates = pd.date_range("2024-01-01", periods=4)
    curves = pd.DataFrame(
        [
            {
                "date": date,
                "country": "US",
                "maturity_years": maturity,
                "yield": 3.0 + date_index * 0.1 + maturity * 0.02,
                "source": "test",
            }
            for date_index, date in enumerate(dates)
            for maturity in [1.0, 2.0, 10.0]
        ]
    )
    curves.to_parquet(processed_dir / "curves.parquet", index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_dir: {tmp_path / 'data'}",
                f"reports_dir: {tmp_path / 'reports'}",
                "pca:",
                "  n_components: 2",
                "  min_maturities: 3",
                "sources:",
                "  test:",
                "    country: US",
                "    source: test",
                f"    raw_file: {tmp_path / 'raw.csv'}",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["build-pca", "--config", str(config_path)])

    assert result.exit_code == 0
    assert (processed_dir / "pca" / "us_scores.parquet").exists()
    assert (processed_dir / "pca" / "us_loadings.parquet").exists()
    assert (processed_dir / "pca" / "us_variance.parquet").exists()
    assert "us_scores.parquet" in result.stdout


def test_build_nelson_siegel_command_writes_outputs(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    dates = pd.date_range("2024-01-01", periods=3)
    curves = pd.DataFrame(
        [
            {
                "date": date,
                "country": "US",
                "maturity_years": maturity,
                "yield": 3.0 + date_index * 0.1 + maturity * 0.02,
                "source": "test",
            }
            for date_index, date in enumerate(dates)
            for maturity in [1.0, 2.0, 5.0, 10.0]
        ]
    )
    curves.to_parquet(processed_dir / "curves.parquet", index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_dir: {tmp_path / 'data'}",
                f"reports_dir: {tmp_path / 'reports'}",
                "nelson_siegel:",
                "  tau: 1.5",
                "  min_maturities: 3",
                "sources:",
                "  test:",
                "    country: US",
                "    source: test",
                f"    raw_file: {tmp_path / 'raw.csv'}",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["build-nelson-siegel", "--config", str(config_path)])

    assert result.exit_code == 0
    assert (processed_dir / "nelson_siegel" / "us_factors.parquet").exists()
    assert (processed_dir / "nelson_siegel" / "us_fitted.parquet").exists()
    assert "us_factors.parquet" in result.stdout


def test_build_targets_command_writes_output(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    dates = pd.date_range("2024-01-01", periods=4)
    curves = pd.DataFrame(
        [
            {
                "date": date,
                "country": "US",
                "maturity_years": maturity,
                "yield": 3.0 + date_index * 0.1 + maturity * 0.01,
                "source": "test",
            }
            for date_index, date in enumerate(dates)
            for maturity in [2.0, 10.0]
        ]
    )
    curves.to_parquet(processed_dir / "curves.parquet", index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_dir: {tmp_path / 'data'}",
                f"reports_dir: {tmp_path / 'reports'}",
                "targets:",
                "  horizons_days: [1]",
                "sources:",
                "  test:",
                "    country: US",
                "    source: test",
                f"    raw_file: {tmp_path / 'raw.csv'}",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["build-targets", "--config", str(config_path)])

    assert result.exit_code == 0
    assert (processed_dir / "targets.parquet").exists()
    assert "targets.parquet" in result.stdout


def test_build_modeling_datasets_command_writes_outputs(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    pca_dir = processed_dir / "pca"
    ns_dir = processed_dir / "nelson_siegel"
    pca_dir.mkdir(parents=True)
    ns_dir.mkdir(parents=True)
    dates = pd.date_range("2024-01-01", periods=2)
    pd.DataFrame(
        {
            "date": dates,
            "country": ["US", "US"],
            "maturity_years": [2.0, 2.0],
            "yield": [4.0, 4.1],
            "source": ["test", "test"],
        }
    ).to_parquet(processed_dir / "curves.parquet", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "country": ["US", "US"],
            "maturity_years": [2.0, 2.0],
            "horizon_days": [1, 1],
            "yield": [4.0, 4.1],
            "future_yield": [4.1, 4.2],
            "target_yield_change": [0.1, 0.1],
        }
    ).to_parquet(processed_dir / "targets.parquet", index=False)
    pd.DataFrame({"date": dates, "PC1": [1.0, 1.1]}).to_parquet(
        pca_dir / "us_scores.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "date": dates,
            "country": ["US", "US"],
            "beta_level": [4.0, 4.1],
            "beta_slope": [-1.0, -0.9],
            "beta_curvature": [0.5, 0.4],
            "tau": [1.5, 1.5],
            "rmse": [0.01, 0.02],
        }
    ).to_parquet(ns_dir / "us_factors.parquet", index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_dir: {tmp_path / 'data'}",
                f"reports_dir: {tmp_path / 'reports'}",
                "evaluation:",
                "  lag_days: [1]",
                "sources:",
                "  test:",
                "    country: US",
                "    source: test",
                f"    raw_file: {tmp_path / 'raw.csv'}",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["build-modeling-datasets", "--config", str(config_path)])

    assert result.exit_code == 0
    assert (processed_dir / "modeling" / "pca_targets.parquet").exists()
    assert (processed_dir / "modeling" / "nelson_siegel_targets.parquet").exists()
    assert (processed_dir / "modeling" / "lagged_targets.parquet").exists()
    assert "pca_targets.parquet" in result.stdout


def test_evaluate_baselines_command_writes_metrics(tmp_path: Path) -> None:
    modeling_dir = tmp_path / "data" / "processed" / "modeling"
    modeling_dir.mkdir(parents=True)
    dates = pd.date_range("2024-01-01", periods=12)
    pca_rows = []
    ns_rows = []
    for index, date in enumerate(dates):
        common = {
            "date": date,
            "country": "US",
            "maturity_years": 2.0,
            "horizon_days": 1,
            "yield": 4.0,
            "future_yield": 4.0 + index * 0.01,
            "target_yield_change": index * 0.01,
        }
        pca_rows.append({**common, "PC1": float(index)})
        ns_rows.append(
            {
                **common,
                "beta_level": float(index),
                "beta_slope": -1.0,
                "beta_curvature": 0.5,
                "rmse": 0.01,
            }
        )
    pd.DataFrame(pca_rows).to_parquet(modeling_dir / "pca_targets.parquet", index=False)
    pd.DataFrame(ns_rows).to_parquet(
        modeling_dir / "nelson_siegel_targets.parquet",
        index=False,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_dir: {tmp_path / 'data'}",
                f"reports_dir: {tmp_path / 'reports'}",
                "evaluation:",
                "  test_fraction: 0.25",
                "  ridge_alpha: 1.0",
                "sources:",
                "  test:",
                "    country: US",
                "    source: test",
                f"    raw_file: {tmp_path / 'raw.csv'}",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["evaluate-baselines", "--config", str(config_path)])

    assert result.exit_code == 0
    assert (tmp_path / "data" / "processed" / "evaluation" / "baseline_metrics.parquet").exists()
    assert (
        tmp_path / "data" / "processed" / "evaluation" / "baseline_metrics_by_maturity.parquet"
    ).exists()
    assert "baseline_metrics.parquet" in result.stdout


def test_plot_pca_command_writes_html_outputs(tmp_path: Path) -> None:
    pca_dir = tmp_path / "data" / "processed" / "pca"
    pca_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "PC1": [1.0, 0.5, -0.5],
            "PC2": [0.0, 0.2, 0.1],
            "PC3": [0.1, -0.1, 0.0],
        }
    ).to_parquet(pca_dir / "us_scores.parquet", index=False)
    pd.DataFrame(
        {
            "component": ["PC1", "PC2", "PC3"],
            "explained_variance_ratio": [0.8, 0.15, 0.05],
        }
    ).to_parquet(pca_dir / "us_variance.parquet", index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_dir: {tmp_path / 'data'}",
                f"reports_dir: {tmp_path / 'reports'}",
                "sources:",
                "  test:",
                "    country: US",
                "    source: test",
                f"    raw_file: {tmp_path / 'raw.csv'}",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["plot-pca", "--config", str(config_path)])

    assert result.exit_code == 0
    assert (tmp_path / "reports" / "figures" / "us_pca_explained_variance.html").exists()
    assert (tmp_path / "reports" / "figures" / "us_pca_scores.html").exists()
    assert "us_pca_scores.html" in result.stdout


def test_plot_nelson_siegel_command_writes_html_outputs(tmp_path: Path) -> None:
    ns_dir = tmp_path / "data" / "processed" / "nelson_siegel"
    ns_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "country": ["US", "US", "US"],
            "beta_level": [4.0, 4.1, 4.2],
            "beta_slope": [-1.0, -0.9, -0.8],
            "beta_curvature": [0.5, 0.4, 0.3],
            "tau": [1.5, 1.5, 1.5],
            "rmse": [0.02, 0.03, 0.01],
        }
    ).to_parquet(ns_dir / "us_factors.parquet", index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_dir: {tmp_path / 'data'}",
                f"reports_dir: {tmp_path / 'reports'}",
                "sources:",
                "  test:",
                "    country: US",
                "    source: test",
                f"    raw_file: {tmp_path / 'raw.csv'}",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["plot-nelson-siegel", "--config", str(config_path)])

    assert result.exit_code == 0
    assert (tmp_path / "reports" / "figures" / "us_nelson_siegel_factors.html").exists()
    assert (tmp_path / "reports" / "figures" / "us_nelson_siegel_rmse.html").exists()
    assert "us_nelson_siegel_factors.html" in result.stdout


def test_plot_curves_command_writes_html_outputs(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    dates = pd.date_range("2024-01-01", periods=3)
    curves = pd.DataFrame(
        [
            {
                "date": date,
                "country": "US",
                "maturity_years": maturity,
                "yield": 3.0 + date_index * 0.1 + maturity * 0.02,
                "source": "test",
            }
            for date_index, date in enumerate(dates)
            for maturity in [1.0, 2.0, 10.0]
        ]
    )
    curves.to_parquet(processed_dir / "curves.parquet", index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_dir: {tmp_path / 'data'}",
                f"reports_dir: {tmp_path / 'reports'}",
                "sources:",
                "  test:",
                "    country: US",
                "    source: test",
                f"    raw_file: {tmp_path / 'raw.csv'}",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["plot-curves", "--config", str(config_path)])

    assert result.exit_code == 0
    assert (tmp_path / "reports" / "figures" / "us_selected_maturities.html").exists()
    assert (tmp_path / "reports" / "figures" / "us_curve_heatmap.html").exists()
    assert "us_curve_heatmap.html" in result.stdout
