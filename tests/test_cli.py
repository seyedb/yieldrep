from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

import yieldrep.cli as cli
from yieldrep.cli import app, run_baseline_pipeline
from yieldrep.config import ProjectConfig, SourceConfig


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
    config_path = _write_config(
        tmp_path,
        [
            "sources:",
            "  fed_gsw:",
            "    country: US",
            "    source: fed_gsw",
            f"    raw_file: {fed_raw}",
        ],
    )

    result = CliRunner().invoke(app, ["normalize", "--config", str(config_path)])

    assert result.exit_code == 0
    assert (tmp_path / "data" / "processed" / "curves.parquet").exists()
    assert "curves.parquet" in result.stdout


def test_evaluate_baselines_command_writes_metric_outputs(tmp_path: Path) -> None:
    modeling_dir = tmp_path / "data" / "processed" / "modeling"
    modeling_dir.mkdir(parents=True)
    dates = pd.date_range("2024-01-01", periods=12)
    rows = [
        {
            "date": date,
            "country": "US",
            "maturity_years": 2.0,
            "horizon_days": 1,
            "yield": 4.0,
            "future_yield": 4.0 + index * 0.01,
            "target_yield_change": index * 0.01,
            "PC1": float(index),
        }
        for index, date in enumerate(dates)
    ]
    pd.DataFrame(rows).to_parquet(modeling_dir / "pca_targets.parquet", index=False)
    config_path = _write_config(
        tmp_path,
        [
            "evaluation:",
            "  test_fraction: 0.25",
            "sources:",
            "  test:",
            "    country: US",
            "    source: test",
            f"    raw_file: {tmp_path / 'raw.csv'}",
        ],
    )

    result = CliRunner().invoke(app, ["evaluate-baselines", "--config", str(config_path)])

    assert result.exit_code == 0
    assert (tmp_path / "data" / "processed" / "evaluation" / "baseline_metrics.parquet").exists()
    assert (
        tmp_path / "data" / "processed" / "evaluation" / "baseline_metrics_by_maturity.parquet"
    ).exists()
    assert (
        tmp_path
        / "data"
        / "processed"
        / "evaluation"
        / "baseline_metrics_by_maturity_point.parquet"
    ).exists()


def test_summarize_baselines_command_writes_csv_tables(tmp_path: Path) -> None:
    evaluation_dir = tmp_path / "data" / "processed" / "evaluation"
    evaluation_dir.mkdir(parents=True)
    metrics = pd.DataFrame(
        {
            "target": ["yield_change", "yield_change"],
            "representation": ["pca", "curve"],
            "model": ["ridge", "ridge"],
            "split_method": ["date_ordered", "date_ordered"],
            "window_id": [0, 0],
            "country": ["US", "US"],
            "horizon_days": [1, 1],
            "rmse": [0.1, 0.2],
            "mae": [0.05, 0.1],
            "directional_accuracy": [0.5, 0.6],
            "train_rows": [10, 10],
            "test_rows": [3, 3],
            "train_dates": [10, 10],
            "test_dates": [3, 3],
        }
    )
    metrics.to_parquet(evaluation_dir / "baseline_metrics.parquet", index=False)
    metrics.assign(maturity_bucket=["front_end", "belly"]).to_parquet(
        evaluation_dir / "baseline_metrics_by_maturity.parquet",
        index=False,
    )
    metrics.assign(maturity_years=[2.0, 10.0]).to_parquet(
        evaluation_dir / "baseline_metrics_by_maturity_point.parquet",
        index=False,
    )
    config_path = _write_config(
        tmp_path,
        [
            "sources:",
            "  test:",
            "    country: US",
            "    source: test",
            f"    raw_file: {tmp_path / 'raw.csv'}",
        ],
    )

    result = CliRunner().invoke(app, ["summarize-baselines", "--config", str(config_path)])

    assert result.exit_code == 0
    assert (tmp_path / "reports" / "tables" / "baseline_summary.csv").exists()
    assert (tmp_path / "reports" / "tables" / "baseline_by_maturity_bucket.csv").exists()
    assert (tmp_path / "reports" / "tables" / "baseline_by_maturity_point_top.csv").exists()


def test_run_baseline_pipeline_orders_steps(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
    )

    def single_step(name: str):
        def _step(_config):
            calls.append(name)
            return tmp_path / f"{name}.out"

        return _step

    def multi_step(name: str):
        def _step(_config):
            calls.append(name)
            return [tmp_path / f"{name}.out"]

        return _step

    monkeypatch.setattr(cli, "build_curves_parquet", single_step("normalize"))
    monkeypatch.setattr(cli, "build_pca", multi_step("build_pca"))
    monkeypatch.setattr(cli, "build_nelson_siegel", multi_step("build_nelson_siegel"))
    monkeypatch.setattr(cli, "build_curve_features", single_step("build_curve_features"))
    monkeypatch.setattr(cli, "build_carry_roll_features", single_step("build_carry_roll_features"))
    monkeypatch.setattr(cli, "build_residual_features", single_step("build_residual_features"))
    monkeypatch.setattr(cli, "build_targets", single_step("build_targets"))
    monkeypatch.setattr(cli, "build_standardized_targets", single_step("build_standardized_targets"))
    monkeypatch.setattr(cli, "build_residual_targets", single_step("build_residual_targets"))
    monkeypatch.setattr(cli, "build_vol_targets", single_step("build_vol_targets"))
    monkeypatch.setattr(cli, "build_modeling_datasets", multi_step("build_modeling_datasets"))
    monkeypatch.setattr(cli, "evaluate_baselines", single_step("evaluate_baselines"))
    monkeypatch.setattr(cli, "summarize_baselines", multi_step("summarize_baselines"))
    monkeypatch.setattr(cli, "plot_baseline_metrics", multi_step("plot_baseline_metrics"))

    output_paths = run_baseline_pipeline(config)

    assert calls == [
        "normalize",
        "build_pca",
        "build_nelson_siegel",
        "build_curve_features",
        "build_carry_roll_features",
        "build_residual_features",
        "build_targets",
        "build_standardized_targets",
        "build_residual_targets",
        "build_vol_targets",
        "build_modeling_datasets",
        "evaluate_baselines",
        "summarize_baselines",
        "plot_baseline_metrics",
    ]
    assert len(output_paths) == len(calls)


def _write_config(tmp_path: Path, lines: list[str]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_dir: {tmp_path / 'data'}",
                f"reports_dir: {tmp_path / 'reports'}",
                *lines,
            ]
        ),
        encoding="utf-8",
    )
    return config_path
