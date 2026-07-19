from pathlib import Path

import pandas as pd

from yieldrep.config import EvaluationConfig, ProjectConfig, SourceConfig
from yieldrep.evaluation.diagnostics import diagnose_lagged_baseline


def test_diagnose_lagged_baseline_writes_autocorrelation_metrics(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    modeling_dir = processed_dir / "modeling"
    modeling_dir.mkdir(parents=True)

    targets = _sample_targets()
    targets.to_parquet(processed_dir / "targets.parquet", index=False)
    _sample_lagged_targets(targets).to_parquet(modeling_dir / "lagged_targets.parquet", index=False)

    config = ProjectConfig(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        sources={"test": SourceConfig(country="US", source="test", raw_file=tmp_path / "raw.csv")},
        evaluation=EvaluationConfig(lag_days=[1, 2]),
    )

    output_path = diagnose_lagged_baseline(config)
    diagnostics = pd.read_csv(output_path)

    assert output_path == tmp_path / "reports" / "tables" / "lagged_diagnostics.csv"
    assert (tmp_path / "data" / "processed" / "evaluation" / "lagged_diagnostics.parquet").exists()
    assert set(diagnostics["target"]) == {"yield_change"}
    assert set(diagnostics["diagnostic"]) == {
        "target_autocorrelation",
        "lag_feature_correlation",
    }
    assert set(diagnostics["sample"]) == {"full", "non_overlapping"}
    assert set(diagnostics["maturity_bucket"]) == {"front_end", "belly"}
    assert set(diagnostics["lag_days"]) == {1, 2}
    assert {
        "country",
        "horizon_days",
        "correlation",
        "sign_agreement",
        "observations",
    }.issubset(diagnostics.columns)


def _sample_targets() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=8)
    target_values = [0.01, 0.02, 0.01, -0.01, -0.02, -0.01, 0.01, 0.02]
    rows = []
    for horizon_days in [1, 2]:
        for maturity_years in [2.0, 5.0]:
            for date, target_value in zip(dates, target_values, strict=True):
                rows.append(
                    {
                        "date": date,
                        "country": "US",
                        "maturity_years": maturity_years,
                        "horizon_days": horizon_days,
                        "yield": 4.0,
                        "future_yield": 4.1,
                        "target_yield_change": target_value + maturity_years * 0.001,
                    }
                )
    return pd.DataFrame(rows)


def _sample_lagged_targets(targets: pd.DataFrame) -> pd.DataFrame:
    lagged = targets.sort_values(["country", "maturity_years", "horizon_days", "date"]).copy()
    grouped = lagged.groupby(["country", "maturity_years", "horizon_days"], sort=False)[
        "target_yield_change"
    ]
    lagged["lag_1_change"] = grouped.shift(1)
    lagged["lag_2_change"] = grouped.shift(2)
    return lagged.dropna(subset=["lag_1_change", "lag_2_change"])
