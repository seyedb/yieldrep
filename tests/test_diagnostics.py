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
    diagnostics = pd.read_parquet(output_path)

    assert output_path == tmp_path / "data" / "processed" / "evaluation" / "lagged_diagnostics.parquet"
    assert set(diagnostics["target"]) == {"yield_change"}
    assert set(diagnostics["diagnostic"]) == {
        "target_autocorrelation",
        "lag_feature_correlation",
    }
    assert set(diagnostics["lag_days"]) == {1, 2}
    assert {"correlation", "sign_agreement", "observations"}.issubset(diagnostics.columns)


def _sample_targets() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=8)
    return pd.DataFrame(
        {
            "date": dates,
            "country": "US",
            "maturity_years": 2.0,
            "horizon_days": 1,
            "yield": 4.0,
            "future_yield": 4.1,
            "target_yield_change": [0.01, 0.02, 0.01, -0.01, -0.02, -0.01, 0.01, 0.02],
        }
    )


def _sample_lagged_targets(targets: pd.DataFrame) -> pd.DataFrame:
    lagged = targets.copy()
    lagged["lag_1_change"] = lagged["target_yield_change"].shift(1)
    lagged["lag_2_change"] = lagged["target_yield_change"].shift(2)
    return lagged.dropna(subset=["lag_1_change", "lag_2_change"])
