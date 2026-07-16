from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from yieldrep.config import ProjectConfig
from yieldrep.evaluation.metrics import directional_accuracy, mae, rmse

TARGET_COLUMN = "target_yield_change"
GROUP_COLUMNS = ["country", "horizon_days"]
PCA_FEATURES = ["PC1", "PC2", "PC3", "PC4", "PC5"]
NELSON_SIEGEL_FEATURES = ["beta_level", "beta_slope", "beta_curvature", "rmse"]


@dataclass(frozen=True)
class EvaluationSpec:
    representation: str
    path: Path
    features: list[str]


def evaluate_baselines(config: ProjectConfig) -> Path:
    """Evaluate simple forecasting baselines on prepared modeling datasets."""
    specs = [
        EvaluationSpec(
            representation="pca",
            path=config.modeling_dir / "pca_targets.parquet",
            features=PCA_FEATURES,
        ),
        EvaluationSpec(
            representation="nelson_siegel",
            path=config.modeling_dir / "nelson_siegel_targets.parquet",
            features=NELSON_SIEGEL_FEATURES,
        ),
    ]

    rows: list[dict[str, object]] = []
    for spec in specs:
        if spec.path.exists():
            rows.extend(_evaluate_representation(config, spec))

    config.evaluation_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(config.baseline_metrics_path, index=False)
    return config.baseline_metrics_path


def _evaluate_representation(
    config: ProjectConfig,
    spec: EvaluationSpec,
) -> list[dict[str, object]]:
    data = pd.read_parquet(spec.path)
    feature_columns = [column for column in spec.features if column in data.columns]
    if not feature_columns:
        return []

    rows: list[dict[str, object]] = []
    for group_values, group in data.groupby(GROUP_COLUMNS, sort=True):
        country, horizon_days = group_values
        group = group.sort_values("date").dropna(subset=[*feature_columns, TARGET_COLUMN])
        split = int(len(group) * (1.0 - config.evaluation.test_fraction))
        if split <= 0 or split >= len(group):
            continue

        x_train = group.iloc[:split][feature_columns].to_numpy(dtype=float)
        y_train = group.iloc[:split][TARGET_COLUMN].to_numpy(dtype=float)
        x_test = group.iloc[split:][feature_columns].to_numpy(dtype=float)
        y_test = group.iloc[split:][TARGET_COLUMN].to_numpy(dtype=float)

        rows.append(
            _metric_row(
                representation=spec.representation,
                model="train_mean",
                country=str(country),
                horizon_days=int(horizon_days),
                y_true=y_test,
                y_pred=np.full_like(y_test, fill_value=float(np.mean(y_train))),
                train_rows=len(y_train),
                test_rows=len(y_test),
            )
        )

        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=config.evaluation.ridge_alpha),
        )
        model.fit(x_train, y_train)
        rows.append(
            _metric_row(
                representation=spec.representation,
                model="ridge",
                country=str(country),
                horizon_days=int(horizon_days),
                y_true=y_test,
                y_pred=model.predict(x_test),
                train_rows=len(y_train),
                test_rows=len(y_test),
            )
        )

    return rows


def _metric_row(
    representation: str,
    model: str,
    country: str,
    horizon_days: int,
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    train_rows: int,
    test_rows: int,
) -> dict[str, object]:
    return {
        "representation": representation,
        "model": model,
        "country": country,
        "horizon_days": horizon_days,
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "train_rows": train_rows,
        "test_rows": test_rows,
    }
