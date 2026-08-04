from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from yieldrep.config import ProjectConfig


COMPARISON_COLUMNS = [
    "model",
    "country",
    "latent_dim",
    "clean_test_rmse",
    "masked_test_rmse",
    "clean_validation_rmse",
    "masked_validation_rmse",
    "clean_test_mae",
    "masked_test_mae",
    "clean_validation_to_test_rmse_gap",
    "masked_validation_to_test_rmse_gap",
    "train_split_dates",
    "fit_train_dates",
    "validation_dates",
    "test_dates",
    "epochs_trained",
    "best_validation_loss",
    "final_train_loss",
    "final_validation_loss",
    "mask_probability",
    "clean_loss_weight",
    "training_protocol",
]

LEARNED_RECONSTRUCTION_REPRESENTATIONS = {
    "autoencoder",
    "transformer",
    "graph_autoencoder",
    "masked_autoencoder",
    "masked_transformer",
    "masked_graph_autoencoder",
}

LEADERBOARD_COLUMNS = [
    "country",
    "best_clean_model",
    "best_clean_rmse",
    "best_masked_model",
    "best_masked_rmse",
    "autoencoder_masked_rmse",
    "graph_autoencoder_masked_rmse",
    "transformer_masked_rmse",
    "graph_beats_autoencoder",
    "graph_beats_transformer",
]


def build_learned_model_comparison(config: ProjectConfig) -> Path:
    """Write a compact learned-model training and reconstruction comparison."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    table = learned_model_comparison_table(config)
    table.to_csv(config.learned_model_comparison_table_path, index=False)
    return config.learned_model_comparison_table_path


def build_learned_reconstruction_leaderboard(config: ProjectConfig) -> Path:
    """Write one-row-per-country learned reconstruction winners from existing metrics."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    table = learned_reconstruction_leaderboard(config)
    table.to_csv(config.learned_reconstruction_leaderboard_table_path, index=False)
    return config.learned_reconstruction_leaderboard_table_path


def learned_model_comparison_table(config: ProjectConfig) -> pd.DataFrame:
    rows = [
        _model_rows("autoencoder", config.autoencoder_dir, max_train_dates=None),
        _model_rows(
            "transformer",
            config.transformer_dir,
            max_train_dates=config.transformer.max_train_dates,
        ),
        _model_rows(
            "graph_autoencoder",
            config.gnn_dir,
            max_train_dates=config.gnn.max_train_dates,
        ),
    ]
    non_empty = [row for row in rows if not row.empty]
    if not non_empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    table = pd.concat(non_empty, ignore_index=True)
    table["clean_validation_to_test_rmse_gap"] = (
        table["clean_test_rmse"] - table["clean_validation_rmse"]
    )
    table["masked_validation_to_test_rmse_gap"] = (
        table["masked_test_rmse"] - table["masked_validation_rmse"]
    )
    table["training_protocol"] = "chronological train/validation/test split"
    return table.loc[:, COMPARISON_COLUMNS].sort_values(["country", "model"]).reset_index(drop=True)


def _model_rows(model: str, model_dir: Path, max_train_dates: int | None) -> pd.DataFrame:
    if not model_dir.exists():
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    rows = []
    for metrics_path in sorted(model_dir.glob("*_metrics.parquet")):
        metrics = pd.read_parquet(metrics_path)
        if metrics.empty:
            continue

        country = str(metrics["country"].iloc[0])
        row: dict[str, object] = {
            "model": model,
            "country": country,
        }
        row.update(
            _split_counts(
                model_dir=model_dir,
                country_key=metrics_path.stem.removesuffix("_metrics"),
                max_train_dates=max_train_dates,
            )
        )
        row.update(_metric_values(metrics))
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    return pd.DataFrame(rows)


def _metric_values(metrics: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {}
    for scope in ["clean", "masked"]:
        for split in ["validation", "test"]:
            selected = metrics.loc[(metrics["metric_scope"] == scope) & (metrics["split"] == split)]
            if selected.empty:
                row[f"{scope}_{split}_rmse"] = np.nan
                row[f"{scope}_{split}_mae"] = np.nan
                continue

            metric_row = selected.iloc[0]
            row[f"{scope}_{split}_rmse"] = float(metric_row["rmse"])
            row[f"{scope}_{split}_mae"] = float(metric_row["mae"])

    metadata = metrics.iloc[0]
    for column in [
        "latent_dim",
        "epochs_trained",
        "best_validation_loss",
        "final_train_loss",
        "final_validation_loss",
        "mask_probability",
        "clean_loss_weight",
    ]:
        row[column] = metadata[column] if column in metadata.index else np.nan
    return row


def _split_counts(
    model_dir: Path,
    country_key: str,
    max_train_dates: int | None,
) -> dict[str, int | float]:
    reconstruction_path = model_dir / f"{country_key}_reconstruction.parquet"
    if not reconstruction_path.exists():
        return {
            "train_split_dates": np.nan,
            "fit_train_dates": np.nan,
            "validation_dates": np.nan,
            "test_dates": np.nan,
        }

    reconstruction = pd.read_parquet(reconstruction_path)
    counts = reconstruction.groupby("split", sort=False)["date"].nunique()
    train_split_dates = int(counts.get("train", 0))
    return {
        "train_split_dates": train_split_dates,
        "fit_train_dates": _fit_train_dates(train_split_dates, max_train_dates),
        "validation_dates": int(counts.get("validation", 0)),
        "test_dates": int(counts.get("test", 0)),
    }


def _fit_train_dates(train_split_dates: int, max_train_dates: int | None) -> int:
    if max_train_dates is None:
        return train_split_dates
    return min(train_split_dates, max_train_dates)


def learned_reconstruction_leaderboard(config: ProjectConfig) -> pd.DataFrame:
    if not config.reconstruction_oos_comparison_table_path.exists():
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    comparison = pd.read_csv(config.reconstruction_oos_comparison_table_path)
    learned = comparison.loc[
        comparison["representation"].isin(LEARNED_RECONSTRUCTION_REPRESENTATIONS)
    ].copy()
    if learned.empty:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    rows = [
        _country_leaderboard_row(country, group) for country, group in learned.groupby("country")
    ]
    return (
        pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS)
        .sort_values("country")
        .reset_index(drop=True)
    )


def _country_leaderboard_row(country: str, group: pd.DataFrame) -> dict[str, object]:
    clean = group.loc[group["reconstruction_task"] == "clean_reconstruction"].copy()
    masked = group.loc[group["reconstruction_task"] == "masked_maturity_reconstruction"].copy()
    best_clean = _best_row(clean)
    best_masked = _best_row(masked)
    ae_masked_rmse = _masked_rmse(masked, "masked_autoencoder")
    graph_masked_rmse = _masked_rmse(masked, "masked_graph_autoencoder")
    transformer_masked_rmse = _masked_rmse(masked, "masked_transformer")
    return {
        "country": country,
        "best_clean_model": best_clean.get("representation", ""),
        "best_clean_rmse": best_clean.get("rmse", np.nan),
        "best_masked_model": best_masked.get("representation", ""),
        "best_masked_rmse": best_masked.get("rmse", np.nan),
        "autoencoder_masked_rmse": ae_masked_rmse,
        "graph_autoencoder_masked_rmse": graph_masked_rmse,
        "transformer_masked_rmse": transformer_masked_rmse,
        "graph_beats_autoencoder": _strictly_less(graph_masked_rmse, ae_masked_rmse),
        "graph_beats_transformer": _strictly_less(graph_masked_rmse, transformer_masked_rmse),
    }


def _best_row(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {}
    row = frame.sort_values(["rmse", "mae", "representation"]).iloc[0]
    return {"representation": row["representation"], "rmse": float(row["rmse"])}


def _masked_rmse(masked: pd.DataFrame, representation: str) -> float:
    selected = masked.loc[masked["representation"] == representation]
    if selected.empty:
        return float("nan")
    return float(selected.sort_values(["rmse", "mae"]).iloc[0]["rmse"])


def _strictly_less(left: float, right: float) -> bool:
    if np.isnan(left) or np.isnan(right):
        return False
    return left < right
