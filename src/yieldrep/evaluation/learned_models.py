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

FINDINGS_COLUMNS = [
    "area",
    "question",
    "best_learned_model",
    "best_classical_or_baseline",
    "result",
    "interpretation",
    "evidence_table",
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


def build_learned_model_findings(config: ProjectConfig) -> Path:
    """Write a compact narrative table of learned-model findings."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    table = learned_model_findings(config)
    table.to_csv(config.learned_model_findings_table_path, index=False)
    return config.learned_model_findings_table_path


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


def learned_model_findings(config: ProjectConfig) -> pd.DataFrame:
    rows = [
        _clean_reconstruction_finding(config),
        _masked_reconstruction_finding(config),
        _learned_state_finding(config),
        _volatility_regime_finding(config),
        _residual_change_finding(config),
    ]
    return pd.DataFrame(rows, columns=FINDINGS_COLUMNS)


def _clean_reconstruction_finding(config: ProjectConfig) -> dict[str, object]:
    scorecard = _scorecard_row(config, "clean_reconstruction")
    best_learned = _value(scorecard, "best_learned_representation")
    best_classical = _value(scorecard, "best_representation")
    learned_rank = _value(scorecard, "best_learned_rank")
    return {
        "area": "clean_reconstruction",
        "question": "Which representation reconstructs observed curves most accurately?",
        "best_learned_model": _model_label(best_learned, _value(scorecard, "best_learned_model")),
        "best_classical_or_baseline": _model_label(
            best_classical,
            _value(scorecard, "best_model"),
        ),
        "result": _metric_result("best learned rank", learned_rank),
        "interpretation": "PCA remains the clean-reconstruction hurdle; the adjacent-edge graph AE is the strongest learned clean reconstructor in the current scorecard.",
        "evidence_table": str(config.representation_task_scorecard_table_path),
    }


def _masked_reconstruction_finding(config: ProjectConfig) -> dict[str, object]:
    leaderboard = _read_csv(config.learned_reconstruction_leaderboard_table_path)
    if leaderboard.empty:
        result = "not evaluated"
        interpretation = "Run reconstruction and learned-model comparison to populate this finding."
    else:
        winners = leaderboard["best_masked_model"].value_counts().to_dict()
        graph_wins = int(winners.get("masked_graph_autoencoder", 0))
        ae_wins = int(winners.get("masked_autoencoder", 0))
        result = f"graph AE wins {graph_wins} markets; AE wins {ae_wins} markets"
        interpretation = "The adjacent-edge graph AE leads masked reconstruction for US and euro-area curves; the MLP autoencoder remains stronger for Canada."
    return {
        "area": "masked_reconstruction",
        "question": "Which learned model infers hidden maturities most accurately?",
        "best_learned_model": _mode_label(leaderboard, "best_masked_model"),
        "best_classical_or_baseline": "masked_autoencoder; masked_transformer",
        "result": result,
        "interpretation": interpretation,
        "evidence_table": str(config.learned_reconstruction_leaderboard_table_path),
    }


def _learned_state_finding(config: ProjectConfig) -> dict[str, object]:
    summary = _read_csv(config.learned_state_regime_summary_table_path)
    if summary.empty:
        best_model = "not_evaluated"
        result = "not evaluated"
        interpretation = "Run learned-state diagnostics to populate this finding."
    else:
        best = summary.sort_values("separation_ratio", ascending=False).iloc[0]
        best_model = str(best["representation"])
        result = (
            f"{best['country']} {best['regime_type']}:{best['indicator']} "
            f"separation_ratio={float(best['separation_ratio']):.3f}"
        )
        interpretation = "Transformer has the strongest current regime-separation diagnostic; graph AE also separates CA unemployment and selected EA inflation/MOVE regimes."
    return {
        "area": "learned_state_regime_separation",
        "question": "Which learned state best separates macro or market regimes?",
        "best_learned_model": best_model,
        "best_classical_or_baseline": "not applicable",
        "result": result,
        "interpretation": interpretation,
        "evidence_table": str(config.learned_state_regime_summary_table_path),
    }


def _volatility_regime_finding(config: ProjectConfig) -> dict[str, object]:
    benchmark = _read_csv(config.volatility_regime_benchmark_table_path)
    if benchmark.empty:
        return {
            "area": "volatility_regime_classification",
            "question": "Do learned embeddings help classify future curve-volatility regimes?",
            "best_learned_model": "not_evaluated",
            "best_classical_or_baseline": "not_evaluated",
            "result": "not evaluated",
            "interpretation": "Run volatility-regime evaluation to populate this finding.",
            "evidence_table": str(config.volatility_regime_benchmark_table_path),
        }

    learned_columns = {
        "autoencoder": "autoencoder_balanced_accuracy",
        "transformer": "transformer_balanced_accuracy",
        "graph_autoencoder": "graph_autoencoder_balanced_accuracy",
    }
    best_learned, best_value = _best_column_value(benchmark, learned_columns)
    best = benchmark.sort_values("best_balanced_accuracy", ascending=False).iloc[0]
    graph_wins = int(benchmark["best_model"].astype(str).str.startswith("graph_autoencoder/").sum())
    return {
        "area": "volatility_regime_classification",
        "question": "Do learned embeddings help classify future curve-volatility regimes?",
        "best_learned_model": _metric_result(best_learned, best_value),
        "best_classical_or_baseline": str(best["best_model"]),
        "result": f"graph AE wins {graph_wins} country-horizon cells",
        "interpretation": "Graph AE wins US 1-day volatility-regime classification; curve-vol, policy, and AE remain the main hurdles elsewhere.",
        "evidence_table": str(config.volatility_regime_benchmark_table_path),
    }


def _residual_change_finding(config: ProjectConfig) -> dict[str, object]:
    rank = _read_csv(config.baseline_rank_table_path)
    residual = rank.loc[rank["target"] == "residual_change"].copy() if not rank.empty else rank
    graph = (
        residual.loc[
            (residual["representation"] == "graph_autoencoder") & (residual["model"] == "ridge")
        ].copy()
        if not residual.empty
        else residual
    )
    if graph.empty:
        result = "not evaluated"
        interpretation = "Run residual-change evaluation to populate this finding."
    else:
        wins = int(graph["rank"].eq(1.0).sum())
        best_ic = graph.sort_values("mean_rank_ic", ascending=False).iloc[0]
        result = (
            f"graph AE ridge wins {wins} country-horizon cells; "
            f"best rank_ic={float(best_ic['mean_rank_ic']):.3f}"
        )
        interpretation = "Graph AE is most interesting on 20-day residual targets, but the overall residual-change scorecard still favors the MLP autoencoder."
    scorecard = _scorecard_row(config, "residual_change_forecasting")
    return {
        "area": "residual_change_rv_forecasting",
        "question": "Do learned graph states help forecast Nelson-Siegel residual changes?",
        "best_learned_model": _model_label(
            _value(scorecard, "best_learned_representation"),
            _value(scorecard, "best_learned_model"),
        ),
        "best_classical_or_baseline": _model_label(
            _value(scorecard, "best_classical_representation"),
            _value(scorecard, "best_classical_model"),
        ),
        "result": result,
        "interpretation": interpretation,
        "evidence_table": str(config.baseline_rank_table_path),
    }


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _scorecard_row(config: ProjectConfig, scenario: str) -> pd.Series:
    scorecard = _read_csv(config.representation_task_scorecard_table_path)
    selected = (
        scorecard.loc[scorecard["scenario"] == scenario] if not scorecard.empty else scorecard
    )
    return selected.iloc[0] if not selected.empty else pd.Series(dtype=object)


def _value(row: pd.Series, column: str) -> object:
    return row[column] if column in row.index and not pd.isna(row[column]) else ""


def _model_label(representation: object, model: object) -> str:
    if not representation:
        return "not_evaluated"
    return f"{representation}/{model}" if model else str(representation)


def _metric_result(label: object, value: object) -> str:
    if _is_missing(value):
        return str(label)
    if _is_number(value):
        return f"{label}={float(str(value)):.3f}"
    return f"{label}: {value}"


def _is_missing(value: object) -> bool:
    if value == "":
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _is_number(value: object) -> bool:
    try:
        float(str(value))
    except ValueError:
        return False
    return True


def _mode_label(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return "not_evaluated"
    mode = frame[column].mode(dropna=True)
    return str(mode.iloc[0]) if not mode.empty else "not_evaluated"


def _best_column_value(
    frame: pd.DataFrame,
    columns_by_label: dict[str, str],
) -> tuple[str, float]:
    rows: list[tuple[str, float]] = []
    for label, column in columns_by_label.items():
        if column not in frame.columns:
            continue
        values = frame[column].dropna()
        if not values.empty:
            rows.append((label, float(values.max())))
    return max(rows, key=lambda row: row[1]) if rows else ("not_evaluated", float("nan"))
