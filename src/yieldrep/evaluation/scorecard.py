from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from yieldrep.config import ProjectConfig


LEARNED_REPRESENTATIONS = {"autoencoder", "transformer"}
SCORECARD_COLUMNS = [
    "scenario",
    "question",
    "best_representation",
    "best_model",
    "primary_metric",
    "best_value",
    "best_learned_representation",
    "best_learned_model",
    "best_learned_value",
    "best_learned_rank",
    "evidence_table",
    "interpretation",
]


def build_representation_task_scorecard(config: ProjectConfig) -> Path:
    """Write a task-level scorecard across classical and learned representations."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    scorecard = representation_task_scorecard(config)
    scorecard.to_csv(config.representation_task_scorecard_table_path, index=False)
    return config.representation_task_scorecard_table_path


def representation_task_scorecard(config: ProjectConfig) -> pd.DataFrame:
    rows = [
        _reconstruction_row(config, "clean_reconstruction"),
        _reconstruction_row(config, "masked_maturity_reconstruction"),
        _forecast_row(config, "yield_change", "yield_change_forecasting"),
        _forecast_row(
            config,
            "standardized_yield_change",
            "standardized_yield_change_forecasting",
        ),
        _forecast_row(config, "residual_change", "residual_change_forecasting"),
        _forecast_row(config, "vol_change", "volatility_change_forecasting"),
        _residual_rv_row(config),
        _volatility_regime_row(config),
        _macro_regime_rv_row(config),
        _learned_state_row(config),
    ]
    non_empty = [row for row in rows if row]
    if not non_empty:
        return pd.DataFrame(columns=SCORECARD_COLUMNS)
    return pd.DataFrame(non_empty).loc[:, SCORECARD_COLUMNS]


def _reconstruction_row(config: ProjectConfig, task: str) -> dict[str, object]:
    table = config.reconstruction_oos_comparison_table_path
    if not table.exists():
        return _empty_row(task, str(table))

    data = pd.read_csv(table)
    data = data.loc[data["reconstruction_task"] == task].copy()
    if data.empty:
        return _empty_row(task, str(table))

    if task == "masked_maturity_reconstruction":
        data["representation"] = data["representation"].str.replace("masked_", "", regex=False)

    best = data.sort_values(["rmse", "mae", "representation", "n_components"]).iloc[0]
    learned = _best_learned(data, metric="rmse", ascending=True)
    return {
        **_base_row(
            scenario=task,
            question=_scenario_question(task),
            evidence_table=str(table),
        ),
        "best_representation": best["representation"],
        "best_model": f"n_components={int(best['n_components'])}",
        "primary_metric": "rmse",
        "best_value": float(best["rmse"]),
        **learned,
        "interpretation": _interpret_learned_rank(
            learned.get("best_learned_rank"),
            best_representation=str(best["representation"]),
        ),
    }


def _forecast_row(config: ProjectConfig, target: str, scenario: str) -> dict[str, object]:
    table = config.baseline_summary_table_path
    if not table.exists():
        table = config.baseline_rank_table_path
        if not table.exists():
            return _empty_row(scenario, str(table))

    data = pd.read_csv(table)
    data = data.loc[(data["target"] == target) & (data["model"] != "train_mean")].copy()
    if data.empty:
        return _empty_row(scenario, str(table))

    data = data.sort_values(["mean_rmse", "mean_mae", "representation", "model"]).reset_index(
        drop=True
    )
    data["scorecard_rank"] = data["mean_rmse"].rank(method="min", ascending=True)
    best = data.iloc[0]
    learned = _best_learned(
        data,
        metric="mean_rmse",
        ascending=True,
        rank_column="scorecard_rank",
    )
    return {
        **_base_row(
            scenario=scenario,
            question=_scenario_question(scenario),
            evidence_table=str(table),
        ),
        "best_representation": best["representation"],
        "best_model": best["model"],
        "primary_metric": "mean_rmse",
        "best_value": float(best["mean_rmse"]),
        **learned,
        "interpretation": _interpret_learned_rank(
            learned.get("best_learned_rank"),
            best_representation=str(best["representation"]),
        ),
    }


def _residual_rv_row(config: ProjectConfig) -> dict[str, object]:
    table = config.residual_relative_value_scorecard_table_path
    if not table.exists():
        return _empty_row("residual_relative_value", str(table))

    data = pd.read_csv(table)
    if data.empty:
        return _empty_row("residual_relative_value", str(table))

    best = data.sort_values(["spread_t_stat", "spread_hit_rate"], ascending=False).iloc[0]
    method = str(best["spread_method"])
    representation, model = _split_method(method)
    return {
        **_base_row(
            scenario="residual_relative_value",
            question=_scenario_question("residual_relative_value"),
            evidence_table=str(table),
        ),
        "best_representation": representation,
        "best_model": model,
        "primary_metric": "spread_t_stat",
        "best_value": float(best["spread_t_stat"]),
        "best_learned_representation": np.nan,
        "best_learned_model": np.nan,
        "best_learned_value": np.nan,
        "best_learned_rank": np.nan,
        "interpretation": str(best["takeaway"]),
    }


def _volatility_regime_row(config: ProjectConfig) -> dict[str, object]:
    table = config.volatility_regime_benchmark_table_path
    if not table.exists():
        return _empty_row("volatility_regime_classification", str(table))

    data = pd.read_csv(table)
    if data.empty:
        return _empty_row("volatility_regime_classification", str(table))

    best = data.sort_values("best_balanced_accuracy", ascending=False).iloc[0]
    representation, model = _split_method(str(best["best_model"]))
    learned_scores = []
    for representation_name in sorted(LEARNED_REPRESENTATIONS):
        column = f"{representation_name}_balanced_accuracy"
        if column in data.columns:
            selected = data.dropna(subset=[column]).sort_values(column, ascending=False)
            if not selected.empty:
                learned_scores.append((representation_name, selected.iloc[0]))
    learned = _best_learned_classification(learned_scores)
    return {
        **_base_row(
            scenario="volatility_regime_classification",
            question=_scenario_question("volatility_regime_classification"),
            evidence_table=str(table),
        ),
        "best_representation": representation,
        "best_model": model,
        "primary_metric": "balanced_accuracy",
        "best_value": float(best["best_balanced_accuracy"]),
        **learned,
        "interpretation": _interpret_learned_rank(
            learned.get("best_learned_rank"),
            best_representation=representation,
        ),
    }


def _macro_regime_rv_row(config: ProjectConfig) -> dict[str, object]:
    table = config.residual_rv_regime_scorecard_table_path
    if not table.exists():
        return _empty_row("macro_market_residual_rv_regimes", str(table))

    data = pd.read_csv(table)
    if data.empty:
        return _empty_row("macro_market_residual_rv_regimes", str(table))

    data["abs_hit_rate_gap"] = data["high_minus_low_hit_rate"].abs()
    best = data.sort_values(["abs_hit_rate_gap", "best_hit_rate"], ascending=False).iloc[0]
    return {
        **_base_row(
            scenario="macro_market_residual_rv_regimes",
            question=_scenario_question("macro_market_residual_rv_regimes"),
            evidence_table=str(table),
        ),
        "best_representation": "nelson_siegel_residual",
        "best_model": f"{best['regime_type']}:{best['indicator']}",
        "primary_metric": "abs_high_minus_low_hit_rate",
        "best_value": float(best["abs_hit_rate_gap"]),
        "best_learned_representation": np.nan,
        "best_learned_model": np.nan,
        "best_learned_value": np.nan,
        "best_learned_rank": np.nan,
        "interpretation": str(best["interpretation"]),
    }


def _learned_state_row(config: ProjectConfig) -> dict[str, object]:
    table = config.learned_state_regime_summary_table_path
    if not table.exists():
        return _empty_row("learned_state_regime_separation", str(table))

    data = pd.read_csv(table)
    if data.empty:
        return _empty_row("learned_state_regime_separation", str(table))

    best = data.sort_values("separation_ratio", ascending=False).iloc[0]
    return {
        **_base_row(
            scenario="learned_state_regime_separation",
            question=_scenario_question("learned_state_regime_separation"),
            evidence_table=str(table),
        ),
        "best_representation": best["representation"],
        "best_model": f"{best['country']} {best['regime_type']}:{best['indicator']}",
        "primary_metric": "separation_ratio",
        "best_value": float(best["separation_ratio"]),
        "best_learned_representation": best["representation"],
        "best_learned_model": f"{best['country']} {best['regime_type']}:{best['indicator']}",
        "best_learned_value": float(best["separation_ratio"]),
        "best_learned_rank": 1.0,
        "interpretation": "Learned-state diagnostic; not a forecasting result.",
    }


def _best_learned(
    data: pd.DataFrame,
    metric: str,
    ascending: bool,
    rank_column: str = "rmse_rank",
) -> dict[str, object]:
    learned = data.loc[data["representation"].isin(LEARNED_REPRESENTATIONS)].copy()
    if learned.empty:
        return _empty_learned()

    sort_columns = [metric, "representation"]
    sort_order = [ascending, True]
    if "model" in learned.columns:
        sort_columns.append("model")
        sort_order.append(True)
    best = learned.sort_values(sort_columns, ascending=sort_order).iloc[0]
    return {
        "best_learned_representation": best["representation"],
        "best_learned_model": best["model"] if "model" in learned.columns else np.nan,
        "best_learned_value": float(best[metric]),
        "best_learned_rank": float(best[rank_column]) if rank_column in best.index else np.nan,
    }


def _best_learned_classification(
    learned_scores: list[tuple[str, pd.Series]],
) -> dict[str, object]:
    if not learned_scores:
        return _empty_learned()

    representation, best = sorted(
        learned_scores,
        key=lambda item: float(item[1][f"{item[0]}_balanced_accuracy"]),
        reverse=True,
    )[0]
    value = float(best[f"{representation}_balanced_accuracy"])
    return {
        "best_learned_representation": representation,
        "best_learned_model": "logistic_l2",
        "best_learned_value": value,
        "best_learned_rank": 1.0 if str(best["best_model"]).startswith(representation) else np.nan,
    }


def _base_row(scenario: str, question: str, evidence_table: str) -> dict[str, object]:
    return {
        "scenario": scenario,
        "question": question,
        "evidence_table": evidence_table,
    }


def _empty_row(scenario: str, evidence_table: str) -> dict[str, object]:
    return {
        **_base_row(scenario, _scenario_question(scenario), evidence_table),
        "best_representation": np.nan,
        "best_model": np.nan,
        "primary_metric": np.nan,
        "best_value": np.nan,
        **_empty_learned(),
        "interpretation": "Evidence table is unavailable or empty.",
    }


def _empty_learned() -> dict[str, object]:
    return {
        "best_learned_representation": np.nan,
        "best_learned_model": np.nan,
        "best_learned_value": np.nan,
        "best_learned_rank": np.nan,
    }


def _interpret_learned_rank(rank: object, best_representation: str) -> str:
    if best_representation in LEARNED_REPRESENTATIONS:
        return "A learned representation is the current best method for this scenario."
    if pd.isna(rank):
        return "No learned representation is available for this scenario."
    rank_value = float(str(rank))
    if rank_value <= 3.0:
        return "A learned representation is competitive but not the main hurdle."
    return "Classical or engineered baselines remain stronger in this scenario."


def _split_method(method: str) -> tuple[str, str]:
    if "/" not in method:
        return method, ""
    representation, model = method.split("/", maxsplit=1)
    return representation, model


def _scenario_question(scenario: str) -> str:
    questions = {
        "clean_reconstruction": "Which representation reconstructs observed curves most accurately?",
        "masked_maturity_reconstruction": "Which learned model infers hidden maturities most accurately?",
        "yield_change_forecasting": "Which feature family forecasts outright yield changes best?",
        "standardized_yield_change_forecasting": "Which feature family forecasts volatility-scaled yield changes best?",
        "residual_change_forecasting": "Which feature family forecasts Nelson-Siegel residual changes best?",
        "volatility_change_forecasting": "Which feature family forecasts realized curve-volatility changes best?",
        "residual_relative_value": "Which residual-RV specification has the strongest convergence evidence?",
        "volatility_regime_classification": "Which representation classifies future curve-volatility regimes best?",
        "macro_market_residual_rv_regimes": "Which macro or market regime most changes residual-RV behavior?",
        "learned_state_regime_separation": "Which learned state best separates macro or market regimes?",
    }
    return questions.get(scenario, scenario.replace("_", " "))
