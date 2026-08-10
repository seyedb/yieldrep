from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from yieldrep.config import ProjectConfig


LEARNED_REPRESENTATIONS = {"autoencoder", "transformer", "graph_autoencoder"}
RMSE_MATERIAL_RELATIVE_GAP = 0.005
CLASSIFICATION_MATERIAL_GAP = 0.02
SCORECARD_COLUMNS = [
    "scenario",
    "question",
    "best_representation",
    "best_model",
    "primary_metric",
    "best_value",
    "best_classical_representation",
    "best_classical_model",
    "best_classical_value",
    "best_learned_representation",
    "best_learned_model",
    "best_learned_value",
    "best_learned_rank",
    "learned_gap_to_best",
    "learned_pct_gap_to_best",
    "learned_improvement_vs_train_mean",
    "materiality_flag",
    "evidence_table",
    "interpretation",
]
RESEARCH_CHECKPOINT_COLUMNS = [
    "task",
    "target",
    "primary_metric",
    "best_classical",
    "best_classical_value",
    "best_learned",
    "best_learned_value",
    "result",
    "conclusion",
    "evidence_table",
]
RESIDUAL_RV_RESULT_COLUMNS = [
    "country",
    "horizon_days",
    "forecast_method",
    "forecast_rmse",
    "forecast_mae",
    "regime_rank_ic_method",
    "regime_rank_ic",
    "regime_spread_method",
    "regime_top_bottom_spread",
    "evidence_tables",
]


def build_representation_task_scorecard(config: ProjectConfig) -> Path:
    """Write a task-level scorecard across classical and learned representations."""
    build_representation_results(config)
    return config.representation_task_scorecard_table_path


def build_research_checkpoint_scorecard(config: ProjectConfig) -> Path:
    """Write a compact research checkpoint table for the current project state."""
    build_research_summary(config)
    return config.research_checkpoint_scorecard_table_path


def build_representation_results(config: ProjectConfig) -> Path:
    """Write task-level representation results across the current research tasks."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    results = representation_task_scorecard(config)
    results.to_csv(config.representation_results_table_path, index=False)
    results.to_csv(config.representation_task_scorecard_table_path, index=False)
    return config.representation_results_table_path


def build_research_summary(config: ProjectConfig) -> Path:
    """Write the compact project-level research summary."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    summary = research_checkpoint_scorecard(config)
    summary.to_csv(config.research_summary_table_path, index=False)
    summary.to_csv(config.research_checkpoint_scorecard_table_path, index=False)
    return config.research_summary_table_path


def build_residual_rv_results(config: ProjectConfig) -> Path:
    """Write a compact country/horizon summary for residual relative-value results."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    results = residual_rv_results(config)
    results.to_csv(config.residual_rv_results_table_path, index=False)
    return config.residual_rv_results_table_path


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


def research_checkpoint_scorecard(config: ProjectConfig) -> pd.DataFrame:
    task_scorecard = _current_task_scorecard(config)
    rows = [
        _checkpoint_row_from_task(
            task_scorecard,
            scenario="clean_reconstruction",
            task="Clean reconstruction",
            target="Observed zero-coupon curve",
        ),
        _checkpoint_row_from_task(
            task_scorecard,
            scenario="masked_maturity_reconstruction",
            task="Masked maturity reconstruction",
            target="Held-out maturity yields",
        ),
        _residual_rv_checkpoint_row(config),
        _checkpoint_row_from_task(
            task_scorecard,
            scenario="volatility_regime_classification",
            task="Volatility regime classification",
            target="Future curve-volatility regime",
        ),
        _macro_conditioned_rv_checkpoint_row(config),
    ]
    return pd.DataFrame(rows).loc[:, RESEARCH_CHECKPOINT_COLUMNS]


def residual_rv_results(config: ProjectConfig) -> pd.DataFrame:
    keys = _residual_rv_result_keys(config)
    if keys.empty:
        return pd.DataFrame(columns=RESIDUAL_RV_RESULT_COLUMNS)

    forecast = _residual_forecast_results(config)
    rank_ic = _residual_regime_rank_ic_results(config)
    spread = _residual_regime_spread_results(config)

    results = keys.merge(forecast, on=["country", "horizon_days"], how="left")
    results = results.merge(rank_ic, on=["country", "horizon_days"], how="left")
    results = results.merge(spread, on=["country", "horizon_days"], how="left")
    evidence_tables = [
        str(path)
        for path in [
            config.supervised_forecast_rank_table_path,
            config.residual_rv_representation_regime_scorecard_table_path,
            config.residual_rv_subperiod_results_table_path,
            config.residual_rv_maturity_bucket_results_table_path,
            config.residual_rv_feature_importance_table_path,
        ]
        if path.exists()
    ]
    results["evidence_tables"] = "; ".join(evidence_tables)
    return results.loc[:, RESIDUAL_RV_RESULT_COLUMNS].sort_values(["country", "horizon_days"])


def _residual_rv_result_keys(config: ProjectConfig) -> pd.DataFrame:
    frames = []
    if config.supervised_forecast_rank_table_path.exists():
        forecast = pd.read_csv(config.supervised_forecast_rank_table_path)
        if {"target", "country", "horizon_days"}.issubset(forecast.columns):
            frames.append(
                forecast.loc[
                    forecast["target"] == "residual_change",
                    ["country", "horizon_days"],
                ]
            )
    if config.residual_rv_representation_regime_scorecard_table_path.exists():
        regime = pd.read_csv(config.residual_rv_representation_regime_scorecard_table_path)
        if {"country", "horizon_days"}.issubset(regime.columns):
            frames.append(regime.loc[:, ["country", "horizon_days"]])
    if not frames:
        return pd.DataFrame(columns=["country", "horizon_days"])
    keys = pd.concat(frames, ignore_index=True).drop_duplicates()
    return keys.sort_values(["country", "horizon_days"]).reset_index(drop=True)


def _residual_forecast_results(config: ProjectConfig) -> pd.DataFrame:
    table = config.supervised_forecast_rank_table_path
    columns = ["country", "horizon_days", "forecast_method", "forecast_rmse", "forecast_mae"]
    if not table.exists():
        return pd.DataFrame(columns=columns)
    data = pd.read_csv(table)
    required = {
        "target",
        "country",
        "horizon_days",
        "representation",
        "model",
        "mean_rmse",
        "mean_mae",
    }
    if not required.issubset(data.columns):
        return pd.DataFrame(columns=columns)
    residual = data.loc[data["target"] == "residual_change"].copy()
    if residual.empty:
        return pd.DataFrame(columns=columns)
    residual = residual.sort_values(
        ["country", "horizon_days", "mean_rmse", "mean_mae", "representation", "model"]
    )
    best = residual.groupby(["country", "horizon_days"], as_index=False).first()
    best["forecast_method"] = best.apply(
        lambda row: _method_label(row["representation"], row["model"]),
        axis=1,
    )
    return best.rename(columns={"mean_rmse": "forecast_rmse", "mean_mae": "forecast_mae"}).loc[
        :, columns
    ]


def _residual_regime_rank_ic_results(config: ProjectConfig) -> pd.DataFrame:
    table = config.residual_rv_representation_regime_scorecard_table_path
    columns = ["country", "horizon_days", "regime_rank_ic_method", "regime_rank_ic"]
    if not table.exists():
        return pd.DataFrame(columns=columns)
    data = pd.read_csv(table)
    required = {"country", "horizon_days", "best_by_rank_ic", "best_rank_ic"}
    if not required.issubset(data.columns):
        return pd.DataFrame(columns=columns)
    data = data.dropna(subset=["best_rank_ic"]).copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        data.groupby(["country", "horizon_days", "best_by_rank_ic"], as_index=False)["best_rank_ic"]
        .mean()
        .sort_values(["country", "horizon_days", "best_rank_ic"], ascending=[True, True, False])
    )
    best = grouped.groupby(["country", "horizon_days"], as_index=False).first()
    return best.rename(
        columns={
            "best_by_rank_ic": "regime_rank_ic_method",
            "best_rank_ic": "regime_rank_ic",
        }
    ).loc[:, columns]


def _residual_regime_spread_results(config: ProjectConfig) -> pd.DataFrame:
    table = config.residual_rv_representation_regime_scorecard_table_path
    columns = [
        "country",
        "horizon_days",
        "regime_spread_method",
        "regime_top_bottom_spread",
    ]
    if not table.exists():
        return pd.DataFrame(columns=columns)
    data = pd.read_csv(table)
    required = {"country", "horizon_days", "best_by_spread", "best_top_bottom_spread"}
    if not required.issubset(data.columns):
        return pd.DataFrame(columns=columns)
    data = data.dropna(subset=["best_top_bottom_spread"]).copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        data.groupby(["country", "horizon_days", "best_by_spread"], as_index=False)[
            "best_top_bottom_spread"
        ]
        .mean()
        .sort_values(
            ["country", "horizon_days", "best_top_bottom_spread"],
            ascending=[True, True, False],
        )
    )
    best = grouped.groupby(["country", "horizon_days"], as_index=False).first()
    return best.rename(
        columns={
            "best_by_spread": "regime_spread_method",
            "best_top_bottom_spread": "regime_top_bottom_spread",
        }
    ).loc[:, columns]


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
    classical = _best_classical(data, metric="rmse", ascending=True)
    materiality = _materiality_for_lower_is_better(
        best_representation=str(best["representation"]),
        best_value=float(best["rmse"]),
        learned_value=learned.get("best_learned_value"),
        classical_value=classical.get("best_classical_value"),
        train_mean_value=np.nan,
        learned_rank=learned.get("best_learned_rank"),
        learned_only=task == "masked_maturity_reconstruction",
    )
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
        **classical,
        **learned,
        **_learned_gap_columns(
            best_value=float(best["rmse"]),
            learned_value=learned.get("best_learned_value"),
            lower_is_better=True,
        ),
        "learned_improvement_vs_train_mean": np.nan,
        "materiality_flag": materiality,
        "interpretation": _interpret_learned_rank(
            learned.get("best_learned_rank"),
            best_representation=str(best["representation"]),
            materiality_flag=materiality,
        ),
    }


def _current_task_scorecard(config: ProjectConfig) -> pd.DataFrame:
    if config.representation_task_scorecard_table_path.exists():
        return pd.read_csv(config.representation_task_scorecard_table_path)
    return representation_task_scorecard(config)


def _checkpoint_row_from_task(
    task_scorecard: pd.DataFrame,
    scenario: str,
    task: str,
    target: str,
) -> dict[str, object]:
    if task_scorecard.empty:
        return _empty_checkpoint_row(task, target, evidence_table="")

    rows = task_scorecard.loc[task_scorecard["scenario"] == scenario]
    if rows.empty:
        return _empty_checkpoint_row(task, target, evidence_table="")

    row = rows.iloc[0]
    return {
        "task": task,
        "target": target,
        "primary_metric": row["primary_metric"],
        "best_classical": _method_label(
            row.get("best_classical_representation"),
            row.get("best_classical_model"),
        ),
        "best_classical_value": row.get("best_classical_value"),
        "best_learned": _method_label(
            row.get("best_learned_representation"),
            row.get("best_learned_model"),
        ),
        "best_learned_value": row.get("best_learned_value"),
        "result": _checkpoint_result(row),
        "conclusion": row["interpretation"],
        "evidence_table": row["evidence_table"],
    }


def _residual_rv_checkpoint_row(config: ProjectConfig) -> dict[str, object]:
    task_scorecard = _current_task_scorecard(config)
    base = _checkpoint_row_from_task(
        task_scorecard,
        scenario="residual_relative_value",
        task="Residual relative value",
        target="Nelson-Siegel residual convergence",
    )
    base["primary_metric"] = "spread_t_stat"
    best_classical = base.get("best_classical")
    if not pd.isna(best_classical):
        base["best_classical"] = _residual_method_label(str(best_classical))
    result = str(base.get("result", ""))
    if "selected=" in result:
        base["result"] = result.replace(
            str(best_classical), _residual_method_label(str(best_classical)), 1
        )
    return base


def _macro_conditioned_rv_checkpoint_row(config: ProjectConfig) -> dict[str, object]:
    table = config.residual_rv_representation_regime_findings_table_path
    if not table.exists():
        return _empty_checkpoint_row(
            task="Macro/market-conditioned residual RV",
            target="Cross-sectional residual-change ranking by regime",
            evidence_table=str(table),
        )

    findings = pd.read_csv(table)
    value_by_finding = dict(zip(findings["finding"], findings["value"], strict=False))
    interpretation_by_finding = dict(
        zip(findings["finding"], findings["interpretation"], strict=False)
    )
    counts = _winner_counts(str(value_by_finding.get("best_overall_counts", "")))
    best_classical = _best_count_method(
        counts,
        learned=False,
    )
    best_learned = _best_count_method(
        counts,
        learned=True,
    )
    learned_win_count = str(value_by_finding.get("learned_win_count", ""))
    best_overall_counts = str(value_by_finding.get("best_overall_counts", ""))
    strongest_edge = str(value_by_finding.get("strongest_learned_edge", ""))
    conclusion = str(
        interpretation_by_finding.get(
            "best_overall_counts",
            "Residual and engineered classical features remain the main benchmark.",
        )
    )
    return {
        "task": "Macro/market-conditioned residual RV",
        "target": "Cross-sectional residual-change ranking by regime",
        "primary_metric": "mean_rank_ic selection count",
        "best_classical": _residual_method_label(str(best_classical[0]))
        if not pd.isna(best_classical[0])
        else best_classical[0],
        "best_classical_value": best_classical[1],
        "best_learned": best_learned[0],
        "best_learned_value": best_learned[1],
        "result": f"{learned_win_count}; {best_overall_counts}",
        "conclusion": f"{conclusion} {strongest_edge}".strip(),
        "evidence_table": str(table),
    }


def _empty_checkpoint_row(task: str, target: str, evidence_table: str) -> dict[str, object]:
    return {
        "task": task,
        "target": target,
        "primary_metric": np.nan,
        "best_classical": np.nan,
        "best_classical_value": np.nan,
        "best_learned": np.nan,
        "best_learned_value": np.nan,
        "result": "Evidence table is unavailable or empty.",
        "conclusion": "No checkpoint conclusion is available.",
        "evidence_table": evidence_table,
    }


def _checkpoint_result(row: pd.Series) -> str:
    best = _method_label(row.get("best_representation"), row.get("best_model"))
    learned = _method_label(
        row.get("best_learned_representation"),
        row.get("best_learned_model"),
    )
    classical = _method_label(
        row.get("best_classical_representation"),
        row.get("best_classical_model"),
    )
    materiality = str(row.get("materiality_flag", ""))
    if pd.isna(row.get("best_learned_value")):
        return f"selected={best}; no learned comparison"
    if pd.isna(row.get("best_classical_value")):
        return f"selected={best}; best_learned={learned}; learned-only benchmark"
    return f"selected={best}; best_classical={classical}; best_learned={learned}; {materiality}"


def _method_label(representation: object, model: object) -> object:
    if pd.isna(representation):
        return "not_applicable"
    if pd.isna(model) or str(model) == "":
        return str(representation)
    return f"{representation}/{model}"


def _residual_method_label(method: str) -> str:
    if method == "not_applicable":
        return method
    if method.startswith("nelson_siegel_residual/"):
        return method.replace("/lagged/ridge", "/lagged_ridge")
    if method == "residual/ridge":
        return "nelson_siegel_residual/ridge"
    if method == "lagged/ridge":
        return "nelson_siegel_residual/lagged_ridge"
    return method


def _winner_counts(value: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in value.split(";"):
        if ":" not in item:
            continue
        method, count = item.rsplit(":", maxsplit=1)
        method = method.strip()
        try:
            counts[method] = int(count.strip())
        except ValueError:
            continue
    return counts


def _best_count_method(counts: dict[str, int], learned: bool) -> tuple[object, object]:
    filtered = {
        method: count
        for method, count in counts.items()
        if (method.split("/", maxsplit=1)[0] in LEARNED_REPRESENTATIONS) == learned
    }
    if not filtered:
        return np.nan, np.nan
    method, count = sorted(filtered.items(), key=lambda item: (-item[1], item[0]))[0]
    return method, count


def _forecast_row(config: ProjectConfig, target: str, scenario: str) -> dict[str, object]:
    table = config.baseline_summary_table_path
    if not table.exists():
        table = config.baseline_rank_table_path
        if not table.exists():
            return _empty_row(scenario, str(table))

    target_data = pd.read_csv(table)
    target_data = target_data.loc[target_data["target"] == target].copy()
    data = target_data.loc[target_data["model"] != "train_mean"].copy()
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
    classical = _best_classical(data, metric="mean_rmse", ascending=True)
    train_mean_value = _train_mean_value(
        target_data,
        learned.get("best_learned_representation"),
    )
    learned_improvement = _learned_improvement_vs_train_mean(
        learned_value=learned.get("best_learned_value"),
        train_mean_value=train_mean_value,
        lower_is_better=True,
    )
    materiality = _materiality_for_lower_is_better(
        best_representation=str(best["representation"]),
        best_value=float(best["mean_rmse"]),
        learned_value=learned.get("best_learned_value"),
        classical_value=classical.get("best_classical_value"),
        train_mean_value=train_mean_value,
        learned_rank=learned.get("best_learned_rank"),
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
        **classical,
        **learned,
        **_learned_gap_columns(
            best_value=float(best["mean_rmse"]),
            learned_value=learned.get("best_learned_value"),
            lower_is_better=True,
        ),
        "learned_improvement_vs_train_mean": learned_improvement,
        "materiality_flag": materiality,
        "interpretation": _interpret_learned_rank(
            learned.get("best_learned_rank"),
            best_representation=str(best["representation"]),
            materiality_flag=materiality,
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
        "best_classical_representation": representation,
        "best_classical_model": model,
        "best_classical_value": float(best["spread_t_stat"]),
        "best_learned_representation": np.nan,
        "best_learned_model": np.nan,
        "best_learned_value": np.nan,
        "best_learned_rank": np.nan,
        "learned_gap_to_best": np.nan,
        "learned_pct_gap_to_best": np.nan,
        "learned_improvement_vs_train_mean": np.nan,
        "materiality_flag": "no_learned_comparison",
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
    classical = _best_classical_classification(data)
    materiality = _materiality_for_higher_is_better(
        best_representation=representation,
        best_value=float(best["best_balanced_accuracy"]),
        learned_value=learned.get("best_learned_value"),
        classical_value=classical.get("best_classical_value"),
        learned_rank=learned.get("best_learned_rank"),
        material_gap=CLASSIFICATION_MATERIAL_GAP,
    )
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
        **classical,
        **learned,
        **_learned_gap_columns(
            best_value=float(best["best_balanced_accuracy"]),
            learned_value=learned.get("best_learned_value"),
            lower_is_better=False,
        ),
        "learned_improvement_vs_train_mean": np.nan,
        "materiality_flag": materiality,
        "interpretation": _interpret_learned_rank(
            learned.get("best_learned_rank"),
            best_representation=representation,
            materiality_flag=materiality,
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
        "best_classical_representation": "nelson_siegel_residual",
        "best_classical_model": f"{best['regime_type']}:{best['indicator']}",
        "best_classical_value": float(best["abs_hit_rate_gap"]),
        "best_learned_representation": np.nan,
        "best_learned_model": np.nan,
        "best_learned_value": np.nan,
        "best_learned_rank": np.nan,
        "learned_gap_to_best": np.nan,
        "learned_pct_gap_to_best": np.nan,
        "learned_improvement_vs_train_mean": np.nan,
        "materiality_flag": "no_learned_comparison",
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
        "best_classical_representation": np.nan,
        "best_classical_model": np.nan,
        "best_classical_value": np.nan,
        "best_learned_representation": best["representation"],
        "best_learned_model": f"{best['country']} {best['regime_type']}:{best['indicator']}",
        "best_learned_value": float(best["separation_ratio"]),
        "best_learned_rank": 1.0,
        "learned_gap_to_best": 0.0,
        "learned_pct_gap_to_best": 0.0,
        "learned_improvement_vs_train_mean": np.nan,
        "materiality_flag": "diagnostic_only",
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


def _best_classical(data: pd.DataFrame, metric: str, ascending: bool) -> dict[str, object]:
    classical = data.loc[~data["representation"].isin(LEARNED_REPRESENTATIONS)].copy()
    classical = (
        classical.loc[classical["model"] != "train_mean"] if "model" in classical else classical
    )
    if classical.empty:
        return _empty_classical()

    sort_columns = [metric, "representation"]
    sort_order = [ascending, True]
    if "model" in classical.columns:
        sort_columns.append("model")
        sort_order.append(True)
    best = classical.sort_values(sort_columns, ascending=sort_order).iloc[0]
    return {
        "best_classical_representation": best["representation"],
        "best_classical_model": best["model"] if "model" in classical.columns else np.nan,
        "best_classical_value": float(best[metric]),
    }


def _best_classical_classification(data: pd.DataFrame) -> dict[str, object]:
    classical_columns = {
        "curve_vol": "curve_vol_balanced_accuracy",
        "policy": "policy_balanced_accuracy",
        "curve": "curve_balanced_accuracy",
    }
    rows = []
    for representation, column in classical_columns.items():
        if column not in data.columns:
            continue
        selected = data.dropna(subset=[column]).sort_values(column, ascending=False)
        if selected.empty:
            continue
        rows.append(
            {
                "best_classical_representation": representation,
                "best_classical_model": "logistic_l2",
                "best_classical_value": float(selected.iloc[0][column]),
            }
        )
    if not rows:
        return _empty_classical()
    return sorted(
        rows,
        key=lambda row: float(str(row["best_classical_value"])),
        reverse=True,
    )[0]


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


def _empty_classical() -> dict[str, object]:
    return {
        "best_classical_representation": np.nan,
        "best_classical_model": np.nan,
        "best_classical_value": np.nan,
    }


def _learned_gap_columns(
    best_value: float,
    learned_value: object,
    lower_is_better: bool,
) -> dict[str, float]:
    if pd.isna(learned_value):
        return {"learned_gap_to_best": np.nan, "learned_pct_gap_to_best": np.nan}

    learned_float = float(str(learned_value))
    gap = learned_float - best_value if lower_is_better else best_value - learned_float
    pct_gap = gap / abs(best_value) if best_value != 0.0 else np.nan
    return {"learned_gap_to_best": gap, "learned_pct_gap_to_best": pct_gap}


def _train_mean_value(data: pd.DataFrame, learned_representation: object) -> float:
    train_mean = data.loc[data["model"] == "train_mean"].copy()
    if train_mean.empty:
        return np.nan

    if not pd.isna(learned_representation):
        matched = train_mean.loc[train_mean["representation"] == learned_representation]
        if not matched.empty:
            return float(matched.sort_values("mean_rmse").iloc[0]["mean_rmse"])
    return float(train_mean.sort_values("mean_rmse").iloc[0]["mean_rmse"])


def _learned_improvement_vs_train_mean(
    learned_value: object,
    train_mean_value: float,
    lower_is_better: bool,
) -> float:
    if pd.isna(learned_value) or pd.isna(train_mean_value):
        return np.nan

    learned_float = float(str(learned_value))
    return train_mean_value - learned_float if lower_is_better else learned_float - train_mean_value


def _materiality_for_lower_is_better(
    best_representation: str,
    best_value: float,
    learned_value: object,
    classical_value: object,
    train_mean_value: object,
    learned_rank: object,
    learned_only: bool = False,
) -> str:
    if pd.isna(learned_value):
        return "no_learned_comparison"
    if learned_only and best_representation in LEARNED_REPRESENTATIONS:
        return "material_learned_edge"

    learned_float = float(str(learned_value))
    if best_representation in LEARNED_REPRESENTATIONS:
        if not pd.isna(train_mean_value):
            improvement = float(str(train_mean_value)) - learned_float
            relative_improvement = improvement / abs(float(str(train_mean_value)))
            if relative_improvement < RMSE_MATERIAL_RELATIVE_GAP:
                return "competitive_tie"
        if not pd.isna(classical_value):
            classical_float = float(str(classical_value))
            improvement = classical_float - learned_float
            relative_improvement = improvement / abs(classical_float)
            if relative_improvement >= RMSE_MATERIAL_RELATIVE_GAP:
                return "material_learned_edge"
        return "competitive_tie"

    if pd.isna(learned_rank):
        return "not_material"
    gap = learned_float - best_value
    pct_gap = gap / abs(best_value) if best_value != 0.0 else np.nan
    if pct_gap <= RMSE_MATERIAL_RELATIVE_GAP and float(str(learned_rank)) <= 3.0:
        return "competitive_tie"
    return "not_material"


def _materiality_for_higher_is_better(
    best_representation: str,
    best_value: float,
    learned_value: object,
    classical_value: object,
    learned_rank: object,
    material_gap: float,
) -> str:
    if pd.isna(learned_value):
        return "no_learned_comparison"

    learned_float = float(str(learned_value))
    if best_representation in LEARNED_REPRESENTATIONS:
        if not pd.isna(classical_value):
            if learned_float - float(str(classical_value)) >= material_gap:
                return "material_learned_edge"
        return "competitive_tie"

    if pd.isna(learned_rank):
        return "not_material"
    if best_value - learned_float <= material_gap and float(str(learned_rank)) <= 3.0:
        return "competitive_tie"
    return "not_material"


def _interpret_learned_rank(
    rank: object,
    best_representation: str,
    materiality_flag: str,
) -> str:
    if materiality_flag == "material_learned_edge":
        return "A learned representation has a material edge in this scenario."
    if materiality_flag == "competitive_tie":
        return "A learned representation is competitive, but the edge is small or tied."
    if materiality_flag == "not_material":
        return "Classical or engineered baselines remain stronger in this scenario."
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
