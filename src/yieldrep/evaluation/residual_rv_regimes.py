from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from yieldrep.config import ProjectConfig

DETAIL_COLUMNS = [
    "regime_type",
    "indicator",
    "regime",
    "country",
    "horizon_days",
    "representation",
    "model",
    "feature_count",
    "rows",
    "dates",
    "top_bottom_spread",
    "directional_hit_rate",
    "mean_rank_ic",
    "rank_ic_dates",
]

SCORECARD_COLUMNS = [
    "regime_type",
    "indicator",
    "regime",
    "country",
    "horizon_days",
    "best_by_rank_ic",
    "best_rank_ic",
    "best_by_spread",
    "best_top_bottom_spread",
    "best_learned",
    "best_learned_rank_ic",
    "best_classical",
    "best_classical_rank_ic",
    "learned_rank_ic_edge",
    "learned_beats_classical",
]

FINDINGS_COLUMNS = [
    "finding",
    "value",
    "interpretation",
    "evidence_table",
]

AUDIT_COLUMNS = [
    "audit_item",
    "group",
    "value",
    "interpretation",
    "evidence_table",
]

LEARNED_REPRESENTATIONS = {
    "autoencoder",
    "transformer",
    "graph_autoencoder",
    "graph_autoencoder_macro_market",
}
MACRO_MARKET_COLUMNS = [
    "macro_inflation",
    "macro_unemployment",
    "market_MOVE",
    "market_VIX",
]


@dataclass(frozen=True)
class ResidualRVFeatureSet:
    representation: str
    columns: list[str]


def build_residual_rv_representation_regime_report(config: ProjectConfig) -> list[Path]:
    """Evaluate residual-change RV forecasts by macro and market regime."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    detail = residual_rv_representation_regime_summary(config)
    detail.to_csv(config.residual_rv_representation_regime_table_path, index=False)
    scorecard = residual_rv_representation_regime_scorecard(detail)
    scorecard.to_csv(
        config.residual_rv_representation_regime_scorecard_table_path,
        index=False,
    )
    findings = residual_rv_representation_regime_findings(config, scorecard)
    findings.to_csv(
        config.residual_rv_representation_regime_findings_table_path,
        index=False,
    )
    audit = residual_rv_macro_benchmark_audit(config, detail, scorecard)
    audit.to_csv(config.residual_rv_macro_benchmark_audit_table_path, index=False)
    return [
        config.residual_rv_representation_regime_table_path,
        config.residual_rv_representation_regime_scorecard_table_path,
        config.residual_rv_representation_regime_findings_table_path,
        config.residual_rv_macro_benchmark_audit_table_path,
    ]


def residual_rv_representation_regime_summary(config: ProjectConfig) -> pd.DataFrame:
    """Compare frozen representations on residual-change forecasts by regime."""
    if not config.supervised_residual_change_path.exists():
        return pd.DataFrame(columns=DETAIL_COLUMNS)

    data = _attach_macro_market_features(
        pd.read_parquet(config.supervised_residual_change_path),
        config,
    )
    regimes = _regime_frames(config)
    if data.empty or not regimes:
        return pd.DataFrame(columns=DETAIL_COLUMNS)

    rows: list[pd.DataFrame] = []
    for feature_set in _residual_rv_feature_sets(config):
        available_columns = [column for column in feature_set.columns if column in data.columns]
        if not available_columns:
            continue
        predictions = _residual_change_predictions(data, feature_set, available_columns, config)
        if predictions.empty:
            continue
        rows.extend(_summaries_for_regimes(predictions, regimes))

    non_empty = [row for row in rows if not row.empty]
    if not non_empty:
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    return (
        pd.concat(non_empty, ignore_index=True)
        .loc[:, DETAIL_COLUMNS]
        .sort_values(
            [
                "regime_type",
                "indicator",
                "regime",
                "country",
                "horizon_days",
                "representation",
            ]
        )
        .reset_index(drop=True)
    )


def residual_rv_representation_regime_scorecard(
    detail: pd.DataFrame,
    min_rank_ic_dates: int = 10,
) -> pd.DataFrame:
    """Select best classical and learned representations within each regime cell."""
    if detail.empty:
        return pd.DataFrame(columns=SCORECARD_COLUMNS)

    rows: list[dict[str, object]] = []
    group_columns = ["regime_type", "indicator", "regime", "country", "horizon_days"]
    for group_values, group in detail.groupby(group_columns, sort=True):
        valid_rank = group.dropna(subset=["mean_rank_ic"]).copy()
        valid_rank = valid_rank.loc[valid_rank["rank_ic_dates"] >= min_rank_ic_dates]
        valid_spread = group.dropna(subset=["top_bottom_spread"]).copy()
        valid_spread = valid_spread.loc[valid_spread["dates"] >= min_rank_ic_dates]
        best_rank = _best_metric_row(valid_rank, "mean_rank_ic")
        best_spread = _best_metric_row(valid_spread, "top_bottom_spread")
        learned = valid_rank.loc[valid_rank["representation"].isin(LEARNED_REPRESENTATIONS)]
        classical = valid_rank.loc[~valid_rank["representation"].isin(LEARNED_REPRESENTATIONS)]
        best_learned = _best_metric_row(learned, "mean_rank_ic")
        best_classical = _best_metric_row(classical, "mean_rank_ic")
        learned_edge = _rank_ic_edge(best_learned, best_classical)
        rows.append(
            {
                **dict(zip(group_columns, group_values, strict=True)),
                "best_by_rank_ic": _model_label(best_rank),
                "best_rank_ic": _metric(best_rank, "mean_rank_ic"),
                "best_by_spread": _model_label(best_spread),
                "best_top_bottom_spread": _metric(best_spread, "top_bottom_spread"),
                "best_learned": _model_label(best_learned),
                "best_learned_rank_ic": _metric(best_learned, "mean_rank_ic"),
                "best_classical": _model_label(best_classical),
                "best_classical_rank_ic": _metric(best_classical, "mean_rank_ic"),
                "learned_rank_ic_edge": learned_edge,
                "learned_beats_classical": (
                    bool(learned_edge > 0.0) if np.isfinite(learned_edge) else False
                ),
            }
        )

    return pd.DataFrame(rows, columns=SCORECARD_COLUMNS).reset_index(drop=True)


def residual_rv_representation_regime_findings(
    config: ProjectConfig,
    scorecard: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize the regime-conditioned residual RV scorecard."""
    if scorecard is None:
        if not config.residual_rv_representation_regime_scorecard_table_path.exists():
            return pd.DataFrame(columns=FINDINGS_COLUMNS)
        scorecard = pd.read_csv(config.residual_rv_representation_regime_scorecard_table_path)
    if scorecard.empty:
        return pd.DataFrame(columns=FINDINGS_COLUMNS)

    valid = scorecard.dropna(subset=["best_rank_ic"]).copy()
    learned_wins = int(valid["learned_beats_classical"].sum()) if not valid.empty else 0
    strongest_edge = _strongest_learned_edge(valid)
    best_counts = valid["best_by_rank_ic"].value_counts().to_dict()
    rows = [
        {
            "finding": "valid_regime_cells",
            "value": f"{len(valid)} of {len(scorecard)}",
            "interpretation": "Cells are counted only when the scorecard has enough rank-IC dates after the coverage filter.",
            "evidence_table": str(config.residual_rv_representation_regime_scorecard_table_path),
        },
        {
            "finding": "learned_win_count",
            "value": f"{learned_wins} of {len(valid)}",
            "interpretation": "Learned representations beat the best classical feature set in a minority of regime cells.",
            "evidence_table": str(config.residual_rv_representation_regime_scorecard_table_path),
        },
        {
            "finding": "best_overall_counts",
            "value": _format_counts(best_counts),
            "interpretation": "Raw Nelson-Siegel residuals are the main RV feature; macro-enhanced rows show whether public macro/market variables add incremental information.",
            "evidence_table": str(config.residual_rv_representation_regime_scorecard_table_path),
        },
        {
            "finding": "strongest_learned_edge",
            "value": _edge_label(strongest_edge),
            "interpretation": "The clearest learned edge currently comes from graph-AE states in euro-area 20-day inflation-conditioned RV.",
            "evidence_table": str(config.residual_rv_representation_regime_scorecard_table_path),
        },
        {
            "finding": "phase_conclusion",
            "value": _phase_conclusion(valid),
            "interpretation": "Macro-enhanced rows should be interpreted as incremental public-information benchmarks, not as new target definitions.",
            "evidence_table": str(config.residual_rv_representation_regime_scorecard_table_path),
        },
    ]
    return pd.DataFrame(rows, columns=FINDINGS_COLUMNS)


def residual_rv_macro_benchmark_audit(
    config: ProjectConfig,
    detail: pd.DataFrame | None = None,
    scorecard: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Audit feature coverage and winner patterns for macro-enhanced RV benchmarks."""
    detail = _read_optional_csv(
        detail,
        config.residual_rv_representation_regime_table_path,
        DETAIL_COLUMNS,
    )
    scorecard = _read_optional_csv(
        scorecard,
        config.residual_rv_representation_regime_scorecard_table_path,
        SCORECARD_COLUMNS,
    )
    if detail.empty or scorecard.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)

    evidence = str(config.residual_rv_representation_regime_scorecard_table_path)
    valid_scorecard = scorecard.dropna(subset=["best_rank_ic"]).copy()
    rows = [
        {
            "audit_item": "feature_sets_compared",
            "group": "all",
            "value": _format_counts(detail["representation"].value_counts().to_dict()),
            "interpretation": "Counts are regime-summary rows by representation after macro/market feature attachment.",
            "evidence_table": str(config.residual_rv_representation_regime_table_path),
        },
        {
            "audit_item": "macro_feature_availability",
            "group": "macro_market",
            "value": _feature_count_by_country(detail, "macro_market"),
            "interpretation": "CA and EA use inflation plus market variables; US uses inflation, unemployment, and market variables.",
            "evidence_table": str(config.residual_rv_representation_regime_table_path),
        },
        {
            "audit_item": "macro_feature_availability",
            "group": "graph_autoencoder_macro_market",
            "value": _feature_count_by_country(detail, "graph_autoencoder_macro_market"),
            "interpretation": "Graph-AE macro rows use graph state, maturity interactions, and the available macro/market inputs by country.",
            "evidence_table": str(config.residual_rv_representation_regime_table_path),
        },
        {
            "audit_item": "valid_regime_cells",
            "group": "all",
            "value": f"{len(valid_scorecard)} of {len(scorecard)}",
            "interpretation": "Valid cells have enough rank-IC dates after the scorecard coverage filter.",
            "evidence_table": evidence,
        },
        {
            "audit_item": "best_rank_ic_winner_counts",
            "group": "all",
            "value": _format_counts(valid_scorecard["best_by_rank_ic"].value_counts().to_dict()),
            "interpretation": "Winner counts show whether macro-enhanced rows dominate or only improve selected cells.",
            "evidence_table": evidence,
        },
        {
            "audit_item": "macro_market_alone_wins",
            "group": "macro_market",
            "value": str(_winner_count(scorecard, "macro_market/ridge")),
            "interpretation": "Macro/market variables alone are checked separately from curve and learned state features.",
            "evidence_table": evidence,
        },
        {
            "audit_item": "residual_macro_uplift",
            "group": "residual_macro_market_vs_residual",
            "value": _uplift_summary(detail, "residual_macro_market", "residual"),
            "interpretation": "This checks whether macro inputs improve the raw Nelson-Siegel residual feature.",
            "evidence_table": str(config.residual_rv_representation_regime_table_path),
        },
        {
            "audit_item": "graph_macro_uplift",
            "group": "graph_autoencoder_macro_market_vs_graph_autoencoder",
            "value": _uplift_summary(
                detail,
                "graph_autoencoder_macro_market",
                "graph_autoencoder",
            ),
            "interpretation": "This checks whether macro inputs improve graph-AE residual RV forecasts.",
            "evidence_table": str(config.residual_rv_representation_regime_table_path),
        },
        {
            "audit_item": "graph_macro_winner_cells",
            "group": "graph_autoencoder_macro_market",
            "value": _winner_cells(scorecard, "graph_autoencoder_macro_market/ridge"),
            "interpretation": "These are the regime cells where graph-AE plus macro/market inputs is the best rank-IC model.",
            "evidence_table": evidence,
        },
    ]
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def _residual_rv_feature_sets(config: ProjectConfig) -> list[ResidualRVFeatureSet]:
    return [
        ResidualRVFeatureSet(
            "curve",
            [
                "level",
                "slope_10y_2y",
                "curvature_2s5s10s",
                "front_slope_2y_1y",
                "long_slope_30y_10y",
            ],
        ),
        ResidualRVFeatureSet("residual", ["residual"]),
        ResidualRVFeatureSet("macro_market", MACRO_MARKET_COLUMNS),
        ResidualRVFeatureSet("residual_macro_market", ["residual", *MACRO_MARKET_COLUMNS]),
        ResidualRVFeatureSet(
            "carry_roll",
            [
                "carry_1m",
                "roll_down_1m",
                "carry_3m",
                "roll_down_3m",
                "carry_12m",
                "roll_down_12m",
            ],
        ),
        ResidualRVFeatureSet(
            "autoencoder",
            [f"AE{i}" for i in range(1, config.autoencoder.latent_dim + 1)],
        ),
        ResidualRVFeatureSet(
            "transformer",
            [f"TE{i}" for i in range(1, config.transformer.latent_dim + 1)],
        ),
        ResidualRVFeatureSet(
            "graph_autoencoder",
            _graph_autoencoder_residual_features(config),
        ),
        ResidualRVFeatureSet(
            "graph_autoencoder_macro_market",
            [*_graph_autoencoder_residual_features(config), *MACRO_MARKET_COLUMNS],
        ),
    ]


def _read_optional_csv(
    frame: pd.DataFrame | None,
    path: Path,
    columns: list[str],
) -> pd.DataFrame:
    if frame is not None:
        return frame
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)


def _feature_count_by_country(detail: pd.DataFrame, representation: str) -> str:
    selected = detail.loc[detail["representation"] == representation]
    if selected.empty:
        return ""
    parts = []
    for country, group in selected.groupby("country", sort=True):
        counts = sorted(group["feature_count"].dropna().astype(int).unique())
        parts.append(f"{country}: {','.join(str(count) for count in counts)}")
    return "; ".join(parts)


def _winner_count(scorecard: pd.DataFrame, label: str) -> int:
    return int(scorecard["best_by_rank_ic"].eq(label).sum())


def _winner_cells(scorecard: pd.DataFrame, label: str) -> str:
    winners = scorecard.loc[scorecard["best_by_rank_ic"] == label].copy()
    if winners.empty:
        return "none"
    return "; ".join(
        f"{row.country} {int(row.horizon_days)}d {row.regime_type}:{row.indicator}={row.regime}"
        for row in winners.itertuples(index=False)
    )


def _uplift_summary(detail: pd.DataFrame, enhanced: str, base: str) -> str:
    group_columns = ["regime_type", "indicator", "regime", "country", "horizon_days"]
    selected = detail.loc[detail["representation"].isin([enhanced, base])].copy()
    selected = selected.loc[selected["rank_ic_dates"] >= 10]
    if selected.empty:
        return "0 of 0"

    wide = selected.pivot_table(
        index=group_columns,
        columns="representation",
        values="mean_rank_ic",
        aggfunc="first",
    ).dropna(subset=[enhanced, base])
    if wide.empty:
        return "0 of 0"

    edge = wide[enhanced] - wide[base]
    wins = int((edge > 0.0).sum())
    return f"{wins} of {len(edge)}; mean_edge={float(edge.mean()):.3f}"


def _attach_macro_market_features(data: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = _attach_macro_values(frame, config)
    frame = _attach_market_values(frame, config)
    return frame


def _attach_macro_values(data: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    if not config.macro_indicators_path.exists():
        return data

    macro = pd.read_parquet(config.macro_indicators_path)
    if macro.empty:
        return data

    result = data.copy()
    macro = macro.loc[:, ["date", "country", "indicator", "value"]].copy()
    macro["date"] = pd.to_datetime(macro["date"])
    for (country, indicator), group in macro.groupby(["country", "indicator"], sort=True):
        feature_name = f"macro_{indicator}"
        country_rows = result.loc[result["country"] == country].copy()
        if country_rows.empty:
            continue
        aligned = pd.merge_asof(
            country_rows.sort_values("date"),
            group.sort_values("date").loc[:, ["date", "country", "value"]],
            on="date",
            by="country",
            direction="backward",
        ).rename(columns={"value": feature_name})
        result = result.merge(
            aligned.loc[:, ["date", "country", "maturity_years", "horizon_days", feature_name]],
            on=["date", "country", "maturity_years", "horizon_days"],
            how="left",
        )
    return result


def _attach_market_values(data: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    if not config.market_indicators_path.exists():
        return data

    market = pd.read_parquet(config.market_indicators_path)
    if market.empty:
        return data

    result = data.copy()
    market = market.loc[:, ["date", "indicator", "value"]].copy()
    market["date"] = pd.to_datetime(market["date"])
    for indicator, group in market.groupby("indicator", sort=True):
        feature_name = f"market_{indicator}"
        aligned = pd.merge_asof(
            result.loc[:, ["date"]].drop_duplicates().sort_values("date"),
            group.sort_values("date").loc[:, ["date", "value"]],
            on="date",
            direction="backward",
        ).rename(columns={"value": feature_name})
        result = result.merge(aligned, on="date", how="left")
    return result


def _strongest_learned_edge(scorecard: pd.DataFrame) -> pd.Series:
    learned = scorecard.loc[
        scorecard["best_learned"].str.split("/", expand=True)[0].isin(LEARNED_REPRESENTATIONS)
    ].copy()
    learned = learned.dropna(subset=["learned_rank_ic_edge"])
    learned = learned.loc[learned["learned_rank_ic_edge"] > 0.0]
    if learned.empty:
        return pd.Series(dtype=object)
    return learned.sort_values(
        ["learned_rank_ic_edge", "best_learned_rank_ic"],
        ascending=[False, False],
    ).iloc[0]


def _format_counts(counts: dict[object, int]) -> str:
    if not counts:
        return ""
    return "; ".join(f"{key}: {value}" for key, value in counts.items())


def _edge_label(row: pd.Series) -> str:
    if row.empty:
        return ""
    return (
        f"{row['best_learned']} beats {row['best_classical']} by "
        f"{float(row['learned_rank_ic_edge']):.3f} rank IC "
        f"({row['country']} {int(row['horizon_days'])}d "
        f"{row['regime_type']}:{row['indicator']}={row['regime']})"
    )


def _phase_conclusion(scorecard: pd.DataFrame) -> str:
    if scorecard.empty:
        return ""
    best_counts = scorecard["best_by_rank_ic"].value_counts()
    best_model = str(best_counts.index[0]) if not best_counts.empty else ""
    macro_wins = int(
        scorecard["best_by_rank_ic"].astype(str).str.contains("macro_market", regex=False).sum()
    )
    graph_wins = int(
        scorecard["best_by_rank_ic"].astype(str).str.startswith("graph_autoencoder", na=False).sum()
    )
    if macro_wins:
        return (
            f"{best_model}_leads_overall_with_{macro_wins}_macro_enhanced_wins_"
            f"and_{graph_wins}_graph_wins"
        )
    return f"{best_model}_leads_overall_with_{graph_wins}_graph_wins"


def _graph_autoencoder_residual_features(config: ProjectConfig) -> list[str]:
    graph_features = [f"GE{i}" for i in range(1, config.gnn.latent_dim + 1)]
    maturity_features = ["maturity", "maturity_squared", "log_maturity"]
    interactions = [
        f"{graph_feature}_x_{maturity_feature}"
        for graph_feature in graph_features
        for maturity_feature in maturity_features
    ]
    return [*graph_features, *maturity_features, *interactions]


def _residual_change_predictions(
    data: pd.DataFrame,
    feature_set: ResidualRVFeatureSet,
    columns: list[str],
    config: ProjectConfig,
) -> pd.DataFrame:
    required = [
        "date",
        "country",
        "horizon_days",
        "maturity_years",
        "split_method",
        "window_id",
        "split",
        "target_residual_change",
    ]
    sample = data.dropna(subset=required).loc[:, [*required, *columns]].copy()
    rows: list[pd.DataFrame] = []
    group_columns = ["country", "horizon_days", "split_method", "window_id"]
    for group_values, group in sample.groupby(group_columns, sort=True):
        local_columns = [column for column in columns if group[column].notna().any()]
        if not local_columns:
            continue
        group = group.dropna(subset=local_columns)
        train = group.loc[group["split"] == "train"]
        test = group.loc[group["split"] == "test"]
        if train.empty or test.empty:
            continue

        x_train = train[local_columns].to_numpy(dtype=float)
        y_train = train["target_residual_change"].to_numpy(dtype=float)
        x_test = test[local_columns].to_numpy(dtype=float)
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=config.evaluation.ridge_alpha),
        )
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        rows.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(test["date"]).to_numpy(),
                    "country": test["country"].to_numpy(),
                    "horizon_days": test["horizon_days"].to_numpy(dtype=int),
                    "maturity_years": test["maturity_years"].to_numpy(dtype=float),
                    "target_residual_change": test["target_residual_change"].to_numpy(dtype=float),
                    "prediction": prediction,
                    "representation": feature_set.representation,
                    "model": "ridge",
                    "feature_count": len(local_columns),
                }
            )
        )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _regime_frames(config: ProjectConfig) -> list[pd.DataFrame]:
    frames = []
    if config.macro_regimes_path.exists():
        macro = pd.read_parquet(config.macro_regimes_path)
        if not macro.empty:
            frames.append(
                macro.loc[:, ["date", "country", "indicator", "macro_regime"]]
                .rename(columns={"macro_regime": "regime"})
                .assign(regime_type="macro")
            )
    if config.market_regimes_path.exists():
        market = pd.read_parquet(config.market_regimes_path)
        if not market.empty:
            frames.append(
                market.loc[:, ["date", "indicator", "market_vol_regime"]]
                .rename(columns={"market_vol_regime": "regime"})
                .assign(country="", regime_type="market")
            )
    return frames


def _summaries_for_regimes(
    predictions: pd.DataFrame,
    regimes: list[pd.DataFrame],
) -> list[pd.DataFrame]:
    rows = []
    for regime_frame in regimes:
        regime_type = str(regime_frame["regime_type"].iloc[0])
        joined = (
            _attach_macro_regimes(predictions, regime_frame)
            if regime_type == "macro"
            else _attach_market_regimes(predictions, regime_frame)
        )
        if joined.empty:
            continue
        rows.append(_prediction_summary(joined, regime_type=regime_type))
    return rows


def _attach_macro_regimes(predictions: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    frames = []
    regime_dates = regimes.copy()
    regime_dates["date"] = pd.to_datetime(regime_dates["date"])
    prediction_dates = predictions.copy()
    prediction_dates["date"] = pd.to_datetime(prediction_dates["date"])
    for (country, indicator), group in regime_dates.groupby(["country", "indicator"], sort=True):
        country_predictions = prediction_dates.loc[prediction_dates["country"] == country]
        if country_predictions.empty:
            continue
        joined = pd.merge_asof(
            country_predictions.sort_values("date"),
            group.sort_values("date").loc[:, ["date", "country", "indicator", "regime"]],
            on="date",
            by="country",
            direction="backward",
        ).dropna(subset=["regime"])
        if not joined.empty:
            joined["indicator"] = indicator
            frames.append(joined)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _attach_market_regimes(predictions: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    frames = []
    regime_dates = regimes.copy()
    regime_dates["date"] = pd.to_datetime(regime_dates["date"])
    prediction_dates = predictions.copy()
    prediction_dates["date"] = pd.to_datetime(prediction_dates["date"])
    for indicator, group in regime_dates.groupby("indicator", sort=True):
        joined = pd.merge_asof(
            prediction_dates.sort_values("date"),
            group.sort_values("date").loc[:, ["date", "indicator", "regime"]],
            on="date",
            direction="backward",
        ).dropna(subset=["regime"])
        if not joined.empty:
            joined["indicator"] = indicator
            frames.append(joined)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _prediction_summary(joined: pd.DataFrame, regime_type: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = [
        "indicator",
        "regime",
        "country",
        "horizon_days",
        "representation",
        "model",
        "feature_count",
    ]
    for group_values, group in joined.groupby(group_columns, sort=True):
        rank_ic = _mean_rank_ic(group)
        rows.append(
            {
                "regime_type": regime_type,
                **dict(zip(group_columns, group_values, strict=True)),
                "rows": len(group),
                "dates": group["date"].nunique(),
                "top_bottom_spread": _top_bottom_spread(group),
                "directional_hit_rate": _directional_hit_rate(group),
                "mean_rank_ic": rank_ic["mean_rank_ic"],
                "rank_ic_dates": rank_ic["rank_ic_dates"],
            }
        )
    return pd.DataFrame(rows, columns=DETAIL_COLUMNS)


def _top_bottom_spread(group: pd.DataFrame, quantile: float = 0.3) -> float:
    spreads = []
    for _, date_group in group.groupby("date", sort=True):
        if len(date_group) < 4 or date_group["prediction"].nunique() < 2:
            continue
        lower = date_group["prediction"].quantile(quantile)
        upper = date_group["prediction"].quantile(1.0 - quantile)
        bottom = date_group.loc[date_group["prediction"] <= lower, "target_residual_change"]
        top = date_group.loc[date_group["prediction"] >= upper, "target_residual_change"]
        if bottom.empty or top.empty:
            continue
        spreads.append(float(top.mean() - bottom.mean()))
    return float(np.mean(spreads)) if spreads else float("nan")


def _directional_hit_rate(group: pd.DataFrame) -> float:
    actual = group["target_residual_change"].to_numpy(dtype=float)
    predicted = group["prediction"].to_numpy(dtype=float)
    non_zero = (actual != 0.0) & (predicted != 0.0)
    if not bool(non_zero.any()):
        return float("nan")
    return float(np.mean(np.sign(actual[non_zero]) == np.sign(predicted[non_zero])))


def _mean_rank_ic(group: pd.DataFrame) -> dict[str, float | int]:
    correlations = []
    for _, date_group in group.groupby("date", sort=True):
        if len(date_group) < 3 or date_group["prediction"].nunique() < 2:
            continue
        correlation = (
            date_group["prediction"].rank().corr(date_group["target_residual_change"].rank())
        )
        if pd.notna(correlation):
            correlations.append(float(correlation))
    return {
        "mean_rank_ic": float(np.mean(correlations)) if correlations else float("nan"),
        "rank_ic_dates": len(correlations),
    }


def _best_metric_row(frame: pd.DataFrame, metric: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    return frame.sort_values(
        [metric, "directional_hit_rate", "representation"],
        ascending=[False, False, True],
    ).iloc[0]


def _model_label(row: pd.Series) -> str:
    if row.empty:
        return ""
    return f"{row['representation']}/{row['model']}"


def _metric(row: pd.Series, column: str) -> float:
    return float(row[column]) if column in row.index and pd.notna(row[column]) else float("nan")


def _rank_ic_edge(best_learned: pd.Series, best_classical: pd.Series) -> float:
    learned = _metric(best_learned, "mean_rank_ic")
    classical = _metric(best_classical, "mean_rank_ic")
    if not np.isfinite(learned) or not np.isfinite(classical):
        return float("nan")
    return learned - classical
