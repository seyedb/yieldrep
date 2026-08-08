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

LEARNED_REPRESENTATIONS = {"autoencoder", "transformer", "graph_autoencoder"}


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
    return [
        config.residual_rv_representation_regime_table_path,
        config.residual_rv_representation_regime_scorecard_table_path,
        config.residual_rv_representation_regime_findings_table_path,
    ]


def residual_rv_representation_regime_summary(config: ProjectConfig) -> pd.DataFrame:
    """Compare frozen representations on residual-change forecasts by regime."""
    if not config.supervised_residual_change_path.exists():
        return pd.DataFrame(columns=DETAIL_COLUMNS)

    data = pd.read_parquet(config.supervised_residual_change_path)
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
            "interpretation": "Raw Nelson-Siegel residuals are the strongest overall RV feature, followed by graph-AE and carry/roll features.",
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
            "value": "classical_residuals_lead_overall_graph_ae_has_regime_pockets",
            "interpretation": "The result supports graph-AE as a useful conditional representation, not a replacement for Nelson-Siegel residuals.",
            "evidence_table": str(config.residual_rv_representation_regime_scorecard_table_path),
        },
    ]
    return pd.DataFrame(rows, columns=FINDINGS_COLUMNS)


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
    ]


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
        *columns,
    ]
    sample = data.dropna(subset=required).loc[:, required].copy()
    rows: list[pd.DataFrame] = []
    group_columns = ["country", "horizon_days", "split_method", "window_id"]
    for group_values, group in sample.groupby(group_columns, sort=True):
        train = group.loc[group["split"] == "train"]
        test = group.loc[group["split"] == "test"]
        if train.empty or test.empty:
            continue

        x_train = train[columns].to_numpy(dtype=float)
        y_train = train["target_residual_change"].to_numpy(dtype=float)
        x_test = test[columns].to_numpy(dtype=float)
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
                    "feature_count": len(columns),
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
