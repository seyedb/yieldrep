from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from yieldrep.config import ProjectConfig


def build_learned_state_regime_diagnostics(config: ProjectConfig) -> list[Path]:
    """Join learned states to macro/market regimes and summarize separation."""
    config.learned_states_dir.mkdir(parents=True, exist_ok=True)
    config.tables_dir.mkdir(parents=True, exist_ok=True)

    states = _read_learned_state_embeddings(config)
    regimes = _join_regimes(states, config)
    regimes.to_parquet(config.learned_state_regimes_path, index=False)

    means = learned_state_regime_means(regimes)
    means.to_csv(config.learned_state_regime_means_table_path, index=False)

    summary = learned_state_regime_summary(regimes)
    summary.to_csv(config.learned_state_regime_summary_table_path, index=False)

    return [
        config.learned_state_regimes_path,
        config.learned_state_regime_means_table_path,
        config.learned_state_regime_summary_table_path,
    ]


def learned_state_regime_means(regimes: pd.DataFrame) -> pd.DataFrame:
    """Average each latent dimension by representation, country, and regime."""
    columns = [
        "representation",
        "country",
        "regime_type",
        "indicator",
        "regime",
        "rows",
        "split_test_fraction",
        "latent_feature",
        "latent_mean",
        "latent_std",
    ]
    if regimes.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for group_values, group in regimes.groupby(
        ["representation", "country", "regime_type", "indicator", "regime"],
        sort=True,
    ):
        representation, country, regime_type, indicator, regime = group_values
        features = _latent_columns(group)
        for feature in features:
            rows.append(
                {
                    "representation": representation,
                    "country": country,
                    "regime_type": regime_type,
                    "indicator": indicator,
                    "regime": regime,
                    "rows": len(group),
                    "split_test_fraction": float(group["split"].eq("test").mean()),
                    "latent_feature": feature,
                    "latent_mean": float(group[feature].mean()),
                    "latent_std": float(group[feature].std(ddof=0)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def learned_state_regime_summary(regimes: pd.DataFrame) -> pd.DataFrame:
    """Summarize latent-state separation across low/medium/high regimes."""
    columns = [
        "representation",
        "country",
        "regime_type",
        "indicator",
        "features",
        "rows",
        "regimes",
        "high_rows",
        "low_rows",
        "high_low_distance",
        "between_trace",
        "within_trace",
        "separation_ratio",
    ]
    if regimes.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for group_values, group in regimes.groupby(
        ["representation", "country", "regime_type", "indicator"],
        sort=True,
    ):
        representation, country, regime_type, indicator = group_values
        features = _latent_columns(group)
        if not features:
            continue
        high = group.loc[group["regime"] == "high", features]
        low = group.loc[group["regime"] == "low", features]
        high_low_distance = _mean_distance(high, low)
        between_trace, within_trace = _variance_traces(group, features)
        rows.append(
            {
                "representation": representation,
                "country": country,
                "regime_type": regime_type,
                "indicator": indicator,
                "features": len(features),
                "rows": len(group),
                "regimes": group["regime"].nunique(),
                "high_rows": len(high),
                "low_rows": len(low),
                "high_low_distance": high_low_distance,
                "between_trace": between_trace,
                "within_trace": within_trace,
                "separation_ratio": (
                    between_trace / within_trace if within_trace > 0 else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _read_learned_state_embeddings(config: ProjectConfig) -> pd.DataFrame:
    frames = [
        _read_embedding_dir(config.autoencoder_dir, "autoencoder", "AE"),
        _read_embedding_dir(config.transformer_dir, "transformer", "TE"),
        _read_embedding_dir(config.gnn_dir, "graph_autoencoder", "GE"),
    ]
    non_empty = [frame for frame in frames if not frame.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


def _read_embedding_dir(path: Path, representation: str, prefix: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for file_path in sorted(path.glob("*_embeddings.parquet")):
        frame = pd.read_parquet(file_path)
        features = [column for column in frame.columns if column.startswith(prefix)]
        if not features:
            continue
        selected = frame.loc[:, ["date", "country", "split", *features]].copy()
        selected["date"] = pd.to_datetime(selected["date"])
        selected["representation"] = representation
        frames.append(selected)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _join_regimes(states: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    if states.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    if config.macro_regimes_path.exists():
        macro = pd.read_parquet(config.macro_regimes_path)
        frames.append(_join_macro_regimes(states, macro))
    if config.market_regimes_path.exists():
        market = pd.read_parquet(config.market_regimes_path)
        frames.append(_join_market_regimes(states, market))
    non_empty = [frame for frame in frames if not frame.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


def _join_macro_regimes(states: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    regimes = macro.loc[:, ["date", "country", "indicator", "macro_regime"]].copy()
    regimes["date"] = pd.to_datetime(regimes["date"])
    regimes = regimes.dropna(subset=["macro_regime"])

    frames: list[pd.DataFrame] = []
    for (country, indicator), group in regimes.groupby(["country", "indicator"], sort=True):
        state_group = states.loc[states["country"] == country].copy()
        if state_group.empty:
            continue
        joined = pd.merge_asof(
            state_group.sort_values("date"),
            group.sort_values("date"),
            on="date",
            by="country",
            direction="backward",
        )
        joined = joined.dropna(subset=["macro_regime"])
        joined["regime_type"] = "macro"
        joined["indicator"] = indicator
        joined["regime"] = joined["macro_regime"]
        frames.append(joined.drop(columns=["macro_regime"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _join_market_regimes(states: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    regimes = market.loc[:, ["date", "indicator", "market_vol_regime"]].copy()
    regimes["date"] = pd.to_datetime(regimes["date"])
    regimes = regimes.dropna(subset=["market_vol_regime"])

    frames: list[pd.DataFrame] = []
    for indicator, group in regimes.groupby("indicator", sort=True):
        joined = pd.merge_asof(
            states.sort_values("date"),
            group.sort_values("date"),
            on="date",
            direction="backward",
        )
        joined = joined.dropna(subset=["market_vol_regime"])
        joined["regime_type"] = "market"
        joined["indicator"] = indicator
        joined["regime"] = joined["market_vol_regime"]
        frames.append(joined.drop(columns=["market_vol_regime"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _latent_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.startswith("AE") or column.startswith("TE") or column.startswith("GE")
        if not frame[column].isna().all()
    ]


def _mean_distance(left: pd.DataFrame, right: pd.DataFrame) -> float:
    if left.empty or right.empty:
        return float("nan")
    difference = left.mean().to_numpy(dtype=float) - right.mean().to_numpy(dtype=float)
    return float(np.linalg.norm(difference))


def _variance_traces(group: pd.DataFrame, features: list[str]) -> tuple[float, float]:
    overall_mean = group.loc[:, features].mean().to_numpy(dtype=float)
    between_trace = 0.0
    within_trace = 0.0
    total_rows = len(group)

    for _, regime_group in group.groupby("regime", sort=True):
        values = regime_group.loc[:, features].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        mean = values.mean(axis=0)
        between_trace += len(values) * float(np.sum(np.square(mean - overall_mean)))
        within_trace += float(np.sum(np.square(values - mean)))

    return between_trace / total_rows, within_trace / total_rows
