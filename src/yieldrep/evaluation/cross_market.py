from __future__ import annotations

from pathlib import Path
from itertools import combinations

import pandas as pd

from yieldrep.config import ProjectConfig

COMPONENTS = ["PC1", "PC2", "PC3"]
NS_FACTORS = ["beta_level", "beta_slope", "beta_curvature"]
STATE_LABELS = ["low", "medium", "high"]


def build_cross_market_report(config: ProjectConfig) -> Path:
    """Write cross-market diagnostics for current curve representations."""
    config.tables_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    pca_scores = _read_country_frames(config.pca_dir, suffix="_scores.parquet")
    pca_variance = _read_country_frames(config.pca_dir, suffix="_variance.parquet")
    ns_factors = _read_country_frames(config.nelson_siegel_dir, suffix="_factors.parquet")

    rows.extend(_pca_variance_rows(pca_variance))
    rows.extend(_pca_score_correlation_rows(pca_scores))
    rows.extend(_nelson_siegel_correlation_rows(ns_factors))
    rows.extend(_curve_state_overlap_rows(pca_scores))

    report = pd.DataFrame(rows).sort_values(
        ["metric_group", "metric", "country", "country_a", "country_b"],
        na_position="last",
    )
    report.to_csv(config.cross_market_summary_table_path, index=False)
    return config.cross_market_summary_table_path


def _read_country_frames(directory: Path, suffix: str) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(directory.glob(f"*{suffix}")):
        country = path.name.removesuffix(suffix).upper()
        frame = pd.read_parquet(path)
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
        frame["country"] = country
        frames[country] = frame
    return frames


def _pca_variance_rows(frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for country, frame in frames.items():
        variance = frame.set_index("component")["explained_variance_ratio"]
        for component in COMPONENTS:
            if component in variance.index:
                rows.append(
                    _country_row(
                        metric_group="pca_variance",
                        metric=f"{component.lower()}_explained_variance",
                        country=country,
                        value=float(variance.loc[component]),
                    )
                )
        available = [component for component in COMPONENTS if component in variance.index]
        if available:
            rows.append(
                _country_row(
                    metric_group="pca_variance",
                    metric="pc1_pc3_cumulative_explained_variance",
                    country=country,
                    value=float(variance.loc[available].sum()),
                )
            )
    return rows


def _pca_score_correlation_rows(frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for country_a, country_b in combinations(sorted(frames), 2):
        merged = _merge_by_date(frames[country_a], frames[country_b], COMPONENTS)
        if merged.empty:
            continue
        for component in COMPONENTS:
            left = f"{component}_a"
            right = f"{component}_b"
            if left in merged.columns and right in merged.columns:
                corr = merged[left].corr(merged[right])
                rows.append(
                    _pair_row(
                        metric_group="pca_score_correlation",
                        metric=f"{component.lower()}_absolute_correlation",
                        country_a=country_a,
                        country_b=country_b,
                        value=float(abs(corr)),
                        observations=len(merged),
                    )
                )
    return rows


def _nelson_siegel_correlation_rows(frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for country_a, country_b in combinations(sorted(frames), 2):
        merged = _merge_by_date(frames[country_a], frames[country_b], NS_FACTORS)
        if merged.empty:
            continue
        for factor in NS_FACTORS:
            left = f"{factor}_a"
            right = f"{factor}_b"
            if left in merged.columns and right in merged.columns:
                corr = merged[left].corr(merged[right])
                rows.append(
                    _pair_row(
                        metric_group="nelson_siegel_factor_correlation",
                        metric=f"{factor}_correlation",
                        country_a=country_a,
                        country_b=country_b,
                        value=float(corr),
                        observations=len(merged),
                    )
                )
    return rows


def _curve_state_overlap_rows(frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    states = {country: _add_state_labels(frame) for country, frame in frames.items()}
    rows: list[dict[str, object]] = []
    for country_a, country_b in combinations(sorted(states), 2):
        columns = [f"{component}_state" for component in COMPONENTS]
        merged = _merge_by_date(states[country_a], states[country_b], columns)
        if merged.empty:
            continue
        for component in COMPONENTS:
            left = f"{component}_state_a"
            right = f"{component}_state_b"
            if left in merged.columns and right in merged.columns:
                overlap = (merged[left] == merged[right]).mean()
                rows.append(
                    _pair_row(
                        metric_group="curve_state_overlap",
                        metric=f"{component.lower()}_same_state_share",
                        country_a=country_a,
                        country_b=country_b,
                        value=float(overlap),
                        observations=len(merged),
                    )
                )
    return rows


def _add_state_labels(frame: pd.DataFrame) -> pd.DataFrame:
    state_frame = frame.copy()
    for component in COMPONENTS:
        if component in state_frame.columns:
            state_frame[f"{component}_state"] = _tercile_labels(state_frame[component])
    return state_frame


def _tercile_labels(values: pd.Series) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=3, labels=STATE_LABELS, duplicates="drop").astype("string")


def _merge_by_date(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    left_columns = ["date", *[column for column in columns if column in left.columns]]
    right_columns = ["date", *[column for column in columns if column in right.columns]]
    return left.loc[:, left_columns].merge(
        right.loc[:, right_columns],
        on="date",
        how="inner",
        suffixes=("_a", "_b"),
    )


def _country_row(
    metric_group: str,
    metric: str,
    country: str,
    value: float,
) -> dict[str, object]:
    return {
        "metric_group": metric_group,
        "metric": metric,
        "country": country,
        "country_a": None,
        "country_b": None,
        "value": value,
        "observations": None,
    }


def _pair_row(
    metric_group: str,
    metric: str,
    country_a: str,
    country_b: str,
    value: float,
    observations: int,
) -> dict[str, object]:
    return {
        "metric_group": metric_group,
        "metric": metric,
        "country": None,
        "country_a": country_a,
        "country_b": country_b,
        "value": value,
        "observations": observations,
    }
