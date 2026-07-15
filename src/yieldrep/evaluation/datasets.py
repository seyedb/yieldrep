from __future__ import annotations

from pathlib import Path

import pandas as pd

from yieldrep.config import ProjectConfig


def build_modeling_datasets(config: ProjectConfig) -> list[Path]:
    """Join baseline representations to forward yield-change targets."""
    targets = pd.read_parquet(config.targets_path)
    config.modeling_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    pca_targets = _join_pca_targets(config, targets)
    if not pca_targets.empty:
        pca_path = config.modeling_dir / "pca_targets.parquet"
        pca_targets.to_parquet(pca_path, index=False)
        output_paths.append(pca_path)

    nelson_siegel_targets = _join_nelson_siegel_targets(config, targets)
    if not nelson_siegel_targets.empty:
        ns_path = config.modeling_dir / "nelson_siegel_targets.parquet"
        nelson_siegel_targets.to_parquet(ns_path, index=False)
        output_paths.append(ns_path)

    return output_paths


def _join_pca_targets(config: ProjectConfig, targets: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for scores_path in sorted(config.pca_dir.glob("*_scores.parquet")):
        country = scores_path.name.removesuffix("_scores.parquet").upper()
        scores = pd.read_parquet(scores_path)
        scores["country"] = country
        frames.append(scores)
    if not frames:
        return pd.DataFrame()

    features = pd.concat(frames, ignore_index=True)
    return targets.merge(features, on=["date", "country"], how="inner")


def _join_nelson_siegel_targets(config: ProjectConfig, targets: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for factors_path in sorted(config.nelson_siegel_dir.glob("*_factors.parquet")):
        frames.append(pd.read_parquet(factors_path))
    if not frames:
        return pd.DataFrame()

    features = pd.concat(frames, ignore_index=True)
    return targets.merge(features, on=["date", "country"], how="inner")
