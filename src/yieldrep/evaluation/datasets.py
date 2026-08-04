from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from yieldrep.config import ProjectConfig
from yieldrep.evaluation.splits import evaluation_splits


def build_modeling_datasets(config: ProjectConfig) -> list[Path]:
    """Join baseline representations to forward target datasets."""
    targets = pd.read_parquet(config.targets_path)
    curves = pd.read_parquet(config.curves_path)
    config.modeling_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    supervised = make_supervised_yield_change_dataset(config, targets, curves)
    supervised.to_parquet(config.supervised_yield_change_path, index=False)
    output_paths.append(config.supervised_yield_change_path)
    output_paths.extend(_build_target_family(config, targets, curves, suffix=""))

    if config.standardized_targets_path.exists():
        standardized_targets = pd.read_parquet(config.standardized_targets_path)
        output_paths.extend(
            _build_target_family(config, standardized_targets, curves, suffix="_standardized")
        )

    if config.residual_targets_path.exists():
        residual_targets = pd.read_parquet(config.residual_targets_path)
        supervised_residual = make_supervised_residual_change_dataset(
            config,
            residual_targets,
            curves,
        )
        supervised_residual.to_parquet(config.supervised_residual_change_path, index=False)
        output_paths.append(config.supervised_residual_change_path)
        output_paths.extend(
            _build_target_family(config, residual_targets, curves, suffix="_residual")
        )

    if config.vol_targets_path.exists():
        vol_targets = pd.read_parquet(config.vol_targets_path)
        supervised_vol = make_supervised_vol_change_dataset(config, vol_targets, curves)
        supervised_vol.to_parquet(config.supervised_vol_change_path, index=False)
        output_paths.append(config.supervised_vol_change_path)
        output_paths.extend(_build_target_family(config, vol_targets, curves, suffix="_vol"))

    if config.curve_vol_regime_targets_path.exists():
        curve_vol_regime_targets = pd.read_parquet(config.curve_vol_regime_targets_path)
        output_paths.extend(
            _build_curve_level_target_family(
                config,
                curve_vol_regime_targets,
                suffix="_curve_vol_regime",
            )
        )

    return output_paths


def make_supervised_yield_change_dataset(
    config: ProjectConfig,
    targets: pd.DataFrame,
    curves: pd.DataFrame,
) -> pd.DataFrame:
    """Build the canonical supervised panel for future yield-change forecasting."""
    dataset = make_supervised_feature_dataset(config, targets, curves)
    dataset = _attach_evaluation_splits(dataset, config)
    return _sort_supervised_dataset(dataset)


def make_supervised_residual_change_dataset(
    config: ProjectConfig,
    targets: pd.DataFrame,
    curves: pd.DataFrame,
) -> pd.DataFrame:
    """Build the canonical supervised panel for residual-change forecasting."""
    dataset = make_supervised_feature_dataset(config, targets, curves)
    dataset = _attach_evaluation_splits(dataset, config)
    return _sort_supervised_dataset(dataset)


def make_supervised_vol_change_dataset(
    config: ProjectConfig,
    targets: pd.DataFrame,
    curves: pd.DataFrame,
) -> pd.DataFrame:
    """Build the canonical supervised panel for volatility-change forecasting."""
    dataset = make_supervised_feature_dataset(config, targets, curves)
    dataset = _attach_evaluation_splits(dataset, config)
    return _sort_supervised_dataset(dataset)


def make_supervised_feature_dataset(
    config: ProjectConfig,
    targets: pd.DataFrame,
    curves: pd.DataFrame,
) -> pd.DataFrame:
    """Join all current supervised feature families without assigning split labels."""
    dataset = targets.copy()
    dataset["date"] = pd.to_datetime(dataset["date"])

    for features, keys in [
        (_read_curve_features(config), ["date", "country"]),
        (_read_autoencoder_embeddings(config), ["date", "country"]),
        (_read_transformer_embeddings(config), ["date", "country"]),
        (
            make_lagged_yield_change_features(curves, config.evaluation.lag_days),
            ["date", "country", "maturity_years"],
        ),
        (_read_carry_roll_features(config), ["date", "country", "maturity_years"]),
    ]:
        if not features.empty:
            dataset = _merge_features(dataset, features, keys)

    return dataset


def _sort_supervised_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset.sort_values(
        ["country", "horizon_days", "window_id", "split", "date", "maturity_years"]
    ).reset_index(drop=True)


def _merge_features(
    dataset: pd.DataFrame,
    features: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    feature_columns = [
        column for column in features.columns if column in keys or column not in dataset.columns
    ]
    return dataset.merge(features.loc[:, feature_columns], on=keys, how="left")


def _read_curve_features(config: ProjectConfig) -> pd.DataFrame:
    if not config.curve_features_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(config.curve_features_path)


def _read_carry_roll_features(config: ProjectConfig) -> pd.DataFrame:
    if not config.carry_roll_features_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(config.carry_roll_features_path)


def _read_policy_features(config: ProjectConfig) -> pd.DataFrame:
    if not config.policy_features_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(config.policy_features_path)


def _read_autoencoder_embeddings(config: ProjectConfig) -> pd.DataFrame:
    return _read_learned_embeddings(config.autoencoder_dir, "AE")


def _read_transformer_embeddings(config: ProjectConfig) -> pd.DataFrame:
    return _read_learned_embeddings(config.transformer_dir, "TE")


def _read_graph_autoencoder_embeddings(config: ProjectConfig) -> pd.DataFrame:
    return _read_learned_embeddings(config.gnn_dir, "GE")


def _read_learned_embeddings(model_dir: Path, prefix: str) -> pd.DataFrame:
    if not model_dir.exists():
        return pd.DataFrame()

    rows = []
    for path in sorted(model_dir.glob("*_embeddings.parquet")):
        frame = pd.read_parquet(path)
        feature_columns = [column for column in frame.columns if column.startswith(prefix)]
        if frame.empty or not feature_columns:
            continue
        rows.append(frame.loc[:, ["date", "country", *feature_columns]].copy())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _attach_evaluation_splits(data: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for group_values, group in data.groupby(["country", "horizon_days"], sort=True):
        country, horizon_days = group_values
        for split in evaluation_splits(
            group,
            method=config.evaluation.method,
            test_fraction=config.evaluation.test_fraction,
            min_train_dates=config.evaluation.min_train_dates,
            test_window_dates=config.evaluation.test_window_dates,
            step_dates=config.evaluation.step_dates,
            max_windows=config.evaluation.walk_forward_max_windows,
            horizon_days=int(horizon_days),
            non_overlapping_targets=config.evaluation.non_overlapping_targets,
        ):
            frames.extend(
                [
                    _labeled_split_frame(
                        split.train,
                        country,
                        horizon_days,
                        split.method,
                        split.window_id,
                        "train",
                    ),
                    _labeled_split_frame(
                        split.test,
                        country,
                        horizon_days,
                        split.method,
                        split.window_id,
                        "test",
                    ),
                ]
            )

    if not frames:
        return data.iloc[0:0].copy()

    return pd.concat(frames, ignore_index=True)


def _labeled_split_frame(
    data: pd.DataFrame,
    country: object,
    horizon_days: object,
    split_method: str,
    window_id: int,
    split: str,
) -> pd.DataFrame:
    labeled = data.copy()
    labeled["split_method"] = split_method
    labeled["window_id"] = window_id
    labeled["split"] = split
    labeled["country"] = country
    labeled["horizon_days"] = horizon_days
    return labeled


def _build_target_family(
    config: ProjectConfig,
    targets: pd.DataFrame,
    curves: pd.DataFrame,
    suffix: str,
) -> list[Path]:
    output_paths: list[Path] = []

    lagged_targets = _join_lagged_targets(curves, targets, config.evaluation.lag_days)
    if not lagged_targets.empty:
        lagged_path = config.modeling_dir / f"lagged{suffix}_targets.parquet"
        lagged_targets.to_parquet(lagged_path, index=False)
        output_paths.append(lagged_path)

    curve_feature_targets = _join_curve_feature_targets(config, targets)
    if not curve_feature_targets.empty:
        curve_path = config.modeling_dir / f"curve{suffix}_targets.parquet"
        curve_feature_targets.to_parquet(curve_path, index=False)
        output_paths.append(curve_path)

    carry_roll_targets = _join_carry_roll_targets(config, targets)
    if not carry_roll_targets.empty:
        carry_roll_path = config.modeling_dir / f"carry_roll{suffix}_targets.parquet"
        carry_roll_targets.to_parquet(carry_roll_path, index=False)
        output_paths.append(carry_roll_path)

    for representation, features in [
        ("autoencoder", _read_autoencoder_embeddings(config)),
        ("transformer", _read_transformer_embeddings(config)),
    ]:
        if features.empty:
            continue
        learned_targets = targets.merge(features, on=["date", "country"], how="inner")
        if learned_targets.empty:
            continue
        learned_path = config.modeling_dir / f"{representation}{suffix}_targets.parquet"
        learned_targets.to_parquet(learned_path, index=False)
        output_paths.append(learned_path)

    return output_paths


def _build_curve_level_target_family(
    config: ProjectConfig,
    targets: pd.DataFrame,
    suffix: str,
) -> list[Path]:
    output_paths: list[Path] = []

    for representation, features in [
        ("curve", _read_curve_features(config)),
        ("policy", _read_policy_features(config)),
        ("autoencoder", _read_autoencoder_embeddings(config)),
        ("transformer", _read_transformer_embeddings(config)),
        ("graph_autoencoder", _read_graph_autoencoder_embeddings(config)),
    ]:
        if features.empty:
            continue
        merged = targets.merge(features, on=["date", "country"], how="inner")
        if merged.empty:
            continue
        output_path = config.modeling_dir / f"{representation}{suffix}_targets.parquet"
        merged.to_parquet(output_path, index=False)
        output_paths.append(output_path)

    if "realized_curve_vol" in targets.columns:
        curve_vol = targets.copy()
        curve_vol_path = config.modeling_dir / f"curve_vol{suffix}_targets.parquet"
        curve_vol.to_parquet(curve_vol_path, index=False)
        output_paths.append(curve_vol_path)
    return output_paths


def _join_lagged_targets(
    curves: pd.DataFrame,
    targets: pd.DataFrame,
    lag_days: list[int],
) -> pd.DataFrame:
    features = make_lagged_yield_change_features(curves, lag_days=lag_days)
    return targets.merge(features, on=["date", "country", "maturity_years"], how="inner")


def _join_curve_feature_targets(config: ProjectConfig, targets: pd.DataFrame) -> pd.DataFrame:
    if not config.curve_features_path.exists():
        return pd.DataFrame()

    features = pd.read_parquet(config.curve_features_path)
    merged = targets.merge(features, on=["date", "country"], how="inner")
    return _add_state_maturity_features(
        merged,
        state_columns=[
            "level",
            "slope_10y_2y",
            "curvature_2s5s10s",
            "front_slope_2y_1y",
            "long_slope_30y_10y",
        ],
    )


def _join_carry_roll_targets(config: ProjectConfig, targets: pd.DataFrame) -> pd.DataFrame:
    if not config.carry_roll_features_path.exists():
        return pd.DataFrame()

    features = pd.read_parquet(config.carry_roll_features_path)
    return targets.merge(features, on=["date", "country", "maturity_years"], how="inner")


def _add_state_maturity_features(data: pd.DataFrame, state_columns: list[str]) -> pd.DataFrame:
    frame = data.copy()
    available_state_columns = [column for column in state_columns if column in frame.columns]
    if not available_state_columns:
        return frame

    frame["maturity"] = frame["maturity_years"].astype(float)
    frame["maturity_squared"] = frame["maturity"] ** 2
    frame["log_maturity"] = np.log1p(frame["maturity"])
    maturity_basis = ["maturity", "maturity_squared", "log_maturity"]
    for state_column in available_state_columns:
        for maturity_column in maturity_basis:
            frame[f"{state_column}_x_{maturity_column}"] = (
                frame[state_column] * frame[maturity_column]
            )
    return frame


def make_lagged_yield_change_features(
    curves: pd.DataFrame,
    lag_days: list[int],
) -> pd.DataFrame:
    """Create lagged yield-change features by country and maturity."""
    if not lag_days:
        raise ValueError("At least one lag is required")
    if any(lag <= 0 for lag in lag_days):
        raise ValueError("Lag days must be positive")

    features = curves.loc[:, ["date", "country", "maturity_years", "yield"]].copy()
    features["date"] = pd.to_datetime(features["date"])
    features = features.sort_values(["country", "maturity_years", "date"]).reset_index(drop=True)
    grouped = features.groupby(["country", "maturity_years"], sort=False)["yield"]

    lag_columns: list[str] = []
    for lag in lag_days:
        column = f"lag_{lag}_change"
        features[column] = features["yield"] - grouped.shift(lag)
        lag_columns.append(column)

    return features.dropna(subset=lag_columns).loc[
        :,
        ["date", "country", "maturity_years", *lag_columns],
    ]
