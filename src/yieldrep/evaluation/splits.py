from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitWindow:
    method: str
    window_id: int
    train: pd.DataFrame
    test: pd.DataFrame


def evaluation_splits(
    data: pd.DataFrame,
    method: str,
    test_fraction: float,
    min_train_dates: int,
    test_window_dates: int,
    step_dates: int,
    max_windows: int | None = None,
    horizon_days: int | None = None,
    non_overlapping_targets: bool = False,
) -> list[SplitWindow]:
    if method == "date_ordered":
        train, test = date_ordered_split(data, test_fraction=test_fraction)
        split = SplitWindow(method=method, window_id=0, train=train, test=test)
        return [_apply_non_overlapping_test_filter(split, horizon_days, non_overlapping_targets)]
    if method == "walk_forward":
        splits = walk_forward_splits(
            data,
            min_train_dates=min_train_dates,
            test_window_dates=test_window_dates,
            step_dates=step_dates,
            max_windows=max_windows,
        )
        return [
            _apply_non_overlapping_test_filter(split, horizon_days, non_overlapping_targets)
            for split in splits
        ]
    raise ValueError(f"Unsupported evaluation method: {method}")


def date_ordered_split(
    data: pd.DataFrame,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by unique dates so all maturities for a date remain together."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    dates = pd.Index(sorted(pd.to_datetime(data["date"]).unique()))
    split_index = int(len(dates) * (1.0 - test_fraction))
    if split_index <= 0 or split_index >= len(dates):
        return data.iloc[0:0].copy(), data.iloc[0:0].copy()

    train_dates = set(dates[:split_index])
    test_dates = set(dates[split_index:])
    normalized_dates = pd.to_datetime(data["date"])
    train = data.loc[normalized_dates.isin(train_dates)].copy()
    test = data.loc[normalized_dates.isin(test_dates)].copy()
    return train, test


def walk_forward_splits(
    data: pd.DataFrame,
    min_train_dates: int,
    test_window_dates: int,
    step_dates: int,
    max_windows: int | None = None,
) -> list[SplitWindow]:
    """Create expanding-window chronological train/test splits."""
    if min_train_dates <= 0:
        raise ValueError("min_train_dates must be positive")
    if test_window_dates <= 0:
        raise ValueError("test_window_dates must be positive")
    if step_dates <= 0:
        raise ValueError("step_dates must be positive")
    if max_windows is not None and max_windows <= 0:
        raise ValueError("max_windows must be positive")

    dates = pd.Index(sorted(pd.to_datetime(data["date"]).unique()))
    normalized_dates = pd.to_datetime(data["date"])
    splits: list[SplitWindow] = []
    test_starts = list(range(min_train_dates, len(dates), step_dates))
    if max_windows is not None:
        test_starts = test_starts[-max_windows:]
    for window_id, test_start in enumerate(test_starts):
        test_end = min(test_start + test_window_dates, len(dates))
        train_dates = set(dates[:test_start])
        test_dates = set(dates[test_start:test_end])
        train = data.loc[normalized_dates.isin(train_dates)].copy()
        test = data.loc[normalized_dates.isin(test_dates)].copy()
        splits.append(
            SplitWindow(
                method="walk_forward",
                window_id=window_id,
                train=train,
                test=test,
            )
        )
    return splits


def _apply_non_overlapping_test_filter(
    split: SplitWindow,
    horizon_days: int | None,
    non_overlapping_targets: bool,
) -> SplitWindow:
    if not non_overlapping_targets or horizon_days is None or horizon_days <= 1 or split.test.empty:
        return split

    dates = pd.Index(sorted(pd.to_datetime(split.test["date"]).unique()))
    keep_dates = set(dates[::horizon_days])
    normalized_dates = pd.to_datetime(split.test["date"])
    filtered_test = split.test.loc[normalized_dates.isin(keep_dates)].copy()
    return SplitWindow(
        method=split.method,
        window_id=split.window_id,
        train=split.train,
        test=filtered_test,
    )
