import pandas as pd
import pytest

from yieldrep.evaluation.targets import (
    make_forward_residual_change_targets,
    make_forward_standardized_yield_change_targets,
    make_forward_vol_change_targets,
    make_forward_yield_change_targets,
)


def test_make_forward_yield_change_targets() -> None:
    curves = _sample_curves()

    targets = make_forward_yield_change_targets(curves, horizons_days=[1, 2])

    one_day = targets.loc[
        (targets["horizon_days"] == 1)
        & (targets["country"] == "US")
        & (targets["maturity_years"] == 2.0)
    ]
    assert one_day["target_yield_change"].tolist() == pytest.approx([0.1, 0.1, 0.1])
    assert set(targets["horizon_days"]) == {1, 2}


def test_make_forward_residual_change_targets() -> None:
    fitted = _sample_fitted_curves()

    targets = make_forward_residual_change_targets(fitted, horizons_days=[1, 2])

    one_day = targets.loc[
        (targets["horizon_days"] == 1)
        & (targets["country"] == "US")
        & (targets["maturity_years"] == 2.0)
    ]
    assert one_day["target_residual_change"].tolist() == pytest.approx([0.02, 0.02, 0.02])
    assert set(targets["horizon_days"]) == {1, 2}


def test_make_forward_vol_change_targets() -> None:
    curves = _sample_curves(periods=8)

    targets = make_forward_vol_change_targets(
        curves,
        horizons_days=[1],
        realized_vol_window=2,
    )

    assert {"realized_vol", "future_realized_vol", "target_vol_change", "future_vol_regime"}.issubset(
        targets.columns
    )
    assert set(targets["future_vol_regime"]).issubset({"low", "medium", "high"})
    assert targets["target_vol_change"].notna().all()


def test_make_forward_standardized_yield_change_targets() -> None:
    curves = _sample_curves(periods=8)

    targets = make_forward_standardized_yield_change_targets(
        curves,
        horizons_days=[1],
        realized_vol_window=2,
    )

    assert {"realized_vol", "target_standardized_yield_change"}.issubset(targets.columns)
    assert targets["realized_vol"].gt(0).all()
    assert targets["target_standardized_yield_change"].notna().all()


def test_make_forward_yield_change_targets_rejects_invalid_horizons() -> None:
    with pytest.raises(ValueError, match="At least one"):
        make_forward_yield_change_targets(_sample_curves(), horizons_days=[])
    with pytest.raises(ValueError, match="positive"):
        make_forward_yield_change_targets(_sample_curves(), horizons_days=[0])


def _sample_curves(periods: int = 4) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods)
    return pd.DataFrame(
        [
            {
                "date": date,
                "country": "US",
                "maturity_years": maturity,
                "yield": 3.0 + date_index * 0.1 + maturity * 0.01,
                "source": "test",
            }
            for date_index, date in enumerate(dates)
            for maturity in [2.0, 10.0]
        ]
    )


def _sample_fitted_curves() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4)
    return pd.DataFrame(
        [
            {
                "date": date,
                "country": "US",
                "maturity_years": maturity,
                "fitted_yield": 3.0 + maturity * 0.01,
                "residual": date_index * 0.02 + maturity * 0.001,
                "tau": 1.5,
            }
            for date_index, date in enumerate(dates)
            for maturity in [2.0, 10.0]
        ]
    )
