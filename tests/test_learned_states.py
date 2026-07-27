from __future__ import annotations

import pandas as pd
import pytest

from yieldrep.evaluation.learned_states import (
    learned_state_regime_means,
    learned_state_regime_summary,
)


def test_learned_state_regime_summary_measures_high_low_separation() -> None:
    regimes = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6),
            "country": ["US"] * 6,
            "split": ["train", "train", "test", "test", "test", "test"],
            "representation": ["autoencoder"] * 6,
            "AE1": [0.0, 0.2, 0.1, 2.0, 2.2, 2.1],
            "AE2": [0.0, 0.1, 0.2, 2.0, 2.1, 2.2],
            "regime_type": ["macro"] * 6,
            "indicator": ["inflation"] * 6,
            "regime": ["low", "low", "low", "high", "high", "high"],
        }
    )

    summary = learned_state_regime_summary(regimes)
    means = learned_state_regime_means(regimes)

    assert summary.loc[0, "representation"] == "autoencoder"
    assert summary.loc[0, "high_rows"] == 3
    assert summary.loc[0, "low_rows"] == 3
    assert summary.loc[0, "high_low_distance"] == pytest.approx(2.828427, rel=1e-5)
    assert summary.loc[0, "separation_ratio"] > 1.0
    assert set(means["regime"]) == {"high", "low"}
    assert set(means["latent_feature"]) == {"AE1", "AE2"}
