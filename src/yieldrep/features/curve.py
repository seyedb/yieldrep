from __future__ import annotations

import pandas as pd


def curve_panel(curves: pd.DataFrame, country: str) -> pd.DataFrame:
    """Pivot long-format curves into a date by maturity panel for one country."""
    panel = curves.loc[curves["country"] == country].pivot_table(
        index="date",
        columns="maturity_years",
        values="yield",
        aggfunc="mean",
    )
    panel = panel.sort_index().sort_index(axis=1)
    panel.columns = [float(column) for column in panel.columns]
    return panel
