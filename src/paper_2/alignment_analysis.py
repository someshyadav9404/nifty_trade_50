"""
Paper-2: Alignment analysis between explanation drift and performance
"""

import pandas as pd
from scipy.stats import spearmanr


def lead_lag_correlation(drift: pd.Series,
                         metric: pd.Series,
                         max_lag=20) -> pd.DataFrame:
    """
    Compute Spearman correlation across lead-lag offsets.
    """

    records = []

    for lag in range(-max_lag, max_lag + 1):
        shifted = drift.shift(lag)
        idx = shifted.dropna().index.intersection(metric.dropna().index)

        if len(idx) < 10:
            continue

        corr, _ = spearmanr(shifted.loc[idx], metric.loc[idx])

        records.append({
            "lag": lag,
            "spearman_corr": corr
        })

    return pd.DataFrame(records)
