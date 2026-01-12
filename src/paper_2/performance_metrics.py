"""
Paper-2: Rolling performance metrics
"""

import pandas as pd
import numpy as np


def compute_rolling_performance(returns: pd.Series, window=60) -> pd.DataFrame:
    """
    Compute rolling performance metrics aligned with explanation drift.
    """

    perf = pd.DataFrame(index=returns.index)

    perf["rolling_return"] = returns.rolling(window).sum()

    perf["rolling_drawdown"] = returns.rolling(window).apply(
        lambda x: x.cumsum().max() - x.cumsum().iloc[-1],
        raw=False
    )

    return perf
