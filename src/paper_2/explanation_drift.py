"""
Paper-2: Explanation drift computation
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def compute_js_drift(shap_norm: pd.DataFrame) -> pd.Series:
    """
    Compute Jensen–Shannon explanation drift relative to baseline.

    Baseline = first rolling window (earliest post-2020 explanation).
    """

    baseline = shap_norm.iloc[0].values
    dates = shap_norm.index

    drift_values = []

    for i in range(1, len(shap_norm)):
        current = shap_norm.iloc[i].values
        drift = jensenshannon(baseline, current, base=2)
        drift_values.append(drift)

    drift_series = pd.Series(
        drift_values,
        index=dates[1:],
        name="explanation_drift"
    )

    return drift_series
