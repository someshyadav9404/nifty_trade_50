"""
Paper-2: SHAP aggregation and normalization

Purpose:
Convert rolling SHAP values into normalized explanation distributions
suitable for drift computation.
"""

import pandas as pd


def normalize_shap(shap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize SHAP values row-wise so each row sums to 1.

    This converts SHAP magnitudes into explanation distributions.
    """

    # Safety: no division by zero
    row_sums = shap_df.sum(axis=1)
    assert (row_sums > 0).all(), "Zero SHAP row encountered"

    shap_norm = shap_df.div(row_sums, axis=0)
    return shap_norm
