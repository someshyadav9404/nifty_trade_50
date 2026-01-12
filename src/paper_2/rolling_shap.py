"""
Paper-2: Rolling SHAP computation using native XGBoost TreeSHAP

Why this approach:
- SHAP TreeExplainer has a known bug with multiclass XGBoost base_score
- XGBoost's pred_contribs=True computes exact TreeSHAP values internally
- This is mathematically equivalent and widely accepted
"""

import numpy as np
import pandas as pd
import xgboost as xgb


def compute_rolling_shap(model, X, window=60, stride=5):
    """
    Compute rolling aggregated SHAP values for a frozen multiclass XGBoost model.
    """

    booster = model.get_booster()
    feature_names = list(X.columns)

    records = []

    for start in range(0, len(X) - window + 1, stride):
        end = start + window
        X_window = X.iloc[start:end]

        dmat = xgb.DMatrix(X_window, feature_names=feature_names)

        # pred_contribs=True returns TreeSHAP values
        # Shape (multiclass): [n_samples, n_classes, n_features + 1]
        shap_values = booster.predict(dmat, pred_contribs=True)

        # Remove bias term (last column)
        shap_values = shap_values[..., :-1]

        # Aggregate multiclass safely
        if shap_values.ndim == 3:
            # [N, C, F] → mean over classes
            shap_values = np.mean(np.abs(shap_values), axis=1)
        else:
            shap_values = np.abs(shap_values)

        mean_abs_shap = shap_values.mean(axis=0)

        record = {"end_date": X_window.index[-1]}
        record.update(dict(zip(feature_names, mean_abs_shap)))
        records.append(record)

    shap_df = pd.DataFrame(records).set_index("end_date")
    return shap_df
