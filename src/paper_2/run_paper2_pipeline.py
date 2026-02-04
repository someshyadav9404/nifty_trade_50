"""
Paper-2 main pipeline
Objective: Explanation drift monitoring for a frozen ML trading model
No retraining, no filtering, no intervention
"""

import pandas as pd
import numpy as np

from pathlib import Path

from load_frozen_model import load_model
from explanation_drift import compute_js_drift
from performance_metrics import compute_rolling_performance
from shap_aggregation import normalize_shap
from rolling_shap import compute_rolling_shap
from alignment_analysis import lead_lag_correlation

def load_predictions():
    preds = np.load("artifacts/paper1/preds.npy")
    dates = pd.read_csv("artifacts/paper1/dates.csv", parse_dates=["date"])
    dates = dates["date"]

    pred_series = pd.Series(preds, index=dates, name="pred")
    return pred_series




PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_test_data():
    """
    Load labeled test data (2020 onwards).
    Assumes no feature leakage and frozen preprocessing.
    """
    data_path = PROJECT_ROOT / "data" / "labeled" / "nifty_labeled.csv"

    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.set_index("date")
    df = df.loc["2020-01-01":]

    return df

def main():
    print("Initializing Paper-2 pipeline...")

    # Load model
    model = load_model()
    print("Frozen model loaded.")

    # Load test data
    df = load_test_data()
    print(f"Test data loaded: {df.shape}")

    # -----------------------------
    # DEFINE RETURNS (ONCE, EARLY)
    # -----------------------------
    import numpy as np

    # Load frozen predictions from Paper-1
    preds = np.load("artifacts/paper1/preds.npy")
    pred_dates = pd.read_csv(
        "artifacts/paper1/dates.csv",
        parse_dates=["date"]
    )["date"]

    pred_series = pd.Series(preds, index=pred_dates, name="pred")

    # Align predictions with test data
    pred_series = pred_series.reindex(df.index)

    if pred_series.isna().any():
        raise ValueError("Prediction alignment failed")

    # Directional strategy return (NO filtering, NO retraining)
    returns = df["next_ret"] * pred_series
    print("Returns constructed.")

    # -----------------------------
    # FEATURE MATRIX (STRICT)
    # -----------------------------
    FEATURE_COLUMNS = [
        col for col in df.columns
        if col not in ["label", "next_ret"]
    ]

    X = df[FEATURE_COLUMNS]

    # -----------------------------
    # ROLLING SHAP
    # -----------------------------
    print("Computing rolling SHAP values...")
    shap_df = compute_rolling_shap(model, X, window=60, stride=5)
    print("Rolling SHAP computed.")

    # -----------------------------
    # NORMALIZE SHAP
    # -----------------------------
    shap_norm = normalize_shap(shap_df)

    # -----------------------------
    # EXPLANATION DRIFT
    # -----------------------------
    drift_series = compute_js_drift(shap_norm)
    print("Explanation drift computed.")

    # -----------------------------
    # PERFORMANCE METRICS
    # -----------------------------
    perf_df = compute_rolling_performance(returns)
    perf_df = perf_df.reindex(drift_series.index)

    print("Performance metrics computed.")

    # -----------------------------
    # ALIGNMENT ANALYSIS
    # -----------------------------
    alignment_df = lead_lag_correlation(
        drift_series,
        perf_df["rolling_return"]
    )

    print("Pipeline completed successfully.")
    shap_df.to_csv("artifacts/paper2/rolling_shap_raw.csv")
    shap_norm.to_csv("artifacts/paper2/rolling_shap_normalized.csv")
    drift_series.to_csv("artifacts/paper2/explanation_drift.csv")
    perf_df.to_csv("artifacts/paper2/rolling_performance.csv")
    alignment_df.to_csv("artifacts/paper2/alignment_metrics.csv")





if __name__ == "__main__":
    main()
