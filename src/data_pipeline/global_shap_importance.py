"""
Global SHAP Feature Importance for Random Forest and XGBoost
(With progress tracking and safe execution)
"""

import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.project_config import (
    PROCESSED_DATA,
    FEATURE_COLS,
    RF_MODEL_PATH,
    XGB_MODEL_PATH
)


def load_data():
    import pandas as pd
    print("Loading labeled dataset...")
    df = pd.read_csv(PROCESSED_DATA)
    df = df.dropna(subset=FEATURE_COLS + ["label"])
    X = df[FEATURE_COLS].values
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X


def global_shap_random_forest(X):
    print("\n[1/2] Computing Global SHAP for Random Forest (Kernel SHAP)...")

    rf_model = joblib.load(RF_MODEL_PATH)

    # VERY SMALL representative subset
    sample_size = min(100, X.shape[0])
    X_sample = X[np.random.choice(X.shape[0], size=sample_size, replace=False)]

    background_size = min(30, X_sample.shape[0])
    background = X_sample[:background_size]

    explainer = shap.KernelExplainer(
        rf_model.predict_proba,
        background
    )

    print("Running Kernel SHAP for RF (this WILL show progress)...")

    # shap_values is a list: [class0, class1, class2]
    shap_values = explainer.shap_values(X_sample)
    print(shap_values)

    # Aggregate across classes → keep sample dimension
    shap_global = np.mean(
        np.abs(np.stack(shap_values, axis=0)),
        axis=0
    )  # shape: (n_samples, n_features)

    print("dfbdb")
    print(shap_global)
    shap.summary_plot(
        shap_global,
        X_sample,
        feature_names=FEATURE_COLS,
        show=False
    )
    plt.show()


    print("Random Forest Kernel SHAP completed.")

def global_shap_xgboost(X):
    print("\n[2/2] Computing Global SHAP for XGBoost...")

    xgb_model = joblib.load(XGB_MODEL_PATH)

    # Use limited background for stability
    bg_size = min(50, X.shape[0])
    background = X[np.random.choice(X.shape[0], size=bg_size, replace=False)]

    explainer = shap.KernelExplainer(
        xgb_model.predict_proba,
        background
    )

    # Use subset for SHAP computation to keep runtime reasonable
    sample_size = min(200, X.shape[0])
    X_sample = X[:sample_size]

    shap_values = []

    print("Running SHAP KernelExplainer (this may take time)...")
    for i in tqdm(range(X_sample.shape[0]), desc="XGBoost SHAP Progress"):
        shap_val = explainer.shap_values(X_sample[i:i+1], nsamples=100)
        shap_values.append(shap_val)

    # Convert list → array
    shap_values = np.array(shap_values)

    # Aggregate across samples and classes
    shap_global = np.mean(np.abs(shap_values), axis=(0, 1))

    shap.summary_plot(
        shap_global,
        X_sample,
        feature_names=FEATURE_COLS,
        show=True
    )

    print("XGBoost SHAP completed.")


def main():
    X = load_data()

    global_shap_random_forest(X)
    global_shap_xgboost(X)

    print("\nAll Global SHAP analyses completed successfully.")


if __name__ == "__main__":
    main()
