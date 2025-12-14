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
    print(shap_values.shape)

    # Aggregate across classes → keep sample dimension
    shap_global = np.mean(
        np.abs(np.stack(shap_values, axis=0)),
        axis=2
    )  # shape: (n_samples, n_features)

    print("dfbdb")
    print(shap_global.shape)
    shap.summary_plot(
        shap_global,
        X_sample,
        feature_names=FEATURE_COLS,
        show=False
    )
    plt.show()
    plt.savefig("rf_global_shap.png", dpi=300, bbox_inches="tight")
    plt.close() 

    print("Random Forest Kernel SHAP completed.")

def global_shap_xgboost(X):
    print("\n[2/2] Computing Global SHAP for XGBoost...")

    xgb_model = joblib.load(XGB_MODEL_PATH)

    # Subsample for speed
    sample_size = min(200, X.shape[0])
    X_sample = X[:sample_size]

    # Tree SHAP (correct explainer)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    # ---- Handle multiclass output ----
    if isinstance(shap_values, list):
        # list[class] -> (samples, features)
        shap_values = np.stack(shap_values, axis=2)  # (samples, features, classes)

    print("SHAP values shape:", shap_values.shape)

    # Global importance: mean |SHAP| over samples and classes
    shap_global = np.mean(np.abs(shap_values), axis=(0, 2))  # (features,)

    # Plot global importance
    shap.bar_plot(
        shap_global,
        feature_names=FEATURE_COLS,
        show=False
    )

    plt.savefig("xgb_global_shap.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("XGBoost Tree SHAP completed.")

def main():
    X = load_data()

    global_shap_random_forest(X)
    global_shap_xgboost(X)

    print("\nAll Global SHAP analyses completed successfully.")


if __name__ == "__main__":
    main()
