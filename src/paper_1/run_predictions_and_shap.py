import pandas as pd
import numpy as np
import joblib
import shap
import os

# ----------------------------
# Paths & setup
# ----------------------------
DATA_PATH = "data/labeled/nifty_labeled.csv"
MODEL_PATH = "models/xgboost_paper1_frozen.pkl"
ARTIFACT_DIR = "artifacts/paper1"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ----------------------------
# Load data (TEST SET ONLY)
# ----------------------------
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

test_df = df[df["Date"] >= "2020-01-01"].reset_index(drop=True)

features = [c for c in test_df.columns if c not in ["label", "next_ret", "Date"]]

X_test = test_df[features]
y_test = test_df["label"]   # kept only for sanity checks

print("Test rows:", len(test_df))
print("Feature count:", len(features))

# ----------------------------
# Load frozen model
# ----------------------------
model = joblib.load(MODEL_PATH)

# ----------------------------
# Predictions & confidence
# ----------------------------
probs = model.predict_proba(X_test)          # shape: (N, 3)
pred_classes = probs.argmax(axis=1)          # {0,1,2}
preds = pred_classes - 1                     # map → {-1,0,1}
confidence = probs.max(axis=1)

print("\nPrediction distribution:")
print(pd.Series(preds).value_counts())

# ----------------------------
# SHAP computation
# ----------------------------
print("\nComputing SHAP values (this may take time)...")

# IMPORTANT: pass model.predict_proba, not the model itself
explainer = shap.Explainer(
    model.predict_proba,
    X_test,
    feature_names=X_test.columns
)

shap_exp = explainer(X_test)

# shap_exp.values shape: (N, F, C)
# Extract SHAP values for the predicted class only
trade_shap = np.array([
    shap_exp.values[i, :, pred_classes[i]]
    for i in range(len(pred_classes))
])

print("SHAP matrix shape:", trade_shap.shape)


# ----------------------------
# Save artifacts
# ----------------------------
np.save(f"{ARTIFACT_DIR}/trade_shap.npy", trade_shap)
np.save(f"{ARTIFACT_DIR}/preds.npy", preds)
np.save(f"{ARTIFACT_DIR}/confidence.npy", confidence)

test_df[["Date"]].to_csv(f"{ARTIFACT_DIR}/dates.csv", index=False)
test_df[["label"]].to_csv(f"{ARTIFACT_DIR}/true_labels.csv", index=False)

print("\nArtifacts saved in:", ARTIFACT_DIR)
