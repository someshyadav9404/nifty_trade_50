
import pandas as pd
import numpy as np
import joblib
import shap
import xgboost as xgb
import os

# ----------------------------
# Paths
# ----------------------------
DATA_PATH = "data/labeled/nifty_labeled.csv"

MODELS = {
    "rf": "models/random_forest_paper1.pkl",
    "xgb": "models/xgboost_paper1_frozen.pkl"
}

ARTIFACT_DIR = "artifacts/paper1"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

train_df = df[df["Date"] < "2020-01-01"].reset_index(drop=True)
test_df  = df[df["Date"] >= "2020-01-01"].reset_index(drop=True)

features = [c for c in df.columns if c not in ["label", "next_ret", "Date"]]

X_train = train_df[features]
X_test  = test_df[features]

print("Train rows:", len(train_df))
print("Test rows :", len(test_df))
print("Feature count:", len(features))

# Save test dates
test_df[["Date"]].to_csv(f"{ARTIFACT_DIR}/dates.csv", index=False)

# ----------------------------
# Helper function
# ----------------------------
def compute_trade_shap(model_name, model, X, pred_classes):

    if model_name == "rf":

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = np.stack(shap_values, axis=2)

        elif shap_values.shape[0] == len(model.classes_):
            shap_values = np.transpose(shap_values, (1,2,0))

        trade_shap = np.array([
            shap_values[i, :, pred_classes[i]]
            for i in range(len(pred_classes))
        ])

    elif model_name == "xgb":

        booster = model.get_booster()
        dmatrix = xgb.DMatrix(X)

        shap_values = booster.predict(
            dmatrix,
            pred_contribs=True
        )

        shap_values = shap_values[:, :, :-1]

        trade_shap = np.array([
            shap_values[i, pred_classes[i], :]
            for i in range(len(pred_classes))
        ])

    return trade_shap


# ----------------------------
# Process each model
# ----------------------------
for name, path in MODELS.items():

    print("\n====================")
    print(f"Processing model: {name}")
    print("====================")

    model = joblib.load(path)

    # ----------------------------
    # TRAIN predictions
    # ----------------------------
    train_probs = model.predict_proba(X_train)
    train_pred_classes = train_probs.argmax(axis=1)
    train_conf = train_probs.max(axis=1)

    train_preds = train_pred_classes - 1

    print("Train prediction distribution:")
    print(pd.Series(train_preds).value_counts())

    print("Computing TRAIN SHAP values...")

    train_trade_shap = compute_trade_shap(
        name,
        model,
        X_train,
        train_pred_classes
    )

    print("Train SHAP shape:", train_trade_shap.shape)

    # ----------------------------
    # TEST predictions
    # ----------------------------
    test_probs = model.predict_proba(X_test)
    test_pred_classes = test_probs.argmax(axis=1)

    test_preds = test_pred_classes - 1
    test_confidence = test_probs.max(axis=1)

    print("Test prediction distribution:")
    print(pd.Series(test_preds).value_counts())

    print("Computing TEST SHAP values...")

    test_trade_shap = compute_trade_shap(
        name,
        model,
        X_test,
        test_pred_classes
    )

    print("Test SHAP shape:", test_trade_shap.shape)

    # ----------------------------
    # Save artifacts
    # ----------------------------
    np.save(f"{ARTIFACT_DIR}/{name}_train_confidence.npy", train_conf)
    np.save(f"{ARTIFACT_DIR}/{name}_train_trade_shap.npy", train_trade_shap)
    np.save(f"{ARTIFACT_DIR}/{name}_trade_shap.npy", test_trade_shap)

    np.save(f"{ARTIFACT_DIR}/{name}_train_preds.npy", train_preds)
    np.save(f"{ARTIFACT_DIR}/{name}_preds.npy", test_preds)

    np.save(f"{ARTIFACT_DIR}/{name}_confidence.npy", test_confidence)

print("\nAll artifacts saved in:", ARTIFACT_DIR)
