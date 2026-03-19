import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_models():

    # Load data
    df = pd.read_csv("data/labeled/nifty_labeled.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    # Time-based split
    train_df = df[(df["Date"] > "2013-01-21") & (df["Date"] < "2020-01-01")]
    test_df  = df[df["Date"] >= "2020-01-01"]

    X_train = train_df.drop(columns=["label", "next_ret", "Date"])
    y_train = train_df["label"]

    X_test  = test_df.drop(columns=["label", "next_ret", "Date"])
    y_test  = test_df["label"]

    test_returns = test_df["next_ret"].values


    # ================= Random Forest =================
    print("\n========== Training Random Forest ==========")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    print(train_df["label"].value_counts())
    print(test_df["label"].value_counts())

    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)
    print(preds_rf)
    print(rf.classes_)
    probs = rf.predict_proba(X_test)
    print(probs)
    confidence = probs[np.arange(len(preds_rf)), preds_rf+1]
    print(np.arange(len(preds_rf)))
    print("Confidence scores for Random Forest predictions:")
    print(confidence)

    print("Random Forest Classification Report:")
    print(classification_report(y_test, preds_rf))

    joblib.dump(rf, "models/random_forest_paper1.pkl")
    
    print("Saved Random Forest model.")

    # ================= XGBoost =================
    print("\n========== Training XGBoost ==========")

    # XGBoost needs labels {0,1,2}
    y_train_xgb = y_train.replace({-1: 0, 0: 1, 1: 2})
    y_test_xgb  = y_test.replace({-1: 0, 0: 1, 1: 2})

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42
    )

    xgb.fit(X_train, y_train_xgb)

    preds_xgb = xgb.predict(X_test)
    preds_xgb = pd.Series(preds_xgb).replace({0: -1, 1: 0, 2: 1})

    print("XGBoost Classification Report:")
    print(classification_report(y_test, preds_xgb))

    joblib.dump(xgb, "models/xgboost_paper1_frozen.pkl")
    print("Saved XGBoost model.")

    return rf ,X_test, preds_rf, test_returns

if __name__ == "__main__":
    rf ,X_test, preds_rf, test_returns = train_models()
