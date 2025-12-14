import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_models():

    df = pd.read_csv("data/labeled/nifty_labeled.csv")
    df = df.dropna()

    X = df.drop(columns=["label","next_ret", "Date"])
    y = df["label"]

    print(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    print("\n========== Training Random Forest ==========")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)

    print("Random Forest Classification Report:")
    print(classification_report(y_test, preds_rf))

    joblib.dump(rf, "models/random_forest.joblib")
    print("Saved Random Forest model.")

    # Convert labels for XGBoost compatibility
    y_train_xgb = y_train.replace({-1: 0, 0: 1, 1: 2})
    y_test_xgb  = y_test.replace({-1: 0, 0: 1, 1: 2})

    print("\n========== Training XGBoost ==========")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss"
    )

    xgb.fit(X_train, y_train_xgb)
    preds_xgb = xgb.predict(X_test)
    
  # Convert predictions back to original form
    preds_xgb = pd.Series(preds_xgb).replace({0: -1, 1: 0, 2: 1})

    print("XGBoost Classification Report:")
    print(classification_report(y_test, preds_xgb))

    joblib.dump(xgb, "models/xgboost.joblib")
    print("Saved XGBoost model.")

if __name__ == "__main__":
    train_models()
