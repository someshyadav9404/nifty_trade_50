import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from backtesting.backtest import backtest_strategy
import os
os.makedirs("results", exist_ok=True)

FEATURE_COLS = [
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    "ret_1d",
    "ma_5",
    "ma_20",
    "ma_diff",
    "vol_20",
    "momentum"
]

def run_backtest(model_path, model_name):
    # ---- Load data ----
    df = pd.read_csv(
        "data/labeled/nifty_labeled.csv",
        parse_dates=["Date"]
    ).dropna()

    df.set_index("Date", inplace=True)

    # ---- Features ----
    X = df[FEATURE_COLS]

    # ---- Load model & predict ----
    model = joblib.load(model_path)
    preds = model.predict(X)

    # ---- Run backtest ----
    results = backtest_strategy(df, preds)

    # ---- Metrics table ----
    results_table = pd.DataFrame({
        "Model": [model_name],
        "Final Equity (₹)": [f"{results['final_equity']:.2f}"],
        "Total Return (%)": [f"{results['total_return_%']:.2f}"],
        "Sharpe Ratio": [f"{results['sharpe_ratio']:.2f}"],
        "Max Drawdown (%)": [f"{results['max_drawdown_%']:.2f}"]
    })

    print("\n===== Backtest Results =====")
    print(results_table.to_string(index=False))

    # ---- Save equity curve ----
    plt.figure(figsize=(10, 5))
    plt.plot(results["equity_curve"])
    plt.title(f"Equity Curve – {model_name}")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.savefig(f"results/equity_curve_{model_name}.png", dpi=300)
    plt.close()

    # ---- Save drawdown curve ----
    plt.figure(figsize=(10, 3))
    plt.plot(results["drawdown_curve"])
    plt.title(f"Drawdown – {model_name}")
    plt.ylabel("Drawdown")
    plt.xlabel("Time")
    plt.grid(True)
    plt.savefig(f"results/drawdown_{model_name}.png", dpi=300)
    plt.close()

    # ---- Save trade log ----
    results["trade_log"].to_csv(
        f"results/trades_{model_name}.csv",
        index=False
    )

    return results_table

if __name__ == "__main__":
    models = [
        ("models/random_forest.joblib", "Random Forest"),
        ("models/xgboost.joblib", "XGBoost")
    ]

    for model_path, model_name in models:
        run_backtest(model_path, model_name)