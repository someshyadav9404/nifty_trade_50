import pandas as pd
import joblib
from backtesting.backtest import backtest_strategy

def run_backtest(model_path, model_name):
    df = pd.read_csv("data/labeled/nifty_labeled.csv")
    df = df.dropna()

    X = df.drop(columns=["label", "Date"])
    y_true = df["label"]

    model = joblib.load(model_path)

    preds = model.predict(X)

    results = backtest_strategy(df, preds)

    # Convert output to a dataframe
    results_table = pd.DataFrame({
        "Model": [model_name],
        "Final Equity (₹)": [f"{results['final_equity']:.2f}"],
        "Total Return (%)": [f"{results['total_return']:.2f}"],
        "Sharpe Ratio": [f"{results['sharpe_ratio']:.2f}"],
        "Max Drawdown (%)": [f"{results['max_drawdown']:.2f}"]
    })

    print("\n===== Backtest Results =====")
    print(results_table.to_string(index=False))

    # Optionally save table
    # results_table.to_csv("results/backtest_results.csv", index=False)

    return results_table

if __name__ == "__main__":
    run_backtest("models/random_forest.pkl", "Random Forest")
