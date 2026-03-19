import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

ARTIFACT_DIR = "artifacts/paper1"
DATA_PATH = "data/labeled/nifty_labeled.csv"

MODELS = ["rf", "xgb"]

TRANSACTION_COSTS = [0.00025, 0.0005, 0.001]


# -----------------------------
# Load market data
# -----------------------------
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

test_df = df[df["Date"] >= "2020-01-01"].reset_index(drop=True)

returns = test_df["next_ret"].values
dates = test_df["Date"]


# -----------------------------
# Backtest engine
# -----------------------------
def run_backtest(preds, returns, transaction_cost):

    position = 0
    equity = 1.0

    equity_curve = []
    trades = []

    for i in range(len(preds)):

        signal = preds[i]
        cost = 0
        trade_action = "hold"

        # enter position
        if position == 0 and signal in [-1, 1]:

            position = signal
            cost += transaction_cost
            trade_action = "enter"

        # flip position
        elif position == 1 and signal == -1:

            position = -1
            cost += transaction_cost * 2
            trade_action = "flip"

        elif position == -1 and signal == 1:

            position = 1
            cost += transaction_cost * 2
            trade_action = "flip"

        elif signal == 0 and position != 0:
            trade_action = "hold"

        pnl = position * returns[i]

        equity = equity * (1 + pnl - cost)

        equity_curve.append(equity)

        trades.append({
            "idx": i,
            "date": dates.iloc[i],
            "signal": signal,
            "position": position,
            "return": returns[i],
            "pnl": pnl,
            "equity": equity,
            "action": trade_action
        })

    return pd.DataFrame(trades), np.array(equity_curve)


# -----------------------------
# Performance metrics
# -----------------------------
def compute_metrics(equity_curve):

    daily_returns = np.diff(equity_curve) / equity_curve[:-1]

    if np.std(daily_returns) == 0:
        sharpe = 0
    else:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()

    return sharpe, max_dd



from numpy.random import default_rng

def bootstrap_sharpe_test(ret_a, ret_b, n_boot=1000, seed=42):
    """
    Bootstrap test comparing Sharpe ratios of two strategies.
    ret_a = daily returns of strategy A
    ret_b = daily returns of strategy B
    """

    rng = default_rng(seed)

    n = min(len(ret_a), len(ret_b))

    ret_a = ret_a[:n]
    ret_b = ret_b[:n]

    sharpe_diffs = []

    for _ in range(n_boot):

        idx = rng.integers(0, n, n)

        sample_a = ret_a[idx]
        sample_b = ret_b[idx]

        sharpe_a = np.mean(sample_a) / (np.std(sample_a) + 1e-9) * np.sqrt(252)
        sharpe_b = np.mean(sample_b) / (np.std(sample_b) + 1e-9) * np.sqrt(252)

        sharpe_diffs.append(sharpe_a - sharpe_b)

    sharpe_diffs = np.array(sharpe_diffs)

    p_value = np.mean(sharpe_diffs <= 0)

    return np.mean(sharpe_diffs), p_value
# -----------------------------
# Run experiments
# -----------------------------
results = []

for cost in TRANSACTION_COSTS:

    print("\n==============================")
    print("Transaction cost:", cost)
    print("==============================")

    for model in MODELS:

        print(f"Running baseline for {model}")

        preds = np.load(f"{ARTIFACT_DIR}/{model}_preds.npy")

        trades_df, equity_curve = run_backtest(
            preds,
            returns,
            transaction_cost=cost
        )

        sharpe, max_dd = compute_metrics(equity_curve)

        final_equity = equity_curve[-1]

        num_trades = (trades_df["action"] != "hold").sum()

        trades_df.to_csv(
            f"{ARTIFACT_DIR}/{model}_baseline_cost_{cost}.csv",
            index=False
        )

        results.append({
            "model": model,
            "transaction_cost": cost,
            "final_equity": final_equity,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "trades": num_trades
        })

        print("Final equity:", final_equity)
        print("Sharpe:", sharpe)
        print("Max drawdown:", max_dd)
        print("Trades:", num_trades)


# -----------------------------
# Save results
# -----------------------------
results_df = pd.DataFrame(results)

results_df.to_csv(
    f"{ARTIFACT_DIR}/baseline_results_all_costs.csv",
    index=False
)

print("\nSaved baseline results for all transaction costs.")


# -----------------------------
# Plot equity curves
# -----------------------------
plt.figure(figsize=(10,6))

for cost in TRANSACTION_COSTS:

    rf = pd.read_csv(f"{ARTIFACT_DIR}/rf_baseline_cost_{cost}.csv")
    plt.plot(rf["equity"], label=f"RF cost {cost}")

plt.legend()
plt.title("RF Baseline Equity Curves (Different Costs)")
plt.xlabel("Time")
plt.ylabel("Equity")
plt.grid()

plt.show()