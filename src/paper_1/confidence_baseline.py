import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ARTIFACT_DIR = "artifacts/paper1"
DATA_PATH = "data/labeled/nifty_labeled.csv"

MODELS = ["rf", "xgb"]
TRANSACTION_COSTS = [0.00025, 0.0005, 0.001]


# ----------------------------------
# Load data
# ----------------------------------
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

train_df = df[df["Date"] < "2020-01-01"].reset_index(drop=True)
test_df  = df[df["Date"] >= "2020-01-01"].reset_index(drop=True)

train_returns = train_df["next_ret"].values
test_returns  = test_df["next_ret"].values

train_dates = train_df["Date"]
test_dates  = test_df["Date"]


# ----------------------------------
# Backtest engine
# ----------------------------------
def run_backtest(preds, returns, dates, transaction_cost):

    position = 0
    equity = 1.0

    equity_curve = []
    trade_log = []

    for i, (signal, r) in enumerate(zip(preds, returns)):

        cost = 0
        action = "hold"

        if position == 0 and signal in [-1, 1]:

            position = signal
            cost += transaction_cost
            action = "enter"

        elif position == 1 and signal == -1:

            position = -1
            cost += transaction_cost * 2
            action = "flip"

        elif position == -1 and signal == 1:

            position = 1
            cost += transaction_cost * 2
            action = "flip"

        pnl = position * r

        equity *= (1 + pnl - cost)

        equity_curve.append(equity)

        trade_log.append({
            "date": dates.iloc[i],
            "signal": signal,
            "position": position,
            "market_return": r,
            "pnl": pnl,
            "equity": equity,
            "action": action
        })

    return pd.DataFrame(trade_log), np.array(equity_curve)


# ----------------------------------
# Performance metrics
# ----------------------------------
def compute_metrics(eq):

    daily_ret = np.diff(eq) / eq[:-1]

    if np.std(daily_ret) == 0:
        sharpe = 0
    else:
        sharpe = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252)

    running_max = np.maximum.accumulate(eq)
    drawdown = (eq - running_max) / running_max

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
# ----------------------------------
# Run experiments
# ----------------------------------
results = []

for cost in TRANSACTION_COSTS:

    print("\n==============================")
    print("Transaction cost:", cost)
    print("==============================")

    for model in MODELS:

        print("\nProcessing model:", model)

        train_preds = np.load(f"{ARTIFACT_DIR}/{model}_train_preds.npy")
        test_preds  = np.load(f"{ARTIFACT_DIR}/{model}_preds.npy")

        train_conf = np.load(f"{ARTIFACT_DIR}/{model}_train_confidence.npy")
        test_conf  = np.load(f"{ARTIFACT_DIR}/{model}_confidence.npy")

        # ----------------------------------
        # Candidate thresholds from training confidences
        # ----------------------------------
        thresholds = np.unique(np.round(train_conf, 3))

        best_sharpe = -999
        best_threshold = None

        # ----------------------------------
        # Search best threshold on TRAIN
        # ----------------------------------
        for th in thresholds:

            preds = train_preds.copy()
            preds[train_conf < th] = 0

            trades_df, eq = run_backtest(
                preds,
                train_returns,
                train_dates,
                cost
            )

            sharpe, _ = compute_metrics(eq)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_threshold = th

        print("Best threshold:", best_threshold)
        print("Train Sharpe:", best_sharpe)

        # ----------------------------------
        # Apply threshold on TEST
        # ----------------------------------
        filtered_preds = test_preds.copy()
        filtered_preds[test_conf < best_threshold] = 0

        trades_df, equity_curve = run_backtest(
            filtered_preds,
            test_returns,
            test_dates,
            cost
        )

        sharpe, max_dd = compute_metrics(equity_curve)

        final_equity = equity_curve[-1]

        num_trades = (trades_df["action"] != "hold").sum()

        # save trades
        trades_df.to_csv(
            f"{ARTIFACT_DIR}/{model}_confidence_cost_{cost}.csv",
            index=False
        )

        print("Final equity:", final_equity)
        print("Sharpe:", sharpe)
        print("Max drawdown:", max_dd)
        print("Trades:", num_trades)

        results.append({
            "model": model,
            "transaction_cost": cost,
            "threshold": best_threshold,
            "final_equity": final_equity,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "trades": num_trades
        })


# ----------------------------------
# Save experiment results
# ----------------------------------
results_df = pd.DataFrame(results)

results_df.to_csv(
    f"{ARTIFACT_DIR}/confidence_filter_results_all_costs.csv",
    index=False
)

print("\nConfidence filter results saved.")


# ----------------------------------
# Plot equity curves
# ----------------------------------
plt.figure(figsize=(10,6))

for cost in TRANSACTION_COSTS:

    df_trades = pd.read_csv(
        f"{ARTIFACT_DIR}/rf_confidence_cost_{cost}.csv"
    )

    plt.plot(df_trades["equity"], label=f"RF cost {cost}")

plt.title("RF Confidence Filter Equity Curves")
plt.xlabel("Time")
plt.ylabel("Equity")
plt.legend()
plt.grid()

plt.show()