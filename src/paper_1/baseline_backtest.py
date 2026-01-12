import numpy as np
import pandas as pd
import os

ART_DIR = "artifacts/paper1"

# -----------------------------
# Load artifacts
# -----------------------------
preds = np.load(f"{ART_DIR}/preds.npy")          # {-1,0,1}
confidence = np.load(f"{ART_DIR}/confidence.npy")
dates = pd.read_csv(f"{ART_DIR}/dates.csv")

df = pd.read_csv("data/labeled/nifty_labeled.csv")
df["Date"] = pd.to_datetime(df["Date"])

test_df = df[df["Date"] >= "2020-01-01"].reset_index(drop=True)

prices = test_df["Close"].values

assert len(prices) == len(preds)

# -----------------------------
# Backtest parameters
# -----------------------------
COST = 0.001   # 0.1% per trade

equity = [1.0]
trade_log = []

# -----------------------------
# Backtest loop
# -----------------------------
for i in range(len(preds) - 1):

    signal = preds[i]

    # No trade
    if signal == 0:
        equity.append(equity[-1])
        continue

    entry_price = prices[i]
    exit_price = prices[i + 1]

    raw_ret = signal * (exit_price - entry_price) / entry_price
    net_ret = raw_ret - COST

    new_equity = equity[-1] * (1 + net_ret)
    equity.append(new_equity)

    trade_log.append({
        "idx":i,
        "date": dates.iloc[i]["Date"],
        "signal": signal,
        "entry": entry_price,
        "exit": exit_price,
        "return": net_ret,
        "equity": new_equity,
        "confidence": confidence[i]
    })

equity = np.array(equity)
trades = pd.DataFrame(trade_log)

print("Number of trades:", len(trades))
print("Final equity:", equity[-1])

# -----------------------------
# Performance metrics
# -----------------------------
returns = np.diff(equity) / equity[:-1]

sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)

drawdown = equity / np.maximum.accumulate(equity) - 1
max_dd = drawdown.min()

print("Sharpe Ratio:", round(sharpe, 3))
print("Max Drawdown:", round(max_dd, 3))

# -----------------------------
# Save trade logs
# -----------------------------
TRADE_LOG_PATH = "artifacts/paper1/baseline_trades.csv"

trades.to_csv(TRADE_LOG_PATH, index=False)

print("Trade logs saved to:", TRADE_LOG_PATH)
