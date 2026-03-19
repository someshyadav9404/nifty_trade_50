import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "data/labeled/nifty_labeled.csv"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

# Use same test period as your ML models
test_df = df[df["Date"] >= "2020-01-01"].reset_index(drop=True)

dates = test_df["Date"]
close = test_df["Close"]

# -----------------------------
# Compute Buy & Hold
# -----------------------------
returns = close.pct_change().fillna(0)

equity_curve = (1 + returns).cumprod()

buy_hold_df = pd.DataFrame({
    "date": dates,
    "close": close,
    "return": returns,
    "equity": equity_curve
})

# save results
buy_hold_df.to_csv(
    "artifacts/paper1/buy_hold_equity.csv",
    index=False
)

print("Final Buy & Hold Equity:", equity_curve.iloc[-1])
print("Total Return %:", (equity_curve.iloc[-1] - 1) * 100)

# -----------------------------
# Plot Buy & Hold
# -----------------------------
plt.figure(figsize=(10,6))

plt.plot(dates, equity_curve, label="Buy & Hold NIFTY", color="black")

plt.title("Buy & Hold Equity Curve (NIFTY)")
plt.xlabel("Date")
plt.ylabel("Equity (Start = 1)")
plt.legend()
plt.grid(True)

plt.show()