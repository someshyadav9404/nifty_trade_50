import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load data
# -----------------------------
trades = pd.read_csv("artifacts/paper1/trades_metadata.csv")
shap_trades = np.load("artifacts/paper1/trade_shap_per_trade.npy")

assert len(trades) == shap_trades.shape[0]

# -----------------------------
# Split winners and losers
# -----------------------------
winner_mask = trades["return"] > 0
loser_mask  = trades["return"] <= 0

shap_winners = shap_trades[winner_mask.values]
shap_losers  = shap_trades[loser_mask.values]

print("Winner SHAP shape:", shap_winners.shape)
print("Loser SHAP shape :", shap_losers.shape)

# -----------------------------
# Compute centroids
# -----------------------------
winner_centroid = shap_winners.mean(axis=0)
loser_centroid  = shap_losers.mean(axis=0)

# -----------------------------
# Explanation-aware filter
# -----------------------------
keep_trade = []

for i in range(len(shap_trades)):
    s = shap_trades[i].reshape(1, -1)

    sim_win = cosine_similarity(s, winner_centroid.reshape(1, -1))[0][0]
    sim_lose = cosine_similarity(s, loser_centroid.reshape(1, -1))[0][0]

    # Core decision rule
    keep_trade.append(sim_win > sim_lose)

trades["keep"] = keep_trade

print("Trades kept:", trades["keep"].sum())
print("Trades rejected:", (~trades["keep"]).sum())

# -----------------------------
# Save filtered trades
# -----------------------------
trades.to_csv("artifacts/paper1/filtered_trades.csv", index=False)
