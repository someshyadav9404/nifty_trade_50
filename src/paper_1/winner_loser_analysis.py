import pandas as pd
import numpy as np

# Load trade metadata
trades = pd.read_csv("artifacts/paper1/trades_metadata.csv")

# Load SHAP vectors
trade_shap = np.load("artifacts/paper1/trade_shap_per_trade.npy")

assert len(trades) == trade_shap.shape[0]

# Split winners and losers
winners_idx = trades["return"] > 0
losers_idx  = trades["return"] <= 0

shap_win = trade_shap[winners_idx.values]
shap_lose = trade_shap[losers_idx.values]

print("Winning trades:", shap_win.shape[0])
print("Losing trades :", shap_lose.shape[0])

# Mean explanation patterns
mean_shap_win = shap_win.mean(axis=0)
mean_shap_lose = shap_lose.mean(axis=0)

# Similarity
from sklearn.metrics.pairwise import cosine_similarity

sim = cosine_similarity(
    mean_shap_win.reshape(1, -1),
    mean_shap_lose.reshape(1, -1)
)[0][0]

print("Cosine similarity (winner vs loser explanations):", round(sim, 3))
