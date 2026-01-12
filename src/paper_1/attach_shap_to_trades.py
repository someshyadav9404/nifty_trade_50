import numpy as np
import pandas as pd

trades = pd.read_csv("artifacts/paper1/baseline_trades.csv")
trade_shap = np.load("artifacts/paper1/trade_shap.npy")

# Extract SHAP vectors for each trade
trade_shap_vectors = np.array([
    trade_shap[int(idx)]
    for idx in trades["idx"]
])

# Save separately
np.save("artifacts/paper1/trade_shap_per_trade.npy", trade_shap_vectors)

# Save trade metadata WITHOUT shap vectors
trades.to_csv("artifacts/paper1/trades_metadata.csv", index=False)

print("Saved:")
print("- trades_metadata.csv")
print("- trade_shap_per_trade.npy")
