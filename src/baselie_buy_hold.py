import pandas as pd
from pathlib import Path

path = Path("data/raw/nifty_clean.csv")  # change file if needed
df = pd.read_csv(path)
# skip first row if it contains labels/metadata (do not compute on it)
if len(df) > 0:
    df = df.iloc[1:].reset_index(drop=True)
for col in ("close","Close","adj_close","Adj Close","ClosePrice","close_price"):
    if col in df.columns:
        closes = df[col].astype(float)
        break
else:
    raise SystemExit("No close column found; update column name")

buy_hold_pct = (closes.iloc[-1] / closes.iloc[0] - 1) * 100
print(f"Buy-and-hold return: {buy_hold_pct:.2f}%")