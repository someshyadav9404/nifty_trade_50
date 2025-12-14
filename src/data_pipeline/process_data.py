import pandas as pd
import os
from feature_engineering import add_features
from label_data import label_data

RAW_PATH = "data/raw/nifty_clean.csv"
OUT_PATH = "data/labeled/nifty_labeled.csv"

def process():
    df = pd.read_csv(RAW_PATH, parse_dates=["Date"])

    # Convert numeric columns from string to float
    df[["Open","High","Low","Close","Volume"]] = df[["Open","High","Low","Close","Volume"]].apply(pd.to_numeric, errors="coerce")

    # df = df.set_index("date")

    df = add_features(df)
    df = label_data(df)

    df = df.dropna()
    os.makedirs("data/labeled", exist_ok=True)
    df.to_csv(OUT_PATH,index=False)

    print("Processed + labeled dataset saved:", OUT_PATH)
    print(df.tail())

if __name__ == "__main__":
    process()
