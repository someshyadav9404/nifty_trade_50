import pandas as pd
import yfinance as yf
import os

def fetch_nifty():
    df = yf.download("^NSEI", period="max", interval="1d", progress=False)

    # Reset index to have Date column
    df = df.reset_index()

    # Remove first 3 rows (if they exist)
    df = df.iloc[3:].reset_index(drop=True)

    # # Force correct column order & add exact header names
    # df = df.rename(columns={
    #     "Date": "Date",
    #     "Close": "Close",
    #     "High": "High",
    #     "Low": "Low",
    #     "Open": "Open",
    #     "Volume": "Volume"
    # })

    df = df[["Date", "Close", "High", "Low", "Open", "Volume"]]

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/nifty_clean.csv", index=False)

    print("🔥 Clean file created: data/raw/nifty_clean.csv")
    print(df.head())

if __name__ == "__main__":
    fetch_nifty()
