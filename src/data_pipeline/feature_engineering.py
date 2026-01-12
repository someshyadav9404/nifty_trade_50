def add_features(df):
    df["ret_1d"] = df["Close"].pct_change()
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_diff"] = df["ma_5"] - df["ma_20"]
    df["vol_20"] = df["Close"].rolling(20).std()
    df["momentum"] = df["Close"] - df["Close"].shift(5)
    return df

