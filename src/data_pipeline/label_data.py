def label_data(df):
    df["next_ret"] = df["Close"].pct_change().shift(-1)

    df["label"] = 0
    df.loc[df["next_ret"] > 0.003, "label"] = 1
    df.loc[df["next_ret"] < -0.003, "label"] = -1

    return df

