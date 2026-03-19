import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Configuration
# -----------------------------
ART = "artifacts/paper1"
DATA_PATH = "data/labeled/nifty_labeled.csv"

MODELS = ["rf","xgb"]
TRANSACTION_COSTS = [0.00025,0.0005,0.001]


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

train_df = df[df["Date"] < "2020-01-01"].reset_index(drop=True)
test_df  = df[df["Date"] >= "2020-01-01"].reset_index(drop=True)

train_ret = train_df["next_ret"].values
test_ret  = test_df["next_ret"].values


# -----------------------------
# Normalize SHAP vectors
# -----------------------------
def normalize(X):
    norm = np.linalg.norm(X,axis=1,keepdims=True)
    return X/(norm+1e-9)


# -----------------------------
# Backtest Engine
# -----------------------------
def run_backtest(signals, returns, dates, transaction_cost):

    position = 0
    equity = 1

    equity_curve=[]
    trade_log=[]

    for i,(s,r) in enumerate(zip(signals,returns)):

        cost=0
        action="hold"

        if position==0 and s!=0:
            position=s
            cost+=transaction_cost
            action="enter"

        elif position==1 and s==-1:
            position=-1
            cost+=transaction_cost*2
            action="flip"

        elif position==-1 and s==1:
            position=1
            cost+=transaction_cost*2
            action="flip"

        pnl = position*r

        equity *= (1+pnl-cost)

        equity_curve.append(equity)

        trade_log.append({
            "date":dates.iloc[i],
            "signal":s,
            "position":position,
            "market_return":r,
            "pnl":pnl,
            "equity":equity,
            "action":action
        })

    return np.array(equity_curve), pd.DataFrame(trade_log)


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(eq):

    r = np.diff(eq)/eq[:-1]

    sharpe = 0 if np.std(r)==0 else np.mean(r)/np.std(r)*np.sqrt(252)

    dd = (eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq)

    return sharpe, dd.min()



from numpy.random import default_rng

def bootstrap_sharpe_test(ret_a, ret_b, n_boot=1000, seed=42):
    """
    Bootstrap test comparing Sharpe ratios of two strategies.
    ret_a = daily returns of strategy A
    ret_b = daily returns of strategy B
    """

    rng = default_rng(seed)

    n = min(len(ret_a), len(ret_b))

    ret_a = ret_a[:n]
    ret_b = ret_b[:n]

    sharpe_diffs = []

    for _ in range(n_boot):

        idx = rng.integers(0, n, n)

        sample_a = ret_a[idx]
        sample_b = ret_b[idx]

        sharpe_a = np.mean(sample_a) / (np.std(sample_a) + 1e-9) * np.sqrt(252)
        sharpe_b = np.mean(sample_b) / (np.std(sample_b) + 1e-9) * np.sqrt(252)

        sharpe_diffs.append(sharpe_a - sharpe_b)

    sharpe_diffs = np.array(sharpe_diffs)

    p_value = np.mean(sharpe_diffs <= 0)

    return np.mean(sharpe_diffs), p_value
# -----------------------------
# Main experiment
# -----------------------------
results=[]

for cost in TRANSACTION_COSTS:

    print("\n==============================")
    print("Transaction cost:",cost)
    print("==============================")

    for model in MODELS:

        print("\nRunning SHAP filter for",model)

        # -------------------------
        # Load artifacts
        # -------------------------
        train_shap = np.load(f"{ART}/{model}_train_trade_shap.npy")
        test_shap  = np.load(f"{ART}/{model}_trade_shap.npy")

        train_preds = np.load(f"{ART}/{model}_train_preds.npy")
        test_preds  = np.load(f"{ART}/{model}_preds.npy")

        # -------------------------
        # Normalize explanations
        # -------------------------
        train_shap = normalize(train_shap)
        test_shap  = normalize(test_shap)

        # -------------------------
        # Training trade outcomes
        # -------------------------
        pnl_train = train_preds * train_ret

        winners = train_shap[pnl_train > 0]
        losers  = train_shap[pnl_train <= 0]

        win_cent = winners.mean(axis=0)
        lose_cent = losers.mean(axis=0)

        # -------------------------
        # SHAP filtering
        # -------------------------
        filt_preds = test_preds.copy()

        for i,exp in enumerate(test_shap):

            s = exp.reshape(1,-1)

            sim_win  = cosine_similarity(s,win_cent.reshape(1,-1))[0,0]
            sim_lose = cosine_similarity(s,lose_cent.reshape(1,-1))[0,0]

            if sim_lose > sim_win:
                filt_preds[i] = 0

        # -------------------------
        # Backtest filtered signals
        # -------------------------
        equity_curve, trades_df = run_backtest(
            filt_preds,
            test_ret,
            test_df["Date"],
            cost
        )

        sharpe, max_dd = compute_metrics(equity_curve)

        final_equity = equity_curve[-1]

        trade_count = (trades_df["action"]!="hold").sum()

        # -------------------------
        # Save trades
        # -------------------------
        trades_df.to_csv(
            f"{ART}/{model}_shap_cost_{cost}.csv",
            index=False
        )

        print("Final equity:",final_equity)
        print("Sharpe:",sharpe)
        print("Max drawdown:",max_dd)
        print("Trades:",trade_count)

        results.append({
            "model":model,
            "transaction_cost":cost,
            "final_equity":final_equity,
            "sharpe":sharpe,
            "max_drawdown":max_dd,
            "trades":trade_count
        })


# -----------------------------
# Save experiment results
# -----------------------------
results_df = pd.DataFrame(results)

results_df.to_csv(
    f"{ART}/shap_filter_results_all_costs.csv",
    index=False
)

print("\nSHAP filtering results saved.")