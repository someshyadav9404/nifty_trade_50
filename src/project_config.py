from pathlib import Path

# src directory
SRC_DIR = Path(__file__).resolve().parent

# project root
BASE_DIR = SRC_DIR.parent

# data paths
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA = DATA_DIR / "labeled" / "nifty_labeled.csv"

# model paths
MODEL_DIR = BASE_DIR / "models"
RF_MODEL_PATH = MODEL_DIR / "random_forest.joblib"
XGB_MODEL_PATH = MODEL_DIR / "xgboost.joblib"

# features
FEATURE_COLS = [
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    "ret_1d",
    "ma_5",
    "ma_20",
    "ma_diff",
    "vol_20",
    "momentum",
    "next_ret",
]


# backtesting parameters
INITIAL_CAPITAL = 100000
TRANSACTION_COST = 0.0005
