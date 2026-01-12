"""
Utility for loading frozen XGBoost model (Paper-1)
"""

import joblib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_model():
    model_path = PROJECT_ROOT / "models" / "xgboost_paper1_frozen.pkl"
    return joblib.load(model_path)
