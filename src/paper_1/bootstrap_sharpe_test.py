import numpy as np
import pandas as pd
from bootstrap_utils import bootstrap_sharpe_test

ART = "artifacts/paper1"

MODELS = ["rf","xgb"]

COSTS = [0.00025,0.0005,0.001]

results = []

for model in MODELS:

    for cost in COSTS:

        print("\nModel:",model," Cost:",cost)

        baseline = pd.read_csv(f"{ART}/{model}_baseline_cost_{cost}.csv")
        confidence = pd.read_csv(f"{ART}/{model}_confidence_cost_{cost}.csv")
        shap = pd.read_csv(f"{ART}/{model}_shap_cost_{cost}.csv")

        base_eq = baseline["equity"].values
        conf_eq = confidence["equity"].values
        shap_eq = shap["equity"].values

        base_ret = np.diff(base_eq) / base_eq[:-1]
        conf_ret = np.diff(conf_eq) / conf_eq[:-1]
        shap_ret = np.diff(shap_eq) / shap_eq[:-1]

        # SHAP vs Baseline
        diff1, p1 = bootstrap_sharpe_test(shap_ret, base_ret)

        # SHAP vs Confidence
        diff2, p2 = bootstrap_sharpe_test(shap_ret, conf_ret)

        results.append({
            "model":model,
            "cost":cost,
            "comparison":"SHAP vs Baseline",
            "sharpe_diff":diff1,
            "p_value":p1
        })

        results.append({
            "model":model,
            "cost":cost,
            "comparison":"SHAP vs Confidence",
            "sharpe_diff":diff2,
            "p_value":p2
        })

        print("SHAP vs Baseline  p-value:",p1)
        print("SHAP vs Confidence p-value:",p2)

pd.DataFrame(results).to_csv(
    f"{ART}/bootstrap_stat_tests.csv",
    index=False
)

print("\nBootstrap statistical tests saved.")