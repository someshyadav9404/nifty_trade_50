import numpy as np
import pandas as pd


def sharpe_ratio(returns):
    if np.std(returns) == 0:
        return 0
    return np.mean(returns) / np.std(returns) * np.sqrt(252)


def max_drawdown(equity_curve):
    cumulative_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    return np.min(drawdown)


def confidence_filter(
    model,
    X_test,
    baseline_signals,
    returns,
    shap_trade_count,
    transaction_cost=0.001
):
    """
    model: trained classifier (RF or XGB)
    X_test: test features
    baseline_signals: predicted trade signals (-1, 0, +1)
    returns: next-day returns (aligned with X_test)
    shap_trade_count: number of trades executed by SHAP filter
    transaction_cost: per trade cost (default 0.1%)
    """

    # Step 1: Get predicted probabilities
    probs = model.predict_proba(X_test)

    # Step 2: Get predicted labels
    pred_labels = model.predict(X_test)

    # Step 3: Compute confidence for predicted class
    confidence = probs[np.arange(len(pred_labels)), pred_labels]

    # Step 4: Sort trades by confidence (descending)
    sorted_indices = np.argsort(confidence)[::-1]

    # Step 5: Select top N trades
    selected_indices = sorted_indices[:shap_trade_count]

    # Create mask
    mask = np.zeros(len(confidence), dtype=bool)
    mask[selected_indices] = True

    # Step 6: Apply filtering
    filtered_signals = baseline_signals.copy()
    filtered_signals[~mask] = 0  # reject other trades

    # Step 7: Compute strategy returns
    strategy_returns = filtered_signals * returns

    # Apply transaction cost only when trade executed
    trade_mask = filtered_signals != 0
    strategy_returns[trade_mask] -= transaction_cost

    # Step 8: Compute equity curve
    equity_curve = np.cumprod(1 + strategy_returns)

    # Step 9: Compute metrics
    sharpe = sharpe_ratio(strategy_returns)
    mdd = max_drawdown(equity_curve)
    final_equity = equity_curve[-1]
    total_trades = np.sum(trade_mask)

    results = {
        "Final Equity": final_equity,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": mdd,
        "Total Trades": total_trades,
        "Returns": strategy_returns
    }

    return results

