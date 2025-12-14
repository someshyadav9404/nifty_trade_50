import pandas as pd
import numpy as np

# def backtest_strategy(df, preds, initial_capital=100000, cost_perc=0.0005):
#     """
#     Backtest Buy/Hold/Sell strategy.
#     cost_perc = transaction cost (e.g., 0.0005 = 0.05%)
#     """

#     capital = initial_capital
#     position = 0     # 0 = no position, 1 = long
#     entry_price = 0

#     equity_curve = []  # to record daily values

#     close_prices = df["Close"].values

#     for i in range(len(preds)):
#         action = preds[i]

#         # --- BUY ACTION ---
#         if action == 1 and position == 0:
#             position = 1
#             entry_price = close_prices[i]
#             capital -= capital * cost_perc  # transaction cost

#         # --- SELL ACTION ---
#         elif action == -1 and position == 1:
#             pnl = (close_prices[i] - entry_price) / entry_price
#             capital *= (1 + pnl)
#             capital -= capital * cost_perc  # transaction cost
#             position = 0

#         # --- HOLD ACTION ---
#         # do nothing

#         # Mark-to-market equity
#         if position == 1:
#             equity_curve.append(capital * (close_prices[i] / entry_price))
#         else:
#             equity_curve.append(capital)

#     equity_curve = np.array(equity_curve)

#     # ==== Metrics ====
#     total_return = (equity_curve[-1] / initial_capital - 1) * 100

#     # daily returns
#     daily_ret = np.diff(equity_curve) / equity_curve[:-1]
#     sharpe = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252) if np.std(daily_ret) != 0 else 0

#     # max drawdown
#     rolling_max = np.maximum.accumulate(equity_curve)
#     dd = (equity_curve - rolling_max) / rolling_max
#     max_dd = dd.min() * 100

#     return {
#         "final_equity": equity_curve[-1],
#         "total_return": total_return,
#         "sharpe_ratio": sharpe,
#         "max_drawdown": max_dd,
#         "equity_curve": equity_curve
#     }
def backtest_strategy(
    df,
    preds,
    initial_capital=100_000,
    cost_perc=0.0005,
    position_size=1.0
):
    close = df["Close"].values
    n = len(preds)

    cash = initial_capital
    position = 0
    units = 0

    equity_curve = []
    trade_log = []

    for i in range(n):
        price = close[i]
        signal = preds[i]

        if signal == 1 and position == 0:
            invest_amt = cash * position_size
            units = invest_amt / price
            cash -= invest_amt * cost_perc
            position = 1

            trade_log.append({
                "type": "BUY",
                "date": df.index[i],
                "price": price
            })

        elif signal == -1 and position == 1:
            proceeds = units * price
            cash = proceeds - proceeds * cost_perc
            units = 0
            position = 0

            trade_log.append({
                "type": "SELL",
                "date": df.index[i],
                "price": price
            })

        equity_curve.append(cash + units * price)

    # Force exit at end
    if position == 1:
        proceeds = units * close[-1]
        cash = proceeds - proceeds * cost_perc
        equity_curve[-1] = cash

    equity_curve = np.array(equity_curve)

    daily_ret = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = (
        np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252)
        if np.std(daily_ret) != 0 else 0
    )

    rolling_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - rolling_max) / rolling_max

    return {
        "final_equity": equity_curve[-1],
        "total_return_%": (equity_curve[-1] / initial_capital - 1) * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_%": drawdown.min() * 100,
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown,
        "trade_log": pd.DataFrame(trade_log)
    }