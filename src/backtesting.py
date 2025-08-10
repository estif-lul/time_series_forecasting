import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Download data for backtesting period
tickers = ['TSLA', 'BND', 'SPY']
data = yf.download(tickers, start='2024-08-01', end='2025-07-31')['Close']
returns = data.pct_change().dropna()

# Define portfolio weights
# Strategy portfolio from Task 4
strategy_weights = {'TSLA': 0.0, 'BND': 0.53, 'SPY': 0.46}

# Benchmark portfolio
benchmark_weights = {'TSLA': 0.0, 'BND': 0.40, 'SPY': 0.60}

# Calculate daily portfolio returns
def portfolio_returns(weights, returns_df):
    weights_array = np.array([weights[ticker] for ticker in returns_df.columns])
    return returns_df.dot(weights_array)

strategy_daily = portfolio_returns(strategy_weights, returns)
benchmark_daily = portfolio_returns(benchmark_weights, returns)

# Calculate cumulative returns
strategy_cum = (1 + strategy_daily).cumprod()
benchmark_cum = (1 + benchmark_daily).cumprod()

# Plot performance
plt.figure(figsize=(12,6))
plt.plot(strategy_cum, label='Strategy Portfolio')
plt.plot(benchmark_cum, label='Benchmark Portfolio (60% SPY / 40% BND)')
plt.title('Backtest: Strategy vs Benchmark')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.savefig('plots/backtest_strategy_vs_benchmark.png')

# Evaluate performance
def sharpe_ratio(daily_returns, risk_free_rate=0.0):
    excess_returns = daily_returns - risk_free_rate / 252
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

strategy_total_return = strategy_cum.iloc[-1] - 1
benchmark_total_return = benchmark_cum.iloc[-1] - 1

strategy_sharpe = sharpe_ratio(strategy_daily)
benchmark_sharpe = sharpe_ratio(benchmark_daily)

print(f"Strategy Total Return: {strategy_total_return:.2%}")
print(f"Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
print(f"Benchmark Total Return: {benchmark_total_return:.2%}")
print(f"Benchmark Sharpe Ratio: {benchmark_sharpe:.2f}")
