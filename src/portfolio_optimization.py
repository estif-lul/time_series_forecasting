import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
import matplotlib.pyplot as plt

# load historical data
tickers = ['TSLA', 'BND', 'SPY']
data = yf.download(tickers, start='2015-07-01', end='2025-07-31')['Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Estimate expected returns
# Use forecasted return for TSLA 
forecasted_tsla_return = 0.012 
historical_returns = expected_returns.mean_historical_return(data)

# Override TSLA return with forecast
historical_returns['TSLA'] = forecasted_tsla_return

# Calculate covariance matrix
cov_matrix = risk_models.sample_cov(data)

# Optimize portfolio
ef = EfficientFrontier(historical_returns, cov_matrix)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
performance = ef.portfolio_performance(verbose=True)

# Plot Efficient Frontier

ef_plot = EfficientFrontier(historical_returns, cov_matrix)
plotting.plot_efficient_frontier(ef_plot, show_assets=True)
plt.title("Efficient Frontier with Forecasted TSLA Return")
plt.savefig('plots/efficient_frontier.png')

# Display results
print("Optimal Weights:")
for asset, weight in cleaned_weights.items():
    print(f"{asset}: {weight:.2%}")

print(f"\nExpected Annual Return: {performance[0]:.2%}")
print(f"Annual Volatility: {performance[1]:.2%}")
print(f"Sharpe Ratio: {performance[2]:.2f}")