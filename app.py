import gradio as gr
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models, EfficientFrontier
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# --- Forecasting ---
def forecast_prices(ticker, model_type, horizon):
    data = yf.download(ticker, start="2015-01-01", end="2025-07-31")["Close"]
    data = data.dropna()

    if model_type == "ARIMA":
        model = ARIMA(data, order=(5,1,0))
        fit = model.fit()
        forecast = fit.forecast(steps=horizon)
    else:
        forecast = pd.Series(np.linspace(data.iloc[-1], data.iloc[-1]*1.1, horizon))  # placeholder for LSTM

    fig, ax = plt.subplots()
    data.plot(ax=ax, label="Historical")
    forecast.plot(ax=ax, label="Forecast", color="orange")
    ax.legend()
    ax.set_title(f"{model_type} Forecast for {ticker}")
    return fig

# --- Optimization ---
def optimize_portfolio(start_date, tsla_ret, spy_ret, bnd_ret, risk_model):
    tickers = ["TSLA", "SPY", "BND"]
    data = yf.download(tickers, start=start_date, end="2025-07-31")["Close"]
    mu = pd.Series({"TSLA": tsla_ret, "SPY": spy_ret, "BND": bnd_ret})
    S = risk_models.sample_cov(data) if risk_model == "Sample Covariance" else risk_models.semicovariance(data)
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    perf = ef.portfolio_performance()
    return weights, f"Return: {perf[0]:.2%}, Volatility: {perf[1]:.2%}, Sharpe: {perf[2]:.2f}"

# --- Backtesting ---
def backtest_portfolio(start_date, end_date, weights):
    tickers = list(weights.keys())
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    returns = data.pct_change().dropna()
    weighted_returns = returns.dot(pd.Series(weights))
    cumulative = (1 + weighted_returns).cumprod()

    fig, ax = plt.subplots()
    cumulative.plot(ax=ax, label="Portfolio")
    ax.set_title("Backtest: Cumulative Returns")
    ax.legend()
    return fig, f"Final Value: {cumulative.iloc[-1]:.2f}"

# --- Gradio Layout ---
with gr.Blocks(title="Advanced Portfolio Dashboard") as demo:
    gr.Markdown("# 📊 Portfolio Forecasting & Optimization")

    with gr.Tab("Forecasting"):
        ticker = gr.Dropdown(["TSLA", "SPY", "BND"], label="Select Asset")
        model_type = gr.Radio(["ARIMA", "LSTM"], label="Model")
        horizon = gr.Slider(5, 60, value=30, label="Forecast Horizon (days)")
        forecast_btn = gr.Button("Run Forecast")
        forecast_plot = gr.Plot()

        forecast_btn.click(forecast_prices, inputs=[ticker, model_type, horizon], outputs=forecast_plot)

    with gr.Tab("Optimization"):
        gr.Markdown("### Input Forecasted Returns")
        start_date_opt = gr.Textbox(value="2015-01-01", label="Start Date for Historical Data")
        tsla_ret = gr.Slider(0.0, 0.2, value=0.05, label="TSLA Return")
        spy_ret = gr.Slider(0.0, 0.2, value=0.07, label="SPY Return")
        bnd_ret = gr.Slider(0.0, 0.2, value=0.03, label="BND Return")
        risk_model = gr.Radio(["Sample Covariance", "Semicovariance"], label="Risk Model")
        opt_btn = gr.Button("Optimize Portfolio")
        weights_json = gr.JSON(label="Optimal Weights")
        perf_text = gr.Text(label="Performance")

        opt_btn.click(optimize_portfolio, 
                      inputs=[start_date_opt, tsla_ret, spy_ret, bnd_ret, risk_model], 
                      outputs=[weights_json, perf_text])

    with gr.Tab("Backtesting"):
        start_date = gr.Textbox(value="2020-01-01", label="Start Date")
        end_date = gr.Textbox(value="2025-07-31", label="End Date")
        weights_input = gr.Textbox(value='{"TSLA": 0.3, "SPY": 0.5, "BND": 0.2}', label="Portfolio Weights (JSON)")
        backtest_btn = gr.Button("Run Backtest")
        backtest_plot = gr.Plot()
        backtest_text = gr.Text()

        def parse_weights_and_backtest(start, end, weights_json_str):
            weights = eval(weights_json_str)
            return backtest_portfolio(start, end, weights)

        backtest_btn.click(parse_weights_and_backtest, inputs=[start_date, end_date, weights_input], outputs=[backtest_plot, backtest_text])

demo.launch()
