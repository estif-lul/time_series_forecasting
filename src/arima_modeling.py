from typing import Dict
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt

def load_data(ticker):

    df = pd.read_csv(f'data/{ticker}.csv', parse_dates=['Date'], index_col='Date')
    return df

def test_stationarity(series: pd.Series) -> Dict[str, float]:
    """
    Perform Augmented Dickey-Fuller test to check stationarity.
    
    Args:
        series (pd.Series): Time series data.
    
    Returns:
        dict: ADF statistic, p-value, and critical values.
    """
    try:
        result = adfuller(series.dropna())
        return {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4]
        }
    except Exception as e:
        print(f"Error performing ADF test: {e}")
        return {}
def train_model(df, ticker):
    """
    Performs ARIMA time series forecasting on the 'Close' column of the provided DataFrame for a given ticker symbol.
    The function tests for stationarity using the Augmented Dickey-Fuller (ADF) test, applies differencing if necessary,
    splits the data into training and testing sets, automatically selects the best ARIMA parameters, fits the ARIMA model,
    generates forecasts, evaluates the forecast using MAE and RMSE, and saves a plot comparing actual vs. forecasted values.
    Args:
        df (pd.DataFrame): DataFrame containing a 'Close' column with time series data indexed by date.
        ticker (str): The ticker symbol for the time series being forecasted (used for labeling and saving plots).
    Returns:
        None. Prints evaluation metrics and saves a forecast plot to the 'plots' directory.
    """

    adf_results = test_stationarity(df['Close'])
    print(f"ADF Test Results for {ticker}:")
    for key, value in adf_results.items():
        print(f"{key}: {value}")

    # If non-stationary, apply differencing
    if adf_results['p-value'] > 0.05:
        df['Close_diff'] = df['Close'].diff()
        series_to_model = df['Close_diff'].dropna()
    else:
        series_to_model = df['Close']


    # Train-Test Split
    train= series_to_model[:'2023-12-31']
    test = series_to_model['2024-01-01':]

    # Auto ARIMA to Find Best Parameters
    model_auto = auto_arima(train, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    print(model_auto.summary())

    # Fit ARIMA Model
    order = model_auto.order
    model = ARIMA(train, order=order)
    model_fit = model.fit()

    # Forcasting
    forecast = model_fit.forecast(steps=len(test))
    # forecast = pd.Series(forecast, index=test.index)

    # Evaluation
    mae = mean_absolute_error(test, forecast)
    rmsa = root_mean_squared_error(test, forecast)
    print(f'{ticker} --> MAE: {mae:.2f}, RMSE: {rmsa:.2f}')

    # Plot Forecast vs Actual
    plt.figure(figsize=(12,6))
    plt.plot(test.index, test, label='Actual')
    plt.plot(test.index, forecast, label='Forecast', linestyle='--')
    plt.title('ARIMA Forecast vs Actual')
    plt.legend()
    plt.savefig(f'plots/{ticker}_ARIMA_forecast.png')

def main():
    tickers = ['TSLA', 'BND', 'SPY']
    for ticker in tickers:
        df = load_data(ticker)
        if df.empty:
            print(f"No data found for {ticker}.")
            continue
        train_model(df, ticker)
        

if __name__ == "__main__":
    main()
