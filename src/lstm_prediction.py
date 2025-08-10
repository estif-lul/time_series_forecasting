from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

def predict(df, ticker):
    series = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series)

    time_step = 60
    model = load_model(f'models/{ticker}_lstm_model.h5')
    
    # Forecast next 189 days
    future_preds = []
    input_seq = scaled_series[-time_step:].reshape(1, time_step, 1)

    for _ in range(180):
        pred = model.predict(input_seq)
        future_preds.append(pred[0][0])
        input_seq = np.concatenate([input_seq[:, 1:, :], pred.reshape(1, 1, 1)], axis=1)

    # Inverse transform to get actual prices
    future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    # Plot forecast
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], label='Historical')
    future_dates = pd.date_range(start=df.index[-1], periods=181, freq='B')[1:]
    plt.plot(future_dates, future_prices, label='Forecast')
    plt.legend()
    plt.title(f'{ticker} Stock Price Forecast with LSTM')
    plt.savefig(f'plots/{ticker}_LSTM_predictions.png')

def load_data(ticker):
    df = pd.read_csv(f'data/{ticker}.csv', parse_dates=['Date'], index_col='Date')
    return df

def main():
    tickers = ['TSLA', 'BND', 'SPY']
    for ticker in tickers:
        df = load_data(ticker)
        if df.empty:
            print(f"No data found for {ticker}.")
            continue
        predict(df, ticker)

if __name__ == "__main__":
    main()