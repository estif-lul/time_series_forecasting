import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def load_data(ticker):
    df = pd.read_csv(f'data/{ticker}.csv', parse_dates=['Date'], index_col='Date')
    return df

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def forcast(df, ticker):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])

    window_size = 60
    X, y = create_sequences(scaled_data, window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train-Test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the model
    model.fit(X_train, y_train, epochs=20, batch_size=32)

    model.save(f'models/{ticker}_lstm_model.h5')
    # Predict and Inverse Transform
    predictions = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = root_mean_squared_error(actual_prices, predicted_prices)
    print(f'{ticker} --> MAE: {mae:.2f}, RMSE: {rmse:.2f}')

    # Plot Forecast vs Actual
    plt.figure(figsize=(12,6))
    plt.plot(actual_prices, label='Actual')
    plt.plot(predicted_prices, label='Forecast', linestyle='--')
    plt.title('LSTM Forecast vs Actual')
    plt.legend()
    plt.savefig(f'plots/{ticker}_LSTM_forecast.png')



def main():
    tickers = ['TSLA', 'BND', 'SPY']
    for ticker in tickers:
        df = load_data(ticker)
        if df.empty:
            print(f"No data found for {ticker}.")
            continue
        forcast(df, ticker)

if __name__ == "__main__":
    main()
