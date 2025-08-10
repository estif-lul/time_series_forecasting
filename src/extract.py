import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from typing import Dict

def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical financial data from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol.
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DataFrame: Historical price data.
    """
    try:
        # Download data from Yahoo Finance
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError(f"No data returned for {ticker}. Check ticker or date range.")
        data.reset_index(inplace=True)
        data['Ticker'] = ticker
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the financial data.
    
    Args:
        df (pd.DataFrame): Raw financial data.
    
    Returns:
        pd.DataFrame: Cleaned data.
    """
    try:
        df.info()
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)
        # Convert 'Date' to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return pd.DataFrame()
    
def add_features(ticker, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add daily returns and rolling volatility features.
    
    Args:
        df (pd.DataFrame): Cleaned financial data.
    
    Returns:
        pd.DataFrame: Data with additional features.
    """
    try:
        # Add daily returns and rolling statistics
        df[ticker]['Daily Return'] = df['Close'].pct_change()
        df[ticker]['Rolling Mean'] = df['Daily Return'].rolling(window=21).mean()
        df[ticker]['Rolling Std'] = df['Daily Return'].rolling(window=21).std()
        return df
    except Exception as e:
        print(f"Error adding features: {e}")
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
        # Perform Augmented Dickey-Fuller test
        result = adfuller(series.dropna())
        return {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4]
        }
    except Exception as e:
        print(f"Error performing ADF test: {e}")
        return {}
    
def save_dataframe(df: pd.DataFrame, filename: str, filetype: str = 'csv') -> None:
    """
    Save a pandas DataFrame to a file in CSV or Excel format.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the output file (without extension).
        filetype (str): The file format to save ('csv' or 'excel'). Default is 'csv'.

    Returns:
        None
    """
    try:
        df.columns = ['_'.join(map(str, col)).strip() if isinstance(col, tuple) else col for col in df.columns]
        if filetype.lower() == 'csv':
            df.to_csv(f"data/{filename}.csv", index=True)
            print(f"✅ DataFrame saved as {filename}.csv")
        elif filetype.lower() == 'excel':
            df.to_excel(f"data/{filename}.xlsx", index=True)
            print(f"✅ DataFrame saved as {filename}.xlsx")
        else:
            raise ValueError("Unsupported file type. Use 'csv' or 'excel'.")
    except Exception as e:
        print(f"❌ Error saving DataFrame: {e}")
    
def main():
    tickers = ['TSLA', 'BND', 'SPY']
    start_date = '2015-07-01'
    end_date = '2025-07-31'

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        raw_data = fetch_data(ticker, start_date, end_date)
        if raw_data.empty:
            continue

        cleaned_data = clean_data(raw_data)
        enriched_data = add_features(ticker, cleaned_data)
        save_dataframe(cleaned_data, ticker, 'csv')
        adf_results = test_stationarity(enriched_data['Close'])
        print(f"ADF Test Results for {ticker}:")
        for key, value in adf_results.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()