from matplotlib import pyplot as plt
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import os


from credentials import ALPHAVANTAGE_API_KEY

def fetch_stock_data(symbol: str, interval: str = 'daily', outputsize: str = 'compact') -> pd.DataFrame:
    """
    Fetches historical stock data from Alpha Vantage and returns it as a pandas DataFrame.

    Parameters:
    - symbol (str): The stock symbol (e.g., "AAPL" for Apple).
    - api_key (str): Your Alpha Vantage API key.
    - interval (str): Interval between data points ('daily', 'weekly', 'monthly').
    - outputsize (str): Size of the output ('compact' or 'full').

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the stock data.
    """

    ts = TimeSeries(key=ALPHAVANTAGE_API_KEY, output_format='pandas')
    
    if interval == 'daily':
        data, _ = ts.get_daily(symbol=symbol, outputsize=outputsize)
    elif interval == 'weekly':
        data, _ = ts.get_weekly(symbol=symbol)
    elif interval == 'monthly':
        data, _ = ts.get_monthly(symbol=symbol)
    else:
        raise ValueError("Invalid interval. Choose from 'daily', 'weekly', or 'monthly'.")

    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)
    
    return data


def save_data(df: pd.DataFrame, local_path: str) -> None:
    """
    Saves a pandas DataFrame to a CSV file.

    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - local_path (str): The path where the CSV file will be saved.
    """
    df.to_csv(local_path)

def load_csv(local_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.

    Parameters:
    - local_path (str): The local path of the CSV file.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the CSV data.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"No such file: '{local_path}'")

    df = pd.read_csv(local_path, parse_dates=['date'], index_col='date')
    
    return df



# Example usage
if __name__ == "__main__":
    
    
    symbol = "AAPL"
    interval = 'daily'

    # Fetch the data
    df = fetch_stock_data(symbol, interval)

    # Save to CSV

    relative_path = '../../data/raw/AAPL.csv'
    dirname = os.path.dirname(__file__)
    local_path = os.path.join(dirname, relative_path)
    
    save_data(df, local_path)

    # Load from CSV and print the first few rows
    df_loaded = load_csv(local_path)
    print(df_loaded.head())

    df2 = df_loaded.reset_index()['4. close']
    plt.plot(df2)
    plt.show()