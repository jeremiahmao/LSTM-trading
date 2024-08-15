import pandas as pd
import requests
from datetime import datetime
from constants.credentials import ALPHAVANTAGE_API_KEY

import time # for fetching historical news data

import os
from matplotlib import pyplot as plt
from constants.constants import TRAINING_SYMBOLS, TEST_SYMBOLS

def fetch_data_with_retry(url, params=None, retries=5, backoff_factor=1):
    retried = False
    for i in range(retries):
        try:
            # Set connect and read timeouts separately to ensure the timeout works correctly
            response = requests.get(url, params=params, allow_redirects=True, timeout=(10, 30))
            response.raise_for_status()  # Check for HTTP errors
            if retried: print('Retry successful!')
            return response.json()  # Return the JSON response if successful
        except Exception as err:  # Catch any type of exception
            print(f"Error occurred: {err}")
            if i < retries - 1:  # Only retry if we haven't exhausted our retries
                wait_time = backoff_factor * (2 ** i)  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                retried = True
                time.sleep(wait_time)
            else:
                print("All retry attempts failed.")
                raise Exception(f"Failed to fetch data after {retries} attempts. Last error: {err}")

def test_tickers_sentiments(tickers: list):
    """
    Test a list of ticker symbols to check for invalid inputs and API responses.

    Parameters:
        tickers (list): A list of stock ticker symbols to test.
        interval (str): The interval between data points in the time series.
        outputsize (str): The amount of data to return.

    Returns:
        dict: A dictionary with ticker symbols as keys and error messages as values, if any.
    """
    base_url = 'https://www.alphavantage.co/query'
    invalid_tickers = set()

    for ticker in tickers:
        print(f"testing {ticker}")
        # Parameters for the API request
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'apikey': ALPHAVANTAGE_API_KEY
        }
        
        data = fetch_data_with_retry(base_url, params=params)
        
        # Check if the "Information" field starts with "Invalid inputs"
        if data.get("Information", "").startswith("Invalid inputs"):
            invalid_tickers.add(ticker)
    
    return invalid_tickers

def fetch_candles_adjusted_data(symbol: str, interval: str = 'daily', outputsize: str = 'compact') -> pd.DataFrame:
    """
    Fetches stock data from Alpha Vantage API and returns it as a pandas DataFrame.

    Parameters:
        symbol (str): The stock ticker symbol (e.g., 'IBM').
        interval (str): The interval between data points in the time series. Options include 'daily', 'weekly', 'monthly'.
        outputsize (str): The amount of data to return. Options are 'compact' (100 data points) or 'full' (all available data).

    Returns:
        pd.DataFrame: A DataFrame containing the stock data.
    """
    # API parameters
    api_key = ALPHAVANTAGE_API_KEY  # Replace with your Alpha Vantage API key
    base_url = 'https://www.alphavantage.co/query'
    
    # Determine the correct function based on interval
    if interval == 'daily':
        function = 'TIME_SERIES_DAILY_ADJUSTED'
    elif interval == 'weekly':
        function = 'TIME_SERIES_WEEKLY_ADJUSTED'
    elif interval == 'monthly':
        function = 'TIME_SERIES_MONTHLY_ADJUSTED'
    else:
        raise ValueError(f"Invalid interval: {interval}. Choose from 'daily', 'weekly', or 'monthly'.")
    
    # Parameters for the API request
    params = {
        'function': function,
        'symbol': symbol,
        'outputsize': outputsize,
        'apikey': api_key
    }
    
    data = fetch_data_with_retry(base_url, params=params)
    
    # Extract the relevant time series data
    time_series_key = list(data.keys())[1]  # The second key contains the time series data
    time_series_data = data[time_series_key]
    
    # Convert the time series data to a pandas DataFrame
    df = pd.DataFrame.from_dict(time_series_data, orient='index', dtype=float)
    
    # Rename the columns correctly, skipping the date index
    df.columns = [col.split('. ')[1] if '. ' in col else col for col in df.columns]
    
    # Convert the index to a datetime object
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'  # Optionally name the index as 'date'
    
    # Sort the DataFrame by date
    df = df.sort_index()
    
    return df

def fetch_news_sentiment_data(req_count: int, tickers: str, start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
    #retrieves news article sentiments, generates data for each date containing the number of articles published as well as the weighted sentiment of articles for that date
    
    # Default end_date to today's date if not provided
    if end_date is None:
        end_date = datetime.now()
    
    base_url = 'https://www.alphavantage.co/query'
    df_list = []
    
    # Initialize the current start date for the API requests
    current_start_date = start_date

    # Rate limit parameters
    max_requests_per_minute = 75
    request_count = req_count
    start_time = time.time()
    
    while current_start_date <= end_date:
        # Define the end date for the current request
        current_end_date = min(current_start_date + pd.DateOffset(days=29), end_date)
        
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': tickers,
            'time_from': current_start_date.strftime('%Y%m%dT%H%M'),
            'time_to': current_end_date.strftime('%Y%m%dT%H%M'),
            'limit': 1000,  # Limit the number of articles to 1,000 per request
            'apikey': ALPHAVANTAGE_API_KEY
        }
        
        # Print request information
        print(f"Requesting data from {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')}")
        
        data = fetch_data_with_retry(base_url, params=params)
        
        # Check if the "Information" field starts with "Invalid inputs"
        if data.get("Information", "").startswith("Invalid inputs"):
            raise Exception(f"Invalid inputs detected for {tickers}. Response: {data.get('Information')}")
        
        news_data = data.get("feed", [])

        # Print number of articles returned
        print(f"Number of articles returned: {len(news_data)}")
        
        # Extract relevant information from the feed
        for article in news_data:
            time_published = pd.to_datetime(article['time_published'])
            date = time_published.date()

            # Iterate through ticker_sentiment to filter by provided tickers
            for ticker_sent in article.get('ticker_sentiment', []):
                ticker = ticker_sent.get('ticker')
                if ticker in tickers:
                    sentiment_score = float(ticker_sent.get('ticker_sentiment_score', 0))
                    relevance_score = float(ticker_sent.get('relevance_score', 0))
                    
                    # Calculate relevance as the product of sentiment_score and relevance_score
                    weighted_sentiment = sentiment_score * relevance_score
                    
                    # Create a DataFrame for each article and append it to the list
                    df = pd.DataFrame([{
                        'date': date,
                        'ticker': ticker,
                        'sentiment': weighted_sentiment
                    }])
                    df_list.append(df)
        
        # Update request count
        request_count += 1
        print(f"Current request count: {request_count}")
        
        # Check if rate limit has been reached
        if request_count >= max_requests_per_minute:
            elapsed_time = time.time() - start_time
            if elapsed_time < 60:
                # Wait until a minute has passed since the first request in this batch
                wait_time = 60 - elapsed_time
                print(f"Rate limit hit. Sleeping for {wait_time:.2f} seconds.")
                time.sleep(wait_time)

            # Reset counters
            request_count = 0
            start_time = time.time()

        # Move the start date forward by 30 days
        current_start_date = current_start_date + pd.DateOffset(days=30)
    
    if df_list:
        full_df = pd.concat(df_list)
        
        # Perform a single groupby operation to calculate both sentiment sum and article count
        relevance_df = full_df.groupby('date').agg(
            sentiment_sum=('sentiment', 'sum'),
            article_count=('sentiment', 'size')
        ).reset_index()
        
        # Convert the 'date' column to datetime and set it as the index
        relevance_df['date'] = pd.to_datetime(relevance_df['date'])
        relevance_df.set_index('date', inplace=True)  # Set the index to the date
        
        # Rename the sentiment_sum column back to sentiment
        relevance_df.rename(columns={'sentiment_sum': 'sentiment'}, inplace=True)
        relevance_df.rename(columns={'article_count': 'article count'}, inplace=True)
        
        return request_count, relevance_df
    else:
        raise Exception("No news sentiment data found.")
    
def merge_candles_with_sentiments(candles_df, sentiments_df):
    """
    Merge the candles dataset with the sentiments dataset on the date column.
    
    Parameters:
    candles_df (pd.DataFrame): DataFrame containing the candles data.
    sentiments_df (pd.DataFrame): DataFrame containing the sentiment data.

    Returns:
    pd.DataFrame: Merged DataFrame with an additional sentiment column.
    """
    # Merge on date using a left join
    merged_df = pd.merge(candles_df, sentiments_df, on='date', how='left')
    
    return merged_df

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


#generating data for this project


if __name__ == "__main__":

    dirname = os.path.dirname(__file__)

    start_date = datetime(2013, 6, 1)
    end_date = datetime(2023, 6, 1)

    request_count = 0

    # print(test_tickers_sentiments(TRAINING_SYMBOLS))
    # import sys
    # sys.exit()

    #Code to test that all tickers are valid in Alphavantage:  print(test_tickers(TRAINING_SYMBOLS))

    for s in TRAINING_SYMBOLS:

        # start_time = time.time()

        # # Perform your loop body tasks here
        # print(f"Loop iteration {s} started")

        print(f"\nStarting data generation for {s}.")

        # Candles Data Saved
        candles_relative_path = '../data/raw/' + s + '_c.csv'
        candles_local_path = os.path.join(dirname, candles_relative_path)

        if os.path.isfile(candles_local_path):
            print(f"File exists: {s} candles data")
        else:
            request_count +=1
            interval = 'daily'
            candles = fetch_candles_adjusted_data(s, interval, outputsize="full")
            filtered_candles = candles.loc[start_date:end_date]

            print(f"Saving file: {s} candles data")
            save_data(filtered_candles, candles_local_path)
        
        # News Sentiments Data Saved
        sentiments_relative_path = '../data/raw/' + s + '_s.csv'
        sentiments_local_path = os.path.join(dirname, sentiments_relative_path)

        if os.path.isfile(sentiments_local_path):
            print(f"File exists: {s} news sentiments data")
        else:
            rc, sentiments = fetch_news_sentiment_data(request_count, s, start_date=start_date, end_date=end_date)
            request_count = rc
            print(f"Saving file: {s} news sentiments data")
            save_data(sentiments, sentiments_local_path)

        # Merged Dataset Saved
        merged_relative_path = '../data/interim/' + s + '_merged.csv'
        merged_local_path = os.path.join(dirname, merged_relative_path)

        if os.path.isfile(merged_local_path):
            print(f"File exists: {s} merged data")
        else:
            # Reload candles and sentiments
            candles = load_csv(candles_local_path)
            sentiments = load_csv(sentiments_local_path)

            merged = merge_candles_with_sentiments(candles, sentiments)

            print(f"Saving file: {s} merged data")
            save_data(merged, merged_local_path)

            print(f"Finished data generation for {s}.")

            # # Calculate how long the loop body took
            # elapsed_time = time.time() - start_time

            # # Determine how much time to wait to round up to the nearest minute
            # extra = elapsed_time % 60
            # if extra != 0:
            #     # Round up to the nearest minute
            #     time_to_wait = 60 - extra
            #     print(f'{s} iteration waiting to nearest minute')
            #     # Sleep for the calculated time
            #     time.sleep(time_to_wait)

            # print(f"Loop iteration {s} ended after {time.time() - start_time} seconds")


    # #Load from CSV and print the first few rows
    # df_loaded = load_csv(local_path)
    # print(df_loaded.head())

    # df2 = df_loaded.reset_index()['close']
    # plt.plot(df2)
    # plt.show()