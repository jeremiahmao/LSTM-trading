from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

#for testing
import os
from make_dataset import load_csv
from constants.constants import TRAINING_SYMBOLS
from sklearn.model_selection import train_test_split

def create_preprocessed_merged_sequences(df: pd.DataFrame, time_step: int=100, target_column: str='adjusted close'):
    """
    Create sequences of stock data for LSTM training, with scaling applied within each time_step window,
    excluding the specified column from scaling and replacing NaN values in the excluded column with 0.

    Parameters:
    df (pd.DataFrame): DataFrame containing stock data with candles in chronological order. The 'date' should be the index.
    time_step (int): Number of days to include in each feature sequence.
    target_column (str): Column name for the target label (closing price the next day).

    Returns:
    np.array: X (features) array of shape (num_samples, time_step, num_features).
    np.array: y (labels) array of shape (num_samples,).
    """
    X, y = [], []
    
    # Replace NaN values in the sentiment column with 0
    df['sentiment'] = df['sentiment'].fillna(0)
    
    # Extract feature columns excluding the sentiment column
    feature_columns = [col for col in df.columns if col != 'sentiment']
    
    # Iterate through the DataFrame to create sequences
    for i in range(len(df) - time_step):
        # Extract the sequence of data for the current time_step
        sequence = df.iloc[i:i + time_step].copy()
        
        # Separate the features to scale and the column to exclude from scaling
        features = sequence[feature_columns]
        
        # Initialize the scaler and scale the features
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Combine scaled features with the excluded column
        sequence_scaled = sequence.copy()
        sequence_scaled[feature_columns] = features_scaled
        
        # Append the scaled sequence and corresponding label
        X.append(sequence_scaled.values)
        y.append(df.iloc[i + time_step][target_column])
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    for s in TRAINING_SYMBOLS:
        dirname = os.path.dirname(__file__)
        merged_relative_path = '../data/interim/' + s + '_merged.csv'
        merged_local_path = os.path.join(dirname, merged_relative_path)

        #check file
        if not os.path.isfile(merged_local_path):
            print(f"File missing: {s} merged data")
            continue

        # Load from CSV and print the first few rows
        df = load_csv(merged_local_path)
        print(df.head())

        X, y = create_preprocessed_merged_sequences(df=df,time_step=5) # (target_column='close')

        print(X[19])
        print(y[19])

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)


