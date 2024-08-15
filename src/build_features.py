from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

#for testing
import os
from make_dataset import load_csv
from constants.constants import TRAINING_SYMBOLS, TEST_SYMBOLS, TIME_STEP

def create_preprocessed_merged_sequences(df: pd.DataFrame, time_step: int=120, target_column: str='adjusted close'):
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
    
    # Replace NaN values in the sentiment column with 0 (Note: article count will still be scaled so that it doesn't overly affect results)
    df[['sentiment', 'article count']] = df[['sentiment', 'article count']].fillna(0)

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

def save_sequences_to_files(symbols, merged_relative_dir = '../data/interim/', save_relative_dir = '../data/processed_individual/', time_step=120, target_column='adjusted close'):
    """
    Save preprocessed sequences to individual files for each symbol.

    Parameters:
    symbols (list): List of stock symbols to process.
    save_dir (str): Directory path to save the individual sequence files.
    time_step (int): Number of days to include in each feature sequence.
    target_column (str): Column name for the target label (closing price the next day).
    """
    dirname = os.path.dirname(__file__)
    save_dir = os.path.join(dirname, save_relative_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for s in symbols:

        X_file = os.path.join(save_dir, f'{s}_X.npy')
        y_file = os.path.join(save_dir, f'{s}_y.npy')
        
        # Check if files already exist
        if os.path.isfile(X_file) and os.path.isfile(y_file):
            print(f"Files already exist for {s}, skipping individual sequences generation.")
            continue

        dirname = os.path.dirname(__file__)
        merged_relative_path = merged_relative_dir + s + '_merged.csv'
        merged_local_path = os.path.join(dirname, merged_relative_path)

        # Check if file exists
        if not os.path.isfile(merged_local_path):
            print(f"File missing: {s} merged data")
            continue

        # Load from CSV
        df = load_csv(merged_local_path)
        print(f"Processing {s}...")

        X, y = create_preprocessed_merged_sequences(df=df, time_step=time_step, target_column=target_column)

        # Save X and y to individual files
        np.save(os.path.join(save_dir, f'{s}_X.npy'), X)
        np.save(os.path.join(save_dir, f'{s}_y.npy'), y)

    print(f"Individual sequences saved to {save_dir}")

def merge_sequences_from_files(source_relative_dir = '../data/processed_individual/', save_relative_dir = '../data/processed/'):
    """
    Merge preprocessed sequences from individual files into a single dataset.

    Parameters:
    source_dir (str): Directory path to read individual sequence files.
    save_dir (str): Directory path to save the combined dataset.
    """
    dirname = os.path.dirname(__file__)
    source_dir = os.path.join(dirname, source_relative_dir)
    save_dir = os.path.join(dirname, save_relative_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    X_combined, y_combined = [], []

    for file in os.listdir(source_dir):
        if file.endswith('_X.npy'):
            symbol = file.split('_')[0]
            X = np.load(os.path.join(source_dir, file))
            y = np.load(os.path.join(source_dir, f'{symbol}_y.npy'))
            
            X_combined.append(X)
            y_combined.append(y)

    # Concatenate all the arrays
    X_combined = np.concatenate(X_combined, axis=0)
    y_combined = np.concatenate(y_combined, axis=0)

    # Save the combined data
    np.save(os.path.join(save_dir, 'X.npy'), X_combined)
    np.save(os.path.join(save_dir, 'y.npy'), y_combined)

    print(f"Combined data saved to {save_dir}")

## NOT RELATIVE PATH
def load_combined_sequences(target_dir):
    """
    Load the combined sequence files from the specified directory.

    Parameters:
    target_dir (str): Directory path to load the combined dataset.

    Returns:
    np.array: Combined X array of shape (num_samples, time_step, num_features).
    np.array: Combined y array of shape (num_samples,).
    """
    X = np.load(os.path.join(target_dir, 'X.npy'))
    y = np.load(os.path.join(target_dir, 'y.npy'))
    return X, y

if __name__ == "__main__":

    # Save individual sequences
    save_sequences_to_files(symbols=TEST_SYMBOLS, merged_relative_dir = '../data/interim/', save_relative_dir = '../data/processed_individual/', time_step=TIME_STEP, target_column='adjusted close')

    # Merge and save combined sequences
    merge_sequences_from_files(source_relative_dir = '../data/processed_individual/', save_relative_dir = '../data/processed/')
    
    # Load combined sequences

    dirname = os.path.dirname(__file__)
    source_dir = os.path.join(dirname, '../data/processed/')

    X_combined, y_combined = load_combined_sequences(source_dir)
    
    print(f"Loaded combined X: {X_combined.shape}")
    print(f"Loaded combined y: {y_combined.shape}")