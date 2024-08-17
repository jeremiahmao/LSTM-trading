from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from make_dataset import load_csv
from constants.constants import MERGED_TRAINING_SYMBOLS, CANDLES_TRAINING_SYMBOLS
from constants.constants import CANDLES_TIME_STEP, MERGED_TIME_STEP
from constants.constants import CANDLES_NUM_FEATURES, MERGED_NUM_FEATURES

def create_preprocessed_candles_sequences(df: pd.DataFrame, time_step: int=CANDLES_TIME_STEP, target_column: str='adjusted close'):
    """
    Create sequences of stock data for LSTM training, with scaling applied within each time_step window.

    Parameters:
    df (pd.DataFrame): DataFrame containing stock data with candles in chronological order. The 'date' should be the index.
    time_step (int): Number of days to include in each feature sequence.
    target_column (str): Column name for the target label (closing price the next day).

    Returns:
    np.array: X (features) array of shape (num_samples, time_step, num_features).
    np.array: y (labels) array of shape (num_samples,).
    """
    X, y = [], []

    # Iterate through the DataFrame to create sequences
    for i in range(len(df) - time_step):
        # Extract the sequence of data for the current time_step
        sequence = df.iloc[i:i + time_step].copy()
        
        # Initialize the scaler and scale the entire sequence
        scaler = MinMaxScaler()
        sequence_scaled = scaler.fit_transform(sequence)
        
        # Append the scaled sequence and corresponding label
        X.append(sequence_scaled)
        y.append(df.iloc[i + time_step][target_column])
    
    return np.array(X), np.array(y)

def save_candles_sequences_to_files(symbols, candles_relative_dir = '../data/raw/', save_relative_dir = '../data/interim/', time_step=CANDLES_TIME_STEP, target_column='adjusted close'):
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
            print(f"Candles files already exist for {s:6}, skipping individual sequences generation.")
            continue

        dirname = os.path.dirname(__file__)
        candles_relative_path = candles_relative_dir + s + '_c.csv'
        candles_local_path = os.path.join(dirname, candles_relative_path)

        # Check if file exists
        if not os.path.isfile(candles_local_path):
            print(f"File missing: {s} candles data")
            continue

        # Load from CSV
        df = load_csv(candles_local_path)
        print(f"Processing {s} candles...")

        X, y = create_preprocessed_merged_sequences(df=df, time_step=time_step, target_column=target_column)

        # Save X and y to individual files
        np.save(X_file, X)
        np.save(y_file, y)

    print(f"Individual candles sequences saved to {save_dir}")

def create_preprocessed_merged_sequences(df: pd.DataFrame, time_step: int=MERGED_TIME_STEP, target_column: str='adjusted close'):
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

def save_merged_sequences_to_files(symbols, merged_relative_dir = '../data/merged/', save_relative_dir = '../data/interim_merged/', time_step=MERGED_TIME_STEP, target_column='adjusted close'):
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

        X_file = os.path.join(save_dir, f'{s}_X_merged.npy')
        y_file = os.path.join(save_dir, f'{s}_y_merged.npy')
        
        # Check if files already exist
        if os.path.isfile(X_file) and os.path.isfile(y_file):
            print(f"Merged files already exist for {s:6}, skipping individual sequences generation.")
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
        print(f"Processing {s} merged...")

        X, y = create_preprocessed_merged_sequences(df=df, time_step=time_step, target_column=target_column)

        # Save X and y to individual files
        np.save(X_file, X)
        np.save(y_file, y)

    print(f"Individual merged sequences saved to {save_dir}")

def combine_sequences_to_memmap(source_relative_dir: str, save_relative_dir: str, candles: bool = True, dtype=np.float32):
    """
    Merge preprocessed sequences from individual files into a single memory-mapped dataset.

    Parameters:
    source_dir (str): Directory path to read individual sequence files.
    save_dir (str): Directory path to save the combined memory-mapped dataset.
    """
    dirname = os.path.dirname(__file__)
    source_dir = os.path.join(dirname, source_relative_dir)
    save_dir = os.path.join(dirname, save_relative_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Determine total size for pre-allocation
    total_samples = 0
    sample_shape = None
    
    if candles:
        for file in os.listdir(source_dir):
            if file.endswith('_X.npy'):
                X = np.load(os.path.join(source_dir, file))
                total_samples += X.shape[0]
                if sample_shape is None:
                    sample_shape = X.shape[1:]
    else:
        for file in os.listdir(source_dir):
            if file.endswith('_X_merged.npy'):
                X = np.load(os.path.join(source_dir, file))
                total_samples += X.shape[0]
                if sample_shape is None:
                    sample_shape = X.shape[1:]

    # Create memory-mapped files
    if candles:
        x_path = os.path.join(save_dir, 'X_memmap.npy')
        y_path = os.path.join(save_dir, 'y_memmap.npy')
    else:
        x_path = os.path.join(save_dir, 'X_merged_memmap.npy')
        y_path = os.path.join(save_dir, 'y_merged_memmap.npy')
    
    X_combined = np.memmap(x_path, dtype=dtype, mode='w+', shape=(total_samples, *sample_shape))
    y_combined = np.memmap(y_path, dtype=dtype, mode='w+', shape=(total_samples,))

    # Write data to memory-mapped files
    start_idx = 0
    if candles:
        for file in os.listdir(source_dir):
            if file.endswith('_X.npy'):
                symbol = file.split('_')[0]
                X = np.load(os.path.join(source_dir, file))
                y = np.load(os.path.join(source_dir, f'{symbol}_y.npy'))
                
                end_idx = start_idx + X.shape[0]
                X_combined[start_idx:end_idx] = X
                y_combined[start_idx:end_idx] = y
                start_idx = end_idx
    else:
        for file in os.listdir(source_dir):
            if file.endswith('_X_merged.npy'):
                symbol = file.split('_')[0]
                X = np.load(os.path.join(source_dir, file))
                y = np.load(os.path.join(source_dir, f'{symbol}_y_merged.npy'))
                
                end_idx = start_idx + X.shape[0]
                X_combined[start_idx:end_idx] = X
                y_combined[start_idx:end_idx] = y
                start_idx = end_idx

    # Flush to disk
    X_combined.flush()
    y_combined.flush()

    print(f"Combined memory-mapped data saved to {save_dir}")

## NOT RELATIVE PATH
def load_combined_sequences_memmap(target_dir, candles=True, dtype=np.float32):
    """
    Load the combined memory-mapped sequence files from the specified directory.

    Parameters:
    target_dir (str): Directory path to load the combined dataset.
    dtype (np.dtype): Data type of the saved memory-mapped files.

    Returns:
    np.memmap: Combined X array as a memory-mapped file.
    np.memmap: Combined y array as a memory-mapped file.
    """
    if candles:
        X_path = os.path.join(target_dir, 'X_memmap.npy')
        y_path = os.path.join(target_dir, 'y_memmap.npy')

        # Load the memory-mapped files (make sure to specify the correct shape)
        X = np.memmap(X_path, dtype=dtype, mode='r')
        y = np.memmap(y_path, dtype=dtype, mode='r')

        # Reshape if needed (for instance, if your data was flattened)
        num_samples = X.shape[0] // (CANDLES_NUM_FEATURES * CANDLES_TIME_STEP)

        if X.ndim == 1:
            X = X.reshape((num_samples, CANDLES_TIME_STEP, CANDLES_NUM_FEATURES))
        
        return X, y
    else:
        X_path = os.path.join(target_dir, 'X_merged_memmap.npy')
        y_path = os.path.join(target_dir, 'y_merged_memmap.npy')
        
        # Load the memory-mapped files (make sure to specify the correct shape)
        X = np.memmap(X_path, dtype=dtype, mode='r')
        y = np.memmap(y_path, dtype=dtype, mode='r')

        # Reshape if needed (for instance, if your data was flattened)
        num_samples = X.shape[0] // (MERGED_TIME_STEP * MERGED_NUM_FEATURES)

        if X.ndim == 1:
            X = X.reshape((num_samples, CANDLES_TIME_STEP, CANDLES_NUM_FEATURES))
        
        return X, y
    
    

if __name__ == "__main__":

    ##THIS IS FOR CANDLES ONLY RIGHT NOW

    # Save individual sequences
    save_candles_sequences_to_files(symbols=CANDLES_TRAINING_SYMBOLS, candles_relative_dir= '../data/raw/', 
                                    save_relative_dir = '../data/interim/', time_step=CANDLES_TIME_STEP, target_column='adjusted close')

    # Merge and save combined sequences
    combine_sequences_to_memmap(source_relative_dir = '../data/interim/', save_relative_dir = '../data/processed/', candles=True)
    
    # Load combined sequences
    dirname = os.path.dirname(__file__)
    source_dir = os.path.join(dirname, '../data/processed/')

    X_combined, y_combined = load_combined_sequences_memmap(source_dir, candles=True)
    
    print(f"Loaded combined X: {X_combined.shape}")
    print(f"Loaded combined y: {y_combined.shape}")