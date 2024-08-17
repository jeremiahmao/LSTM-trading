import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from util.model_training_util import create_lstm_model, artificial_rabbits_optimization
from build_features import load_combined_sequences_memmap
from constants.constants import CANDLES_TIME_STEP, CANDLES_NUM_FEATURES
import os
import time

# Set TensorFlow to use GPU 0
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print("FAILED")
        print(e)

# Stock dataset
dirname = os.path.dirname(__file__)
source_dir = os.path.join(dirname, '../data/processed/')

# Load the memory-mapped sequences
X, y = load_combined_sequences_memmap(source_dir)

# Convert the memory-mapped arrays to TensorFlow datasets for efficient batching
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Define a TensorFlow dataset for efficient data loading
def create_tf_dataset(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# Define the objective function
def objective_function(params):
    print("Running objective_function")
    start_time = time.time()
    lstm_units, dropout_rate, num_layers, batch_size, epochs = params

    model = create_lstm_model(
        input_shape=(CANDLES_TIME_STEP, CANDLES_NUM_FEATURES),
        lstm_units=int(lstm_units),
        dropout_rate=dropout_rate,
        num_layers=int(num_layers)
    )

    # Create TensorFlow datasets
    train_dataset = create_tf_dataset(X_train, y_train, int(batch_size))
    test_dataset = create_tf_dataset(X_test, y_test, int(batch_size))

    # Train the model
    model.fit(train_dataset, epochs=int(epochs), verbose=1)
    val_loss = model.evaluate(test_dataset, verbose=0)

    end_time = time.time()
    print(f"Execution time (objective_function): {end_time - start_time} seconds")

    return val_loss  # Minimize this value

# Define the search space for ARO
search_space = {
    'lstm_units': (40, 80),        # Adjust the range as needed
    'dropout_rate': (0.1, 0.5),     # Dropout rate range
    'num_layers': (1, 3),           # Number of LSTM layers
    'batch_size': (16, 64),         # Batch size range
    'epochs': (10, 50)              # Number of epochs
}

start_time = time.time()
# Run ARO to optimize the LSTM Network
best_hyperparameters, best_fitness = artificial_rabbits_optimization(objective_function, search_space)
end_time = time.time()
print(f"Execution time (artificial_rabbits_optimization): {end_time - start_time} seconds")

print("Best Hyperparameters:", best_hyperparameters)
print("Best Validation Loss:", best_fitness)

# Train the final LSTM model with optimized hyperparameters
final_model = create_lstm_model(
    input_shape=(CANDLES_TIME_STEP, CANDLES_NUM_FEATURES),
    lstm_units=int(best_hyperparameters['lstm_units']),
    dropout_rate=best_hyperparameters['dropout_rate'],
    num_layers=int(best_hyperparameters['num_layers'])
)

# Create TensorFlow datasets for the final training
train_dataset = create_tf_dataset(X_train, y_train, int(best_hyperparameters['batch_size']))
test_dataset = create_tf_dataset(X_test, y_test, int(best_hyperparameters['batch_size']))

final_model.fit(train_dataset, epochs=int(best_hyperparameters['epochs']))

# Evaluate the final model
test_loss = final_model.evaluate(test_dataset)
print("Test Loss:", test_loss)

# Save the trained model

# Define the directory and filename
directory = 'saved_models'
filename = 'candles_lstm_model.h5'
path = os.path.join(directory, filename)

# Create the directory if it does not exist
os.makedirs(directory, exist_ok=True)

# Save the model
final_model.save(path)
print(f"Model saved to '{path}'")