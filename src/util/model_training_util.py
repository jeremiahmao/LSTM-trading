from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import random

def create_lstm_model(input_shape, lstm_units, dropout_rate, num_layers):
    model = Sequential()
    
    for i in range(num_layers):
        if num_layers == 1:
            # Only one LSTM layer, no need for return_sequences
            model.add(LSTM(units=lstm_units, input_shape=input_shape))
        elif i == 0:
            # First LSTM layer, return_sequences=True
            model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape))
        elif i == num_layers - 1:
            # Last LSTM layer, no return_sequences
            model.add(LSTM(units=lstm_units))
        else:
            # Intermediate LSTM layers, return_sequences=True
            model.add(LSTM(units=lstm_units, return_sequences=True))
        
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1))  # Output layer for predicting stock price
    model.compile(optimizer='adam', loss='mse')
    
    return model

def detour_foraging(rabbit, search_space, T):
    d = len(search_space)  # Dimension of the problem
    
    x_i = np.array(list(rabbit.values()))
    
    # Calculate the running length L
    e_t = np.linalg.norm(x_i - np.mean(x_i))  # Euclidean distance of current position from mean
    L = (np.exp(-e_t / T)**2) * np.sin(2 * np.pi * np.random.rand())
    
    # Calculate R
    R = L * np.random.randint(0, 2, size=d)
    
    # Mapping vector c
    c = np.zeros(d)
    g = np.random.permutation(d)
    l = np.ceil(np.random.rand() * d).astype(int)
    c[g[:l]] = 1
    
    # Perturbation term n1
    n1 = np.random.normal(0, 1, d)
    perturbation = R * (np.mean(x_i) - x_i) + 0.5 * (0.05 + np.random.rand()) * n1
    
    # Update position v_i
    v_i = x_i + perturbation
    
    # Clip the new position to the search space
    v_i = np.clip(v_i, [v[0] for v in search_space.values()], [v[1] for v in search_space.values()])
    
    return dict(zip(search_space.keys(), v_i))

def random_hiding(rabbit, search_space, T, t, d):
    x_i = np.array(list(rabbit.values()))
    
    # Generate the random number r4
    r4 = np.random.rand()
    
    # Calculate the hiding parameter H
    H = (T - t + 1) / T * r4
    
    burrows = []
    for j in range(d):
        # Mapping vector g
        g = np.zeros(d)
        g[j] = 1
        
        # Perturbation term n2
        n2 = np.random.normal(0, 1, len(search_space))
        
        # Calculate burrow position
        burrow = x_i + H * g * x_i + 0.5 * n2
        burrow = np.clip(burrow, [v[0] for v in search_space.values()], [v[1] for v in search_space.values()])
        
        burrows.append(dict(zip(search_space.keys(), burrow)))
    
    # Select a random burrow
    selected_burrow = random.choice(burrows)
    
    # Generate the random number r5
    r5 = np.random.rand()
    
    # Generate the mapping vector gr
    gr = np.zeros(d)
    j = int(np.ceil(r5 * d)) - 1  # Index for the selected burrow
    gr[j] = 1
    
    # Calculate the random factor R
    R = np.random.rand() * r4
    
    # Update position v_i using the selected burrow
    b_i_r = np.array(list(selected_burrow.values())) + H * gr
    v_i = x_i + R * (b_i_r - x_i)
    v_i = np.clip(v_i, [v[0] for v in search_space.values()], [v[1] for v in search_space.values()])
    
    return dict(zip(search_space.keys(), v_i))

def artificial_rabbits_optimization(objective_function, search_space, num_rabbits=10, iterations=20, d=5, detour_threshold=1):
    # Initialize rabbits' positions
    rabbits = np.random.uniform(
        low=[v[0] for v in search_space.values()],
        high=[v[1] for v in search_space.values()],
        size=(num_rabbits, len(search_space))
    )
    rabbits = [dict(zip(search_space.keys(), r)) for r in rabbits]
    
    best_rabbit = None
    best_fitness = float('inf')
    
    # Cache for storing previously evaluated positions
    fitness_cache = {}
    
    for t in range(iterations):
        for i in range(num_rabbits):
            x_i = rabbits[i]
            
            # Calculate energy factor A
            A = 4 * (1 - t / iterations) * np.log(1 / num_rabbits)
            
            # Apply detour foraging or random hiding based on energy factor
            if A > detour_threshold:
                v_i = detour_foraging(x_i, search_space, iterations, t)
            else:
                v_i = random_hiding(x_i, search_space, iterations, t, d)
            
            # Update positions based on fitness
            fitness_current = fitness_cache[tuple(x_i.values())] if tuple(x_i.values()) in fitness_cache else objective_function(list(x_i.values()))
            fitness_cache[tuple(x_i.values())] = fitness_current
            
            fitness_candidate = fitness_cache[tuple(v_i.values())] if tuple(v_i.values()) in fitness_cache else objective_function(list(v_i.values()))
            fitness_cache[tuple(v_i.values())] = fitness_candidate
            
            if fitness_candidate < fitness_current:
                rabbits[i] = v_i
        
        # Update the best rabbit and best fitness after all rabbits have moved
        fitnesses = [fitness_cache[tuple(rabbit.values())] for rabbit in rabbits]
        min_fitness = min(fitnesses)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_rabbit = rabbits[fitnesses.index(min_fitness)]
    
    return best_rabbit, best_fitness