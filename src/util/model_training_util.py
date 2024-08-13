from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import random

def create_lstm_model(input_shape, lstm_units, dropout_rate, num_layers):
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape))
        elif i == num_layers - 1:
            model.add(LSTM(units=lstm_units))
        else:
            model.add(LSTM(units=lstm_units, return_sequences=True))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1))  # Output layer for predicting stock price
    model.compile(optimizer='adam', loss='mse')
    return model

def detour_foraging(rabbits, search_space, T, t):
    updated_rabbits = []
    d = len(search_space)  # Dimension of the problem
    
    for i in range(len(rabbits)):
        x_i = np.array(list(rabbits[i].values()))
        
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
        updated_rabbits.append(dict(zip(search_space.keys(), v_i)))
    
    return updated_rabbits

def random_hiding(rabbits, search_space, T, t, d):
    updated_rabbits = []
    
    for i in range(len(rabbits)):
        x_i = np.array(list(rabbits[i].values()))
        
        # Calculate the hiding parameter H
        H = (T - t + 1) / T * np.random.rand()
        
        # Generate burrows
        burrows = []
        for _ in range(d):
            # Random perturbation in each dimension
            burrow = x_i + H * (np.random.rand(len(search_space)) - 0.5) * 2
            burrows.append(dict(zip(search_space.keys(), np.clip(burrow, [v[0] for v in search_space.values()], [v[1] for v in search_space.values()]))))
        
        # Select a random burrow
        selected_burrow = random.choice(burrows)
        
        # Generate the random number r4 and r5
        r4 = np.random.rand()
        r5 = np.random.rand()
        
        # Generate the mapping vector gr
        gr = np.zeros(d)
        j = int(np.ceil(r5 * d)) - 1  # Index for the selected burrow
        gr[j] = 1
        
        # Calculate the random factor R
        R = np.random.rand() * r4
        
        # Update position v_i
        b_i_r = x_i + H * gr
        v_i = x_i + R * (b_i_r - x_i)
        v_i = np.clip(v_i, [v[0] for v in search_space.values()], [v[1] for v in search_space.values()])
        
        updated_rabbits.append(dict(zip(search_space.keys(), v_i)))
    
    return updated_rabbits

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
    
    for t in range(iterations):
        fitnesses = [objective_function(list(rabbit.values())) for rabbit in rabbits]
        min_fitness = min(fitnesses)
        best_rabbit = rabbits[fitnesses.index(min_fitness)]
        best_fitness = min_fitness
        
        # Calculate energy factor A
        A = 4 * (1 - t / iterations) * np.log(1 / num_rabbits)
        
        # Apply detour foraging or random hiding based on energy factor
        if A > detour_threshold:
            rabbits = detour_foraging(rabbits, search_space, iterations, t)
        else:
            rabbits = random_hiding(rabbits, search_space, iterations, t, d)
        
        # Update positions based on fitness
        for i in range(num_rabbits):
            x_i = np.array(list(rabbits[i].values()))
            fitness_current = objective_function(list(x_i))
            
            # Choose the best between current and new position
            v_i = np.array(list(rabbits[i].values()))
            fitness_candidate = objective_function(list(v_i))
            if fitness_candidate < fitness_current:
                rabbits[i] = dict(zip(search_space.keys(), v_i))
    
    return best_rabbit, best_fitness