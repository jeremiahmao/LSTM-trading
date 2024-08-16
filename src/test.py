
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

U = 100
model = Sequential()
model.add(LSTM(units=U, return_sequences=True, input_shape=(120,10)))
model.add(Dropout(.1))
model.add(LSTM(units=U, return_sequences=True))
model.add(Dropout(.1))
model.add(LSTM(units=U))
    
model.add(Dense(1))  # Output layer for predicting stock price
model.compile(optimizer='adam', loss='mse')


total_params = model.count_params()
print(f"Total number of parameters: {total_params}")