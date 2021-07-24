
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

crypto_currency = 'BTC'
pair_currency = 'USD'

start = dt.datetime(2016,1,1)
end = dt.datetime.now()

data = web.DataReader(f'{crypto_currency}-{pair_currency}', 'yahoo', start, end)

#preparing data for ml model, downscaling the model
#print(data.head)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 60

x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append((scaled_data[x-prediction_days:x, 0]))
    y_train.append((scaled_data[x, 0]))

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))


#building neural network, working with long short term memory

model = Sequential()

model.add(LSTM(unit=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
#preventing overfitting
model.add(LSTM(unit=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(unit=50))
model.add(Dropout(0.2))
model.Dense(units=1)

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)












