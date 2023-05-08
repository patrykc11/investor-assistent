from load_data import StockData
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class ModelTrainer:
    def __init__(self, ticker):
        self.ticker = ticker
        stock_data = StockData(ticker)
        self.x_train = []
        self.y_train = []
        self.x_test = []
        values = stock_data.get_raw_data()
        self.raw_values = values
        self.y_test = values[stock_data.get_training_data_len():]
        train_data = stock_data.get_train_data()
        test_data = stock_data.get_test_data()

        for i in range(60, len(train_data)):
            self.x_train.append(train_data[i - 60:i, 0])
            self.y_train.append(train_data[i, 0])

        self.x_train , self.y_train = np.array(self.x_train), np.array(self.y_train)
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))

        for i in range(60, len(test_data)):
            self.x_test.append(test_data[i - 60:i, 0])

        self.x_test = np.array(self.x_test)
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))

    def get_x_test(self):
        return self.x_test

    def get_raw_values(self):
        return self.raw_values

    def get_y_test(self):
        return self.y_test
    def get_model(self):
        tf.keras.backend.clear_session()
        if os.path.exists('models/' + self.ticker + '_model.h5'):
            return load_model('models/' + self.ticker + '_model.h5')
        else:
            self.train_model()
            return load_model('models/' + self.ticker + '_model.h5')

    def train_model(self):
        model = keras.Sequential()
        model.add(layers.LSTM(100, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        model.add(layers.LSTM(100, return_sequences=False))
        model.add(layers.Dense(25))
        model.add(layers.Dense(1))
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(self.x_train, self.y_train, batch_size=1, epochs=3)
        model.save('models/' + self.ticker + '_model.h5')
        tf.keras.backend.clear_session()
