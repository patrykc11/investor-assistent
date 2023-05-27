from load_data import StockData
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ReduceLROnPlateau
class ModelTrainer:
    def __init__(self, ticker):
        tf.keras.backend.clear_session()
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

        for i in range(21, len(train_data)):
            self.x_train.append(train_data[i - 21:i, 0])
            self.y_train.append(train_data[i, 0])

        self.x_train , self.y_train = np.array(self.x_train), np.array(self.y_train)
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))

        for i in range(21, len(test_data)):
            self.x_test.append(test_data[i - 21:i, 0])

        self.x_test = np.array(self.x_test)
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))


    def get_x_test(self):
        return self.x_test

    def get_raw_values(self):
        return self.raw_values

    def get_y_test(self):
        return self.y_test

    def train_model(self):
        # rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100)
        print('trening rozpoczęty')
        model = keras.Sequential()
        print(self.x_train.shape)
        model.add(layers.LSTM(64, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        print('1')
        model.add(layers.Dropout(0.3))
        print('2')
        model.add(layers.LSTM(64, return_sequences=False, input_shape=(self.x_train.shape[1], 1)))
        model.add(layers.Dropout(0.3))
        print('3')
        model.add(layers.Dense(32,kernel_initializer="uniform",activation='relu'))  
        print('4')
        model.add(layers.Dense(1,kernel_initializer="uniform",activation='linear'))
        model.compile(loss='mae',optimizer= 'adam')
        print('5')
        model.fit(self.x_train, self.y_train, batch_size=32, epochs=13, validation_split=0.20, validation_data=(self.x_test, self.y_test), verbose=1)
        print('6')
        model.save('./models/' + self.ticker + '_model.h5')
        # score=(sum(abs(actual_stock[ticker_list[k]]-pred_stock[ticker_list[k]])/actual_stock[ticker_list[k]])/len(actual_stock[ticker_list[k]]))*100
        tf.keras.backend.clear_session()

    def get_model(self):
        tf.keras.backend.clear_session()
        print(os.path.isfile('./models/' + self.ticker + '_model.h5'))
        if os.path.isfile('./models/' + self.ticker + '_model.h5'):
            return load_model('./models/' + self.ticker + '_model.h5')
        else:
            self.train_model()
            return load_model('./models/' + self.ticker + '_model.h5')

