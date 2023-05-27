import yfinance as yf
import math
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler


class StockData:
    def __init__(self, ticker):
        self.ticker = ticker
        today = datetime.now()
        five_years_ago = today - timedelta(days=5 * 365)
        self.start_date = five_years_ago.strftime('%Y-%m-%d')
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        stock_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        stock_data.head()
        close_prices = stock_data['Close']
        self.raw_data = close_prices.values
        values = self.raw_data
        self.training_data_len = math.ceil(len(values) * 0.9)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(values.reshape(-1, 1))
        self.train_data = scaled_data[0: self.training_data_len, :]
        self.test_data = scaled_data[self.training_data_len - 21:, :]

    def get_raw_data(self):
        return self.raw_data

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_training_data_len(self):
        return self.training_data_len
