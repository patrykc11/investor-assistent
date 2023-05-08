from model_trainer import ModelTrainer
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Predicator:
    def __init__(self, ticker):
        model_trainer = ModelTrainer(ticker)
        self.raw_values = model_trainer.get_raw_values()
        self.x_test = model_trainer.get_x_test()
        self.y_test = model_trainer.get_y_test()
        self.model = model_trainer.get_model()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def get_rmse(self):
        predictions = self.model.predict(self.x_test)
        predictions = self.scaler.inverse_transform(predictions)
        self.rmse = np.sqrt(np.mean(predictions - self.y_test) ** 2)
        return self.rmse

    def predict_next_day(self):
        last_60_days = self.raw_values[-60:]
        last_60_days_scaled = self.scaler.fit_transform(last_60_days.reshape(-1, 1))
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)
        predicted_price = self.model.predict(X_test)
        predicted_price = self.scaler.inverse_transform(predicted_price)
        return predicted_price[0][0]

    def predict_next_day_test(self):
        errors = []
        j = 0
        for i in range(1000):
            start_index = np.random.randint(0, len(self.raw_values) - 61)
            indexes = np.arange(start_index, start_index + 60)
            last_60_days = self.raw_values[indexes]
            print('rzeczywista: ' + str(self.raw_values[start_index+61]))
            last_60_days_scaled = self.scaler.fit_transform(last_60_days.reshape(-1, 1))
            X_test = []
            X_test.append(last_60_days_scaled)
            X_test = np.array(X_test)
            predicted_price = self.model.predict(X_test)
            predicted_price = self.scaler.inverse_transform(predicted_price)
            print('przewidywana: ' + str(predicted_price[0][0]))
            diff_between_days = self.raw_values[start_index+60] - predicted_price[0][0]
            diff = self.raw_values[start_index+61] - predicted_price[0][0]
            if( diff > 0 and diff_between_days < 0):
                j = j + 1
            errors.append(diff)
        print(errors)
        print('rzeczywista cena była wyższa od przewidywanej mając na uwadze to, że cena z dnia poprzedniego była niższa: ' + str(j) + ' razy')
