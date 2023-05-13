from model_trainer import ModelTrainer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

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
        last_21_days = self.raw_values[-21:]
        last_21_days_scaled = self.scaler.fit_transform(last_21_days.reshape(-1, 1))
        prediction_base = []
        prediction_base.append(last_21_days_scaled)
        prediction_base = np.array(prediction_base)
        predicted_price = self.model.predict(prediction_base)
        predicted_price = self.scaler.inverse_transform(predicted_price)
        return predicted_price[0][0]

    def predict_next_x_days(self, amount_of_days):
        last_21_days = self.raw_values[-21:]
        last_21_days_scaled = self.scaler.fit_transform(last_21_days.reshape(-1, 1))
        prediction_base = []
        prediction_base.append(last_21_days_scaled)
        prediction_base = np.array(prediction_base)
        today = datetime.now()
        predictions = {}
        final_prediction_base = prediction_base

        for i in range(amount_of_days-1):
            predicted_price = self.model.predict(prediction_base)
            prediction_base = np.roll(prediction_base, -1)
            prediction_base[-1][-1] = predicted_price[0][0]
            final_prediction_base = np.concatenate([final_prediction_base, prediction_base])
            print(final_prediction_base.shape)

        final_prediction = self.model.predict(final_prediction_base)
        final_prediction = self.scaler.inverse_transform(final_prediction)

        for i in range(len(final_prediction)-1):
            next_day = today + timedelta(days=i + 1)
            predictions[next_day.strftime('%m-%d')] = final_prediction[i][0]

        values = list(predictions.values())
        labels = list(predictions.keys())

        plt.plot(labels, values)
        plt.xticks(rotation=45)
        plt.show()

        return predictions

    def predict_next_x_days_without_loop(self, amount_of_days):
        last_21_days = self.raw_values[-(21+amount_of_days):]
        last_21_days_scaled = self.scaler.fit_transform(last_21_days.reshape(-1, 1))
        prediction_base = []
        prediction_base.append(last_21_days_scaled[:21, :])
        prediction_base = np.array(prediction_base)
        today = datetime.now()
        predictions = {}
        final_prediction_base = prediction_base

        for i in range(1, amount_of_days):
            predicted_price = last_21_days_scaled[i:21+i, :]
            prediction_base = np.roll(prediction_base, -1)
            prediction_base[-1][-1] = predicted_price[0]
            final_prediction_base = np.concatenate([final_prediction_base, prediction_base])
            print(final_prediction_base.shape)

        final_prediction = self.model.predict(final_prediction_base)
        final_prediction = self.scaler.inverse_transform(final_prediction)

        for i in range(len(final_prediction)-1):
            next_day = today + timedelta(days=i + 1)
            predictions[next_day.strftime('%m-%d')] = final_prediction[i][0]

        values = list(predictions.values())
        labels = list(predictions.keys())

        plt.plot(labels, values)
        plt.xticks(rotation=45)
        plt.show()

        return predictions
    # def predict_next_day_test(self):
    #     errors = []
    #     j = 0
    #     for i in range(1000):
    #         start_index = np.random.randint(0, len(self.raw_values) - 61)
    #         indexes = np.arange(start_index, start_index + 60)
    #         last_60_days = self.raw_values[indexes]
    #         print('rzeczywista: ' + str(self.raw_values[start_index+61]))
    #         last_60_days_scaled = self.scaler.fit_transform(last_60_days.reshape(-1, 1))
    #         X_test = []
    #         X_test.append(last_60_days_scaled)
    #         X_test = np.array(X_test)
    #         predicted_price = self.model.predict(X_test)
    #         predicted_price = self.scaler.inverse_transform(predicted_price)
    #         print('przewidywana: ' + str(predicted_price[0][0]))
    #         diff_between_days = self.raw_values[start_index+60] - predicted_price[0][0]
    #         diff = self.raw_values[start_index+61] - predicted_price[0][0]
    #         if( diff > 0 and diff_between_days < 0):
    #             j = j + 1
    #         errors.append(diff)
    #     print(errors)
    #     print('rzeczywista cena była wyższa od przewidywanej mając na uwadze to, że cena z dnia poprzedniego była niższa: ' + str(j) + ' razy')
