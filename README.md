# FastAPI Stock Price Prediction API

This FastAPI application provides endpoints to predict stock prices for the next day or a specified number of days. It uses a machine learning model trained on historical stock price data.

## Endpoints

### GET `/{ticker}/predict/next-day`

Returns the predicted stock price for the next day.

#### Path Parameters

- `ticker` (string): The stock ticker symbol.

#### Response

Returns the predicted stock price as a string. If the model is not trained for the specified ticker, it will start the training process in the background and return a message to try again in 30 minutes.

### GET `/{ticker}/predict/next-days/{amount}`

Returns the predicted stock prices for the next specified number of days.

#### Path Parameters

- `ticker` (string): The stock ticker symbol.
- `amount` (integer): The number of days for the prediction.

#### Response

Returns a string of the predicted stock prices. If the model is not trained for the specified ticker, it will start the training process in the background and return a message to try again in 30 minutes.

### GET `/{ticker}/predict-without-loop/next-days/{amount}`

Returns the predicted stock prices for the next specified number of days without using a loop.

#### Path Parameters

- `ticker` (string): The stock ticker symbol.
- `amount` (integer): The number of days for the prediction.

#### Response

Returns a string of the predicted stock prices. If the model is not trained for the specified ticker, it will start the training process in the background and return a message to try again in 30 minutes.

## Usage

You can start the server by running the application script. The server will start on `0.0.0.0:8080`.

Here's how to run the script:

```bash
python main.py