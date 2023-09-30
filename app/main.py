import os
from os import environ
import uvicorn
from predicator import Predicator
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://investor-assistent-frontend-7a03806b8924.herokuapp.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def model_trained(ticker: str):
    if os.path.exists('./models/' + ticker + '_model.h5'):
        print('istnieje')
        return True
    else:
        return False


@app.get("/{ticker}/predict/next-day")
async def next_day_prediction(ticker: str, background_tasks: BackgroundTasks):
    if model_trained(ticker):
        predicator = Predicator(ticker)
        next_price = predicator.predict_next_day()
        return str(next_price)
    else:
        background_tasks.add_task(Predicator, ticker)
    return 'I need to train myself, give me 30 minutes and try again'


@app.get("/{ticker}/predict/next-days/{amount}")
async def next_x_days_prediction(ticker: str, amount: int, background_tasks: BackgroundTasks):
    if model_trained(ticker):
        predicator = Predicator(ticker)
        next_prices = predicator.predict_next_x_days(amount)
        return str(next_prices)
    else:
        background_tasks.add_task(Predicator, ticker)
    return 'I need to train myself, give me 30 minutes and try again'


@app.get("/{ticker}/predict-without-loop/next-days/{amount}")
async def next_x_days_prediction_without_loop(ticker: str, amount: int, background_tasks: BackgroundTasks):
    if model_trained(ticker):
        predicator = Predicator(ticker)
        next_prices = predicator.predict_next_x_days_without_loop(amount)
        return str(next_prices)
    else:
        background_tasks.add_task(Predicator, ticker)
    return 'I need to train myself, give me 30 minutes and try again'

if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=int(environ.get("PORT", 5000)))
