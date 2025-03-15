from flask import Flask, render_template, request
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')  # Prevent GUI issues with Matplotlib
import matplotlib.pyplot as plt
import io
import base64
import os
from dotenv import load_dotenv
from statsmodels.tsa.arima.model import ARIMA

# Load environment variables
load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")  # Read API key from .env

# Function to fetch S&P 500 data from Alpha Vantage
def fetch_sp500_data():
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=SPY&outputsize=full&apikey={API_KEY}"
    response = requests.get(url).json()
    
    if "Time Series (Daily)" not in response:
        return None
    
    df = pd.DataFrame.from_dict(response["Time Series (Daily)"], orient="index")
    df = df.rename(columns={"4. close": "Close"}).astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()  # Ensure data is sorted by date
    return df[["Close"]]

def fetch_lstm_forecast():
    url=f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY_FORECAST",
        "symbol": "SPY",
        "apikey": ALPHA_VANTAGE_API_KEY # type: ignore
    }

    response = requests.get(url, params, params=params).json()

    if "forecasts" not in response: 
        return None
    
    forecast_data = response["forecasts"]
    df_forecast = pd.DataFrame(forecast_data)

    df_forecast["timestamp"] = pd.to_datetime(df_forecast["timestamp"])
    df_forecast.set_index ("timestamp", inplace=True)

    return df_forecast

@app.route("/", methods=["GET", "POST"])
def index():
    forecast_arima = None
    forecast_lstm = None
    plot_url = None
    
    if request.method == "POST":
        train_window = int(request.form["train_window"])
        pred_window = int(request.form["pred_window"])
        
        df = fetch_sp500_data()
        if df is None:
            return "Error fetching S&P 500 data. Try again later."
        
        df_lstm = fetch_lstm_forecast
        
        # Select training data
        train_data = df["Close"].iloc[-train_window:]

        # Fit ARIMA model
        model = ARIMA(train_data, order=(5, 1, 0))  # Adjust p, d, q as needed
        model_fit = model.fit()

        # Forecast
        forecast_arima = model_fit.forecast(steps=pred_window)

        # Generate plot
        plt.figure(figsize=(10, 5))
        plt.plot(df.index[-train_window:], train_data, label="Training Data", color="blue")
        forecast_index = pd.date_range(df.index[-1], periods=pred_window + 1, freq="D")[1:]
        plt.plot(forecast_index, forecast_arima, label="Forecast", linestyle="dashed", color="red")

        if df_lstm is not None:
            forecast_lstm = df_lstm["forecast"].iloc[:pred_window]
            plt.plot(df_lstm.index[:pred_window], forecast_lstm, label="LSTM Forecast", linestyle="dotted", color="green")

        plt.xlabel("Date")
        plt.ylabel("S&P 500 Price")
        plt.legend()
        plt.title("S&P 500 ARIMA Forecast: ARIMA vs LSTM")

        # Convert plot to base64 image
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

    return render_template("index.html", forecast_arima=forecast_arima, forecast_lstm=forecast_lstm, plot_url=plot_url)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
