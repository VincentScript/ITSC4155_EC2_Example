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

@app.route("/", methods=["GET", "POST"])
def index():
    forecast = None
    plot_url = None
    
    if request.method == "POST":
        train_window = int(request.form["train_window"])
        pred_window = int(request.form["pred_window"])
        
        df = fetch_sp500_data()
        if df is None:
            return "Error fetching S&P 500 data. Try again later."
        
        # Select training data
        train_data = df["Close"].iloc[-train_window:]

        # Fit ARIMA model
        model = ARIMA(train_data, order=(5, 1, 0))  # Adjust p, d, q as needed
        model_fit = model.fit()

        # Forecast
        forecast = model_fit.forecast(steps=pred_window)

        # Generate plot
        plt.figure(figsize=(10, 5))
        plt.plot(df.index[-train_window:], train_data, label="Training Data", color="blue")
        forecast_index = pd.date_range(df.index[-1], periods=pred_window + 1, freq="D")[1:]
        plt.plot(forecast_index, forecast, label="Forecast", linestyle="dashed", color="red")
        plt.xlabel("Date")
        plt.ylabel("S&P 500 Price")
        plt.legend()
        plt.title("S&P 500 ARIMA Forecast")

        # Convert plot to base64 image
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

    return render_template("index.html", forecast=forecast if forecast is not None and not forecast.empty else None, plot_url=plot_url)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
