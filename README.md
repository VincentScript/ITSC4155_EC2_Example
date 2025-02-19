# ITSC4155_EC2_Example
This Flask application fetches S&P 500 data from Alpha Vantage, allows users to set training and prediction windows, applies the ARIMA model for forecasting, and visualizes the results.

## Installation
1. Clone this repository:
```bash
git clone https://github.com/rasmuserlemann/ITSC4155_EC2_Example.git
cd ITSC4155_EC2_Example
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create ```.env``` file and add your Alpha Vantager API key:
```bash
ALPHA_VANTAGE_API_KEY=your_api_key_here
```
4. Run the Flask app:
```bash
python app.py
```
5. Open your browser (for local) and go to:
```bash
http://127.0.0.1:5000
```