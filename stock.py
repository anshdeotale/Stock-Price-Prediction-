# stock_price_prediction_improved.py
# Improved Stock Price Prediction with multiple lag features

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import timedelta

# ----------------------------
# 1. Download stock data
# ----------------------------
ticker = "AAPL"   # Change to RELIANCE.NS, TCS.NS, etc. for Indian stocks
data = yf.download(ticker, start="2022-01-01", end="2025-01-01")

data = data[['Close']].dropna()

# ----------------------------
# 2. Create lag features
# ----------------------------
# Example: use last 5 days to predict next day
lags = 5
for i in range(1, lags+1):
    data[f'lag_{i}'] = data['Close'].shift(i)

data['Target'] = data['Close']
data = data.dropna()

X = data[[f'lag_{i}' for i in range(1, lags+1)]]
y = data['Target']

# ----------------------------
# 3. Train/test split
# ----------------------------
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ----------------------------
# 4. Train Linear Regression
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# ----------------------------
# 5. Evaluate & Plot
# ----------------------------
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"MSE: {mse:.2f}, R2: {r2:.2f}")

plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual Prices")
plt.plot(predictions, label="Predicted Prices")
plt.legend()
plt.title(f"{ticker} Stock Price Prediction (Linear Regression with {lags} lags)")
plt.show()

# ----------------------------
# 6. Forecast next N days
# ----------------------------
N = 7  # forecast horizon
last_values = X.iloc[-1].values.reshape(1, -1)
future_dates, future_prices = [], []

for i in range(N):
    next_price = model.predict(last_values)[0]
    next_date = data.index[-1] + timedelta(days=i+1)
    future_dates.append(next_date)
    future_prices.append(next_price)

    # update lags
    last_values = np.roll(last_values, -1)
    last_values[0, -1] = next_price

forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_prices})
print("\nForecast for next", N, "days:")
print(forecast_df)
