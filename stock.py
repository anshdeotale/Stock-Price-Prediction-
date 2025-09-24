# stock_price_prediction_app.py
# Interactive Stock Price Prediction with Streamlit

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import timedelta
import plotly.graph_objs as go

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📈 Stock Price Prediction App")
st.markdown("Predict stock prices using lag features + Linear Regression.")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT, RELIANCE.NS):", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-01-01"))
lags = st.slider("Number of lag days", 3, 15, 5)
forecast_days = st.slider("Forecast horizon (days)", 1, 30, 7)

if st.button("Run Prediction"):
    # ----------------------------
    # 1. Download stock data
    # ----------------------------
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close']].dropna()

    if len(data) < lags + 1:
        st.error("Not enough data to create lag features. Try different dates.")
        st.stop()

    # ----------------------------
    # 2. Create lag features
    # ----------------------------
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
    # 5. Evaluate
    # ----------------------------
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    st.success(f"Model Performance → MSE: {mse:.2f}, R²: {r2:.2f}")

    # ----------------------------
    # 6. Plot Actual vs Predicted
    # ----------------------------
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=y_test.values, mode='lines', name="Actual Prices"))
    fig1.add_trace(go.Scatter(y=predictions, mode='lines', name="Predicted Prices"))
    fig1.update_layout(title=f"{ticker} Stock Price Prediction (with {lags} lags)",
                       xaxis_title="Days", yaxis_title="Price")
    st.plotly_chart(fig1)

    # ----------------------------
    # 7. Forecast next N days
    # ----------------------------
    last_values = X.iloc[-1].values.reshape(1, -1)
    future_dates, future_prices = [], []

    for i in range(forecast_days):
        next_price = model.predict(last_values)[0]
        next_date = data.index[-1] + timedelta(days=i+1)
        future_dates.append(next_date)
        future_prices.append(next_price)

        # update lags
        last_values = np.roll(last_values, -1)
        last_values[0, -1] = next_price

    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_prices})

    # ----------------------------
    # 8. Plot Forecast
    # ----------------------------
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name="Historical"))
    fig2.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted_Price'],
                              mode='lines+markers', name="Forecast"))
    fig2.update_layout(title=f"{ticker} Stock Price Forecast ({forecast_days} days)",
                       xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig2)

    # Show forecast table
    st.subheader("📅 Forecasted Prices")
    st.dataframe(forecast_df)
