import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import timedelta
import streamlit as st

# Function to fetch data
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="5y")  # Fetch 5 years of historical data
    return df[['Close']]  # We only care about the 'Close' price for this example

# Preprocessing the data
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

# Create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(seq_length, len(data)):
        sequences.append(data[i-seq_length:i, 0])
        labels.append(data[i, 0])
    return np.array(sequences), np.array(labels)

# Build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))  # Output layer for price prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict the next 3 months (60 trading days)
def predict_next_3_months(model, data, scaler, seq_length=60, prediction_days=60):
    predictions = []
    current_sequence = data[-seq_length:].reshape(1, seq_length, 1)  # Ensure 3D shape

    for _ in range(prediction_days):
        next_price = model.predict(current_sequence)

        # Update the sequence with the predicted value
        next_price_reshaped = np.reshape(next_price, (1, 1, 1))  # Reshape to match the dimension
        current_sequence = np.append(current_sequence[:, 1:, :], next_price_reshaped, axis=1)

        # Store the prediction
        predictions.append(next_price[0, 0])

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Function to visualize predictions with more granular details
def plot_predictions(df, predictions, ticker, prediction_days):
    future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=len(predictions), freq='B')
    prediction_df = pd.DataFrame(predictions, columns=['Predicted'], index=future_dates)

    # Plot the actual and predicted prices
    fig1 = plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Actual Prices', color='blue')
    plt.plot(prediction_df.index, prediction_df['Predicted'], label='Predicted Prices', color='orange', linestyle='--')
    plt.title(f'{ticker} Stock Price Prediction for Next {prediction_days // 20} Month(s)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig1)

    # Zoomed Views
    for month in range(1, (prediction_days // 20) + 1):
        month_start = (month - 1) * 20
        month_end = month * 20
        if month_end > len(predictions):
            month_end = len(predictions)

        fig = plt.figure(figsize=(14, 7))
        plt.plot(prediction_df.index[month_start:month_end], prediction_df['Predicted'][month_start:month_end],
                 label=f'Predicted Prices for Month {month}', color='orange', linestyle='--')
        plt.title(f'{ticker} Predicted Prices for Month {month}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)

if __name__ == '__main__':
    ticker = input("Enter stock ticker symbol: ")
    df = fetch_stock_data(ticker)
    
    # Preprocessing
    seq_length = 60  # Sequence length for LSTM
    scaled_data, scaler = preprocess_data(df)
    
    # Create sequences
    X, y = create_sequences(scaled_data, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM input

    # Split the data (80% training, 20% testing)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build and train the model
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, batch_size=32, epochs=10)
    model.save(ticker+"_model.h5")
    # Predict the next 3 months
    predictions = predict_next_3_months(model, scaled_data, scaler)

    # Plot the results with more granularity
    plot_predictions(df, predictions, ticker, 60)