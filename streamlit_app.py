import streamlit as st
from main_script import fetch_stock_data, preprocess_data, create_sequences, build_lstm_model, predict_next_3_months, plot_predictions
from keras.models import load_model

st.title("Stock Price Prediction App")
ticker = st.text_input("Enter stock ticker symbol", "BTC-USD")

model = load_model(ticker+'_model.h5')
# Option to select prediction duration
prediction_duration = st.selectbox("Select prediction duration:", ["3 Months", "6 Months", "1 Year"])

if st.button("Predict"):
    df = fetch_stock_data(ticker)

    seq_length = 60
    scaled_data, scaler = preprocess_data(df)
    X, y = create_sequences(scaled_data, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # model = build_lstm_model((X_train.shape[1], 1))
    # model.fit(X_train, y_train, batch_size=32, epochs=10)

    # Determine prediction days based on selection
    if prediction_duration == "3 Months":
        prediction_days = 60
    elif prediction_duration == "6 Months":
        prediction_days = 120
    else:  # "1 Year"
        prediction_days = 240

    predictions = predict_next_3_months(model, scaled_data, scaler, prediction_days=prediction_days)
    
    st.write(f"Predictions for the next {prediction_duration} for {ticker}:")
    plot_predictions(df, predictions, ticker, prediction_days)


