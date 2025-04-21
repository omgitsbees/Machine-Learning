import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Add technical indicators
def add_technical_indicators(data):
    """
    Add technical indicators like moving averages, RSI, and MACD to the dataset.
    """
    data['SMA_20'] = data['Close'].rolling(window=20).mean()  # 20-day Simple Moving Average
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
    data['RSI'] = compute_rsi(data['Close'], window=14)  # Relative Strength Index
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data = data.dropna()  # Drop rows with NaN values
    return data

def compute_rsi(series, window):
    """
    Compute the Relative Strength Index (RSI).
    """
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Load and preprocess data
def load_data(filepath, sequence_length=60):
    """
    Load stock price data, add technical indicators, and preprocess it for the LSTM model.
    """
    data = pd.read_csv(filepath)
    data = add_technical_indicators(data)
    features = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD']
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predict the 'Close' price
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler, data

# Build the LSTM model
def build_model(input_shape):
    """
    Build an LSTM model for stock price prediction.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer for price prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the model with checkpointing
def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    """
    Train the LSTM model with model checkpointing.
    """
    checkpoint = ModelCheckpoint('best_model.h5', monitor='loss', save_best_only=True, verbose=1)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
    return model

# Backtesting
def backtest(data, predictions, initial_balance=10000):
    """
    Simulate a trading strategy using the model's predictions.
    """
    balance = initial_balance
    position = 0  # 0 means no position, 1 means holding stock
    for i in range(len(predictions) - 1):
        if predictions[i + 1] > predictions[i] and position == 0:  # Buy signal
            position = balance / data['Close'].iloc[i]
            balance = 0
        elif predictions[i + 1] < predictions[i] and position > 0:  # Sell signal
            balance = position * data['Close'].iloc[i]
            position = 0
    # Final value
    if position > 0:
        balance = position * data['Close'].iloc[-1]
    print(f"Final Balance: ${balance:.2f}")

# Evaluate the model
def evaluate_model(model, X_test, y_test, scaler, data):
    """
    Evaluate the model and plot predictions vs actual prices.
    """
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], 4)))))[:, 0]
    y_test = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 4)))))[:, 0]

    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'].iloc[-len(y_test):].values, color='blue', label='Actual Prices')
    plt.plot(predictions, color='red', label='Predicted Prices')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse:.2f}")

    # Backtest the strategy
    backtest(data.iloc[-len(y_test):], predictions)

# Main workflow
def main():
    # Replace 'stock_data.csv' with the path to your dataset
    filepath = 'stock_data.csv'
    sequence_length = 60  # Use the last 60 days to predict the next day's price

    # Load and preprocess data
    X, y, scaler, data = load_data(filepath, sequence_length)
    split = int(0.8 * len(X))  # 80% training, 20% testing
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build, train, and evaluate the model
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model = train_model(model, X_train, y_train, epochs=50, batch_size=32)
    evaluate_model(model, X_test, y_test, scaler, data)

if __name__ == "__main__":
    main()