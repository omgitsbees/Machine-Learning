import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import networkx as nx
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import plot_model
import shap
import folium
from datetime import datetime

# Load and preprocess data
def load_data(filepath):
    """
    Load transaction data for AML analysis.
    """
    data = pd.read_csv(filepath)
    return data

# Feature engineering
def feature_engineering(data):
    """
    Create features for transaction analysis.
    """
    data['TransactionAmountLog'] = np.log1p(data['TransactionAmount'])
    data['TransactionFrequency'] = data.groupby('AccountID')['TransactionID'].transform('count')
    data['AverageTransactionAmount'] = data.groupby('AccountID')['TransactionAmount'].transform('mean')
    return data

# Deep learning for anomaly detection
def build_anomaly_detection_model(input_shape):
    """
    Build an LSTM-based deep learning model for anomaly detection.
    """
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_anomaly_detection_model(model, X_train, y_train, epochs=10, batch_size=32):
    """
    Train the LSTM model for anomaly detection.
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

# Explainability using SHAP
def explain_anomalies(model, X_sample):
    """
    Use SHAP to explain why certain transactions were flagged as anomalies.
    """
    explainer = shap.DeepExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample)

# Real-time monitoring
def real_time_monitoring(new_transaction, model, scaler):
    """
    Monitor transactions in real-time and flag anomalies.
    """
    scaled_transaction = scaler.transform([new_transaction])
    prediction = model.predict(scaled_transaction)
    if prediction[0][0] > 0.5:
        print("ALERT: Suspicious transaction detected!")
    else:
        print("Transaction is normal.")

# Geospatial analysis
def geospatial_analysis(data):
    """
    Visualize transaction locations to detect unusual patterns.
    """
    map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
    transaction_map = folium.Map(location=map_center, zoom_start=6)

    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color='red' if row['IsAnomalous'] == 1 else 'green',
            fill=True,
            fill_color='red' if row['IsAnomalous'] == 1 else 'green',
            fill_opacity=0.6
        ).add_to(transaction_map)

    transaction_map.save("transaction_map.html")
    print("Geospatial analysis map saved as 'transaction_map.html'.")

# Regulatory reporting
def generate_regulatory_report(data):
    """
    Generate a report of flagged transactions for regulatory compliance.
    """
    flagged_transactions = data[data['IsAnomalous'] == 1]
    report_filename = f"regulatory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    flagged_transactions.to_csv(report_filename, index=False)
    print(f"Regulatory report saved as {report_filename}.")

# Main workflow
def main():
    # Replace 'transactions.csv' with the path to your dataset
    filepath = 'transactions.csv'

    # Load and preprocess data
    data = load_data(filepath)
    data = feature_engineering(data)

    # Prepare data for deep learning
    features = ['TransactionAmountLog', 'TransactionFrequency', 'AverageTransactionAmount']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])
    X = scaled_data
    y = data['IsAnomalous'] if 'IsAnomalous' in data.columns else np.zeros(len(data))  # Placeholder for labels

    # Build and train anomaly detection model
    model = build_anomaly_detection_model((X.shape[1], 1))
    model = train_anomaly_detection_model(model, X, y, epochs=10, batch_size=32)

    # Explain anomalies
    explain_anomalies(model, X[:100])  # Use a sample of 100 transactions for SHAP

    # Real-time monitoring
    new_transaction = [np.log1p(5000), 10, 2000]  # Example transaction
    real_time_monitoring(new_transaction, model, scaler)

    # Geospatial analysis
    if 'Latitude' in data.columns and 'Longitude' in data.columns:
        geospatial_analysis(data)

    # Regulatory reporting
    generate_regulatory_report(data)

if __name__ == "__main__":
    main()