import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from flask import Flask, request, jsonify

# Step 1: Load Dataset
def load_data():
    # Example dataset: Replace with your actual transaction dataset
    data = pd.read_csv("transactions.csv")
    return data

# Step 2: Preprocess Data
def preprocess_data(data):
    # Handle missing values
    data.fillna(0, inplace=True)

    # Feature engineering (e.g., create new features if needed)
    data['Transaction_Amount_Log'] = np.log1p(data['Transaction_Amount'])

    # Split features and target
    X = data.drop(columns=['Is_Fraud', 'Transaction_ID'])
    y = data['Is_Fraud']

    return X, y

# Step 3: Handle Imbalanced Data
def handle_imbalance(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Step 4: Train Model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Step 5: Anomaly Detection
def detect_anomalies(X):
    isolation_forest = IsolationForest(random_state=42, contamination=0.01)
    isolation_forest.fit(X)
    anomalies = isolation_forest.predict(X)
    return anomalies

# Step 6: Explainability with SHAP
def explain_model(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, plot_type="bar")

# Step 7: Real-Time Fraud Detection API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_fraud():
    data = request.get_json()
    X_new = pd.DataFrame(data)
    prediction = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[:, 1]
    return jsonify({"prediction": prediction.tolist(), "probabilities": probabilities.tolist()})

# Step 8: Visualization Dashboard
def visualize_data(data):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Is_Fraud', data=data)
    plt.title("Fraud vs Non-Fraud Transactions")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Is_Fraud', y='Transaction_Amount', data=data)
    plt.title("Transaction Amount Distribution by Fraud Status")
    plt.show()

# Step 9: Main Workflow
def main():
    # Load and preprocess data
    data = load_data()
    X, y = preprocess_data(data)

    # Visualize data
    visualize_data(data)

    # Handle imbalanced data
    X_resampled, y_resampled = handle_imbalance(X, y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train the model
    global model
    model = train_model(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    # Detect anomalies
    anomalies = detect_anomalies(X)
    print(f"Number of anomalies detected: {sum(anomalies == -1)}")

    # Explain the model
    explain_model(model, X_test.sample(100))

    # Start the real-time fraud detection API
    app.run(debug=True)

if __name__ == "__main__":
    main()