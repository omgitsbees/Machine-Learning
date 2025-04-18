import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import shap
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load dataset
def load_data(filepath):
    """
    Load and preprocess the dataset.
    """
    logging.info("Loading dataset...")
    data = pd.read_csv(filepath)
    data = data.dropna()  # Drop missing values
    data = pd.get_dummies(data, drop_first=True)  # Encode categorical variables
    logging.info("Dataset loaded and preprocessed.")
    return data

# Preprocess data
def preprocess_data(data, target_column):
    """
    Split data into features and target, handle imbalance, and scale features.
    """
    logging.info("Preprocessing data...")
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    
    logging.info("Data preprocessing complete.")
    return train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# Train model
def train_model(X_train, y_train):
    """
    Train a Gradient Boosting Classifier.
    """
    logging.info("Training model...")
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    logging.info("Model training complete.")
    return model

# Hyperparameter tuning
def tune_hyperparameters(X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV.
    """
    logging.info("Tuning hyperparameters...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    model = GradientBoostingClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    logging.info(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Evaluate model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using classification report and ROC AUC score.
    """
    logging.info("Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.2f}")

# Feature importance analysis
def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for the trained model.
    """
    logging.info("Plotting feature importance...")
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(importance)), importance[indices], align="center")
    plt.xticks(range(len(importance)), np.array(feature_names)[indices], rotation=90)
    plt.tight_layout()
    plt.show()

# Explain model predictions
def explain_model_predictions(model, X_sample):
    """
    Use SHAP to explain model predictions.
    """
    logging.info("Explaining model predictions with SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample)

# Cross-validation
def cross_validate_model(model, X, y):
    """
    Perform k-fold cross-validation.
    """
    logging.info("Performing cross-validation...")
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"Cross-Validation AUC Scores: {scores}")
    print(f"Mean AUC: {scores.mean():.2f}")

# Main workflow
def main():
    # Replace 'credit_data.csv' with the path to your dataset
    filepath = 'credit_data.csv'
    target_column = 'default'  # Replace with the actual target column name
    
    data = load_data(filepath)
    X_train, X_test, y_train, y_test = preprocess_data(data, target_column)
    
    # Train and evaluate the model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
    # Optional: Hyperparameter tuning
    # model = tune_hyperparameters(X_train, y_train)
    # evaluate_model(model, X_test, y_test)
    
    # Feature importance
    plot_feature_importance(model, data.drop(columns=[target_column]).columns)
    
    # Explainability with SHAP
    explain_model_predictions(model, X_test[:100])  # Use a sample of test data for SHAP
    
    # Cross-validation
    cross_validate_model(model, X_train, y_train)

if __name__ == "__main__":
    main()