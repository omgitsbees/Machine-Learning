import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error

# Example: Load your data
# Data should include: user_id, notification_type, timestamp, user_features..., label_clicked, label_optimal_time
data = pd.read_csv('notifications_data.csv')

# --- Feature Engineering ---
# Encode categorical features
le_user = LabelEncoder()
data['user_id_enc'] = le_user.fit_transform(data['user_id'])
le_type = LabelEncoder()
data['notification_type_enc'] = le_type.fit_transform(data['notification_type'])

# Extract time features
data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
data['dayofweek'] = pd.to_datetime(data['timestamp']).dt.dayofweek

# Example user features (extend as needed)
user_features = ['user_id_enc', 'hour', 'dayofweek', 'notification_type_enc'] + [col for col in data.columns if col.startswith('user_feat_')]

# --- Relevance Model (Classification: Will user click?) ---
X_relevance = data[user_features]
y_relevance = data['label_clicked']

X_train_rel, X_test_rel, y_train_rel, y_test_rel = train_test_split(X_relevance, y_relevance, test_size=0.2, random_state=42)

clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
clf.fit(X_train_rel, y_train_rel)
y_pred_rel = clf.predict_proba(X_test_rel)[:, 1]
print("Relevance ROC-AUC:", roc_auc_score(y_test_rel, y_pred_rel))

# --- Optimal Send Time Model (Regression: Best hour to send) ---
X_time = data[user_features]
y_time = data['label_optimal_time']  # e.g., hour of day with highest engagement

X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(X_time, y_time, test_size=0.2, random_state=42)

reg = XGBRegressor()
reg.fit(X_train_time, y_train_time)
y_pred_time = reg.predict(X_test_time)
print("Optimal Send Time MAE:", mean_absolute_error(y_test_time, y_pred_time))

# --- Inference Example ---
def recommend_notification(user_id, notification_type, user_feat_dict, current_time):
    # Prepare feature vector
    hour = pd.to_datetime(current_time).hour
    dayofweek = pd.to_datetime(current_time).dayofweek
    user_id_enc = le_user.transform([user_id])[0]
    notification_type_enc = le_type.transform([notification_type])[0]
    features = [user_id_enc, hour, dayofweek, notification_type_enc] + [user_feat_dict[k] for k in sorted(user_feat_dict)]
    features = np.array(features).reshape(1, -1)
    
    # Predict relevance
    relevance_score = clf.predict_proba(features)[0, 1]
    # Predict optimal send time
    optimal_hour = reg.predict(features)[0]
    return relevance_score, optimal_hour

# Example usage
user_feat_dict = {'user_feat_1': 0.5, 'user_feat_2': 1.2}  # Example user features
relevance, send_hour = recommend_notification('user_123', 'promo', user_feat_dict, '2023-07-01 10:00:00')
print(f"Predicted relevance: {relevance:.2f}, Optimal send hour: {send_hour:.2f}")