import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Synthetic Data Generation ---
print("1. Generating Synthetic Dataset...")

np.random.seed(42) # for reproducibility

num_videos = 5000

# Basic video features
video_ids = [f'vid_{i:04d}' for i in range(num_videos)]
title_lengths = np.random.randint(10, 100, num_videos) # characters
description_lengths = np.random.randint(50, 500, num_videos) # characters
tags_count = np.random.randint(1, 20, num_videos) # number of tags

categories = ['Gaming', 'Music', 'Education', 'Comedy', 'News', 'Vlogs', 'Sports', 'Science & Tech']
video_categories = np.random.choice(categories, num_videos, p=[0.15, 0.2, 0.1, 0.15, 0.1, 0.1, 0.05, 0.15])

publish_hours = np.random.randint(0, 24, num_videos)
publish_days = np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], num_videos)

# Initial engagement metrics (e.g., within first 24 hours)
# These will influence trending status
views_initial_24h = np.random.randint(1000, 500000, num_videos)
likes_initial_24h = np.random.randint(100, 50000, num_videos)
dislikes_initial_24h = np.random.randint(10, 5000, num_videos)
comments_initial_24h = np.random.randint(5, 1000, num_videos)

# Simulate trending status: higher engagement -> more likely to trend
# This is a simplified logic. In real life, it's more complex (rate of growth etc.)
trending_threshold_views = np.percentile(views_initial_24h, 75) # Top 25% views
trending_threshold_likes = np.percentile(likes_initial_24h, 70) # Top 30% likes
trending_threshold_comments = np.percentile(comments_initial_24h, 60) # Top 40% comments

is_trending = (
    (views_initial_24h > trending_threshold_views) &
    (likes_initial_24h > trending_threshold_likes) &
    (comments_initial_24h > trending_threshold_comments)
)
# Ensure some videos are trending
is_trending = is_trending.astype(int) # Convert boolean to 0/1

# Create DataFrame
data = pd.DataFrame({
    'video_id': video_ids,
    'title_length': title_lengths,
    'description_length': description_lengths,
    'tags_count': tags_count,
    'category': video_categories,
    'publish_hour': publish_hours,
    'publish_day_of_week': publish_days,
    'views_initial_24h': views_initial_24h,
    'likes_initial_24h': likes_initial_24h,
    'dislikes_initial_24h': dislikes_initial_24h,
    'comments_initial_24h': comments_initial_24h,
    'is_trending': is_trending
})

print("Synthetic data created. Sample Head:")
print(data.head())
print("\nTrending status distribution:")
print(data['is_trending'].value_counts())

# --- 2. Feature Engineering (Basic) ---
print("\n2. Performing Basic Feature Engineering...")

# Engagement Ratios
data['likes_ratio'] = data['likes_initial_24h'] / data['views_initial_24h']
data['comments_ratio'] = data['comments_initial_24h'] / data['views_initial_24h']
# Handle potential division by zero if views can be zero (not in this synthetic data, but good practice)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(0, inplace=True) # Replace NaNs (from division by zero) with 0

# Combine likes and dislikes into a single "engagement score" or similar
data['net_likes_ratio'] = (data['likes_initial_24h'] - data['dislikes_initial_24h']) / data['views_initial_24h']
data.fillna(0, inplace=True)

print("Engineered features added. Sample Head with new features:")
print(data.head())


# --- 3. Data Preprocessing ---
print("\n3. Preprocessing Data...")

# Define features (X) and target (y)
X = data.drop(['video_id', 'is_trending'], axis=1) # video_id is just an identifier
y = data['is_trending']

# Identify categorical and numerical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler() # Scale numerical features
categorical_transformer = OneHotEncoder(handle_unknown='ignore') # One-hot encode categorical features

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 4. Model Training ---
print("\n4. Training the Model...")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# stratify=y ensures that the proportion of trending/non-trending videos is similar in train/test sets

# Create a pipeline that first preprocesses the data and then trains a RandomForestClassifier
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))])
# class_weight='balanced' helps with imbalanced datasets (if one class is much smaller than the other)

# Train the model
model_pipeline.fit(X_train, y_train)

print("Model training complete.")

# --- 5. Model Evaluation ---
print("\n5. Evaluating the Model...")

y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1] # Probability of being trending

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Not Trending', 'Trending'],
            yticklabels=['Not Trending', 'Trending'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# --- 6. Prediction on New Data ---
print("\n6. Demonstrating Prediction on New (Simulated) Data...")

# Create some hypothetical new video data
new_video_data = pd.DataFrame({
    'title_length': [55, 20, 80],
    'description_length': [300, 150, 450],
    'tags_count': [10, 5, 18],
    'category': ['Music', 'Comedy', 'News'],
    'publish_hour': [18, 12, 9],
    'publish_day_of_week': ['Fri', 'Mon', 'Wed'],
    'views_initial_24h': [600000, 15000, 80000], # One potentially trending, one low, one medium
    'likes_initial_24h': [80000, 1500, 9000],
    'dislikes_initial_24h': [500, 200, 1000],
    'comments_initial_24h': [2000, 50, 500]
})

# Apply the same feature engineering to new data
new_video_data['likes_ratio'] = new_video_data['likes_initial_24h'] / new_video_data['views_initial_24h']
new_video_data['comments_ratio'] = new_video_data['comments_initial_24h'] / new_video_data['views_initial_24h']
new_video_data['net_likes_ratio'] = (new_video_data['likes_initial_24h'] - new_video_data['dislikes_initial_24h']) / new_video_data['views_initial_24h']
new_video_data.replace([np.inf, -np.inf], np.nan, inplace=True)
new_video_data.fillna(0, inplace=True)

print("New video data for prediction:")
print(new_video_data)

# Predict
new_predictions = model_pipeline.predict(new_video_data)
new_probabilities = model_pipeline.predict_proba(new_video_data)[:, 1] # Probability of trending

print("\nPrediction Results for New Videos:")
for i, pred in enumerate(new_predictions):
    status = "TRENDING" if pred == 1 else "NOT TRENDING"
    prob = new_probabilities[i]
    print(f"Video {i+1}: Predicted as {status} (Probability: {prob:.4f})")

print("\n--- Model Building Complete ---")
print("Remember: This model is built on synthetic data. For a real-world application,")
print("you would need actual YouTube data and potentially more sophisticated features and models.")