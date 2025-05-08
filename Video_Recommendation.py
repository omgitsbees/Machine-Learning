import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
import pandas as pd

print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow Recommenders version: {tfrs.__version__}")

# --- 1. Dummy Data Generation ---
# Simulates some user-video interactions and video metadata

# Video Metadata
video_data = {
    "video_id": [f"vid{i:03}" for i in range(1, 21)], # 20 unique videos
    "video_title": [
        "Cute Kittens Playing", "Learn Python Basics", "Space Exploration Doc", "Easy Pasta Recipe",
        "Funny Dog Fails", "Advanced JavaScript Tutorial", "History of Rome", "Home Workout Routine",
        "Travel Vlog: Japan", "Financial Planning Tips", "Sci-Fi Movie Review", "Gardening for Beginners",
        "Understanding AI", "Best Video Games 2025", "Acoustic Guitar Cover", "DIY Home Decor",
        "Healthy Breakfast Ideas", "Car Maintenance 101", "Wildlife Photography", "Stand-up Comedy Special"
    ],
    "video_category": [
        "Animals", "Education", "Documentary", "Food", "Animals", "Education", "History", "Fitness",
        "Travel", "Finance", "Movies", "Hobbies", "Technology", "Gaming", "Music", "DIY",
        "Food", "Automotive", "Nature", "Comedy"
    ]
}
videos_df = pd.DataFrame(video_data)

# User-Video Interactions (Ratings/Watches)
# Simulating that users have watched/rated certain videos
num_interactions = 100
user_ids = [f"user{i:03}" for i in range(1, 11)] # 10 unique users

# Generate random interactions
rng = np.random.RandomState(42)
interaction_user_ids = rng.choice(user_ids, size=num_interactions)
interaction_video_ids = rng.choice(videos_df["video_id"].values, size=num_interactions)
# For simplicity, we'll treat every interaction as a positive signal (e.g., a watch)

interactions_df = pd.DataFrame({
    "user_id": interaction_user_ids,
    "video_id": interaction_video_ids
})

print("\nSample Video Data:")
print(videos_df.head())
print("\nSample Interaction Data:")
print(interactions_df.head())

# --- 2. Data Preparation ---

# Convert to TensorFlow Datasets
# For interactions (what we train on)
ratings_ds = tf.data.Dataset.from_tensor_slices({
    "user_id": interactions_df["user_id"].values,
    "video_id": interactions_df["video_id"].values
})

# For videos (the corpus we recommend from)
# We only need video_id for this basic retrieval model, but in a more complex model,
# you'd map video_id to its features (title, category embeddings etc.)
videos_ds = tf.data.Dataset.from_tensor_slices(videos_df["video_id"].values)

# Prepare vocabularies for user IDs and video IDs
unique_user_ids_vocab = np.unique(interactions_df["user_id"].values)
unique_video_ids_vocab = np.unique(videos_df["video_id"].values) # All known video IDs

# --- 3. Model Definition ---
embedding_dimension = 64 # Size of the learned user/video representation

# User Tower: Converts user ID to an embedding vector
user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids_vocab, mask_token=None),
    tf.keras.layers.Embedding(len(unique_user_ids_vocab) + 1, embedding_dimension) # +1 for OOV
])

# Video Tower: Converts video ID to an embedding vector
video_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary=unique_video_ids_vocab, mask_token=None),
    tf.keras.layers.Embedding(len(unique_video_ids_vocab) + 1, embedding_dimension) # +1 for OOV
])

# The TFRS Retrieval Task
# This task computes the loss and metrics for retrieval.
# It needs a dataset of all candidate video embeddings to compute metrics like FactorizedTopK.
# We map the `videos_ds` (which contains video IDs) through our `video_model` to get their embeddings.
candidate_videos_embeddings_ds = videos_ds.batch(128).map(video_model)

retrieval_task = tfrs.tasks.Retrieval(
    metrics=tfrs.metrics.FactorizedTopK(
        candidates=candidate_videos_embeddings_ds # These are video embeddings
    )
)

# The Two-Tower Recommender Model
class TwoTowerRecommender(tfrs.Model):
    def __init__(self, user_model, video_model, retrieval_task):
        super().__init__()
        self.user_model = user_model
        self.video_model = video_model
        self.retrieval_task = retrieval_task

    def compute_loss(self, features, training=False):
        # Get user embeddings from the user_id feature
        user_embeddings = self.user_model(features["user_id"])
        # Get positive video embeddings from the video_id feature (these are the videos users interacted with)
        positive_video_embeddings = self.video_model(features["video_id"])

        # The retrieval task computes the loss and metrics using user embeddings,
        # positive video embeddings, and the pre-computed candidate video embeddings.
        return self.retrieval_task(
            user_embeddings, positive_video_embeddings, compute_metrics=not training
        )

# Instantiate the model
recommender_model = TwoTowerRecommender(user_model, video_model, retrieval_task)

# --- 4. Training ---
# Configure the optimizer, batch size, and epochs
recommender_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# Prepare the training data: shuffle, batch, and cache
# Each element of `ratings_ds` is a dictionary {"user_id": ..., "video_id": ...}
shuffled_ratings_ds = ratings_ds.shuffle(len(interactions_df), seed=42, reshuffle_each_iteration=False)
batched_ratings_ds = shuffled_ratings_ds.batch(1024) # Use a batch size appropriate for your data/memory
cached_ratings_ds = batched_ratings_ds.cache()

print("\nStarting model training (this might take a moment)...")
# In a real scenario, you'd train for many more epochs and on much larger data.
# For this demo, we'll use a small number of epochs.
try:
    history = recommender_model.fit(cached_ratings_ds, epochs=20, verbose=1) # Set verbose=0 for less output
    print("Model training completed.")
    # You can plot training history.history['factorized_top_k/top_100_categorical_accuracy'] if needed
except Exception as e:
    print(f"An error occurred during training: {e}")
    print("Ensure your TensorFlow and TFRS versions are compatible and data is correctly formatted.")

# --- 5. Generating Recommendations ---
# To serve recommendations, we need an index of all video embeddings.
# TFRS provides a BruteForce layer for small candidate sets.
# For large-scale systems, use ScaNN (Scalable Nearest Neighbors).

print("\nBuilding recommendation index...")
# The BruteForce layer takes user queries (user embeddings) and finds the top K
# closest items (video embeddings) from the candidates it has indexed.
# It needs to be populated with all the candidate video IDs and their corresponding embeddings.
index = tfrs.layers.factorized_top_k.BruteForce(recommender_model.user_model) # Query model is the user_model
index.index_from_dataset(
    # We pass (video_id, video_embedding) pairs.
    # The video_id is what will be returned by the index.
    tf.data.Dataset.zip((
        videos_ds.batch(100), # The video IDs
        videos_ds.batch(100).map(recommender_model.video_model) # The video embeddings
    ))
)
print("Recommendation index built.")

# Get recommendations for a specific user
target_user_id = "user001"
print(f"\nGetting top 5 recommendations for user: {target_user_id}")

# The index takes a tensor of user IDs and returns (scores, video_ids)
try:
    _, recommended_video_ids = index(tf.constant([target_user_id]))

    print(f"Recommended video IDs for {target_user_id}: {recommended_video_ids[0, :5].numpy()}")

    # Map IDs back to titles for better readability
    recommended_titles = [
        videos_df[videos_df["video_id"] == vid.decode()]["video_title"].iloc[0]
        for vid in recommended_video_ids[0, :5].numpy()
    ]
    print(f"Recommended video titles for {target_user_id}: {recommended_titles}")

    print("\nPreviously watched videos by this user (for context):")
    watched_by_user = interactions_df[interactions_df["user_id"] == target_user_id]["video_id"].values
    watched_titles = [videos_df[videos_df["video_id"] == vid]["video_title"].iloc[0] for vid in watched_by_user[:5]]
    print(watched_titles)

except tf.errors.InvalidArgumentError as e:
    print(f"\nError getting recommendations: {e}")
    print("This can happen if the user ID was not in the training vocabulary, "
          "or if the model/index wasn't properly trained/built.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


print("\n--- Video Recommendation Example End ---")
print("Note: Recommendation quality depends heavily on data size, feature richness, model architecture, and tuning.")