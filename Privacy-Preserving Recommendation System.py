import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import tkinter as tk
from tkinter import ttk, messagebox

# Step 1: Load Dataset
data = {
    'User_ID': [1, 1, 2, 2, 3, 3, 4],
    'Item_ID': [101, 102, 101, 103, 102, 104, 101],
    'Rating': [5, 4, 4, 5, 3, 4, 2]
}
item_metadata = {
    'Item_ID': [101, 102, 103, 104],
    'Description': ['Action movie', 'Romantic comedy', 'Sci-fi thriller', 'Drama']
}
df = pd.DataFrame(data)
item_df = pd.DataFrame(item_metadata)

# Step 2: Create User-Item Matrix
user_item_matrix = df.pivot_table(index='User_ID', columns='Item_ID', values='Rating').fillna(0)

# Step 3: Compute Similarity
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Step 4: Content-Based Filtering
tfidf = TfidfVectorizer()
item_profiles = tfidf.fit_transform(item_df['Description'])
item_similarity = cosine_similarity(item_profiles)

# Step 5: Hybrid Recommendation System
def recommend_items(user_id, user_item_matrix, user_similarity_df, item_similarity, top_n=2):
    user_ratings = user_item_matrix.loc[user_id]
    similar_users = user_similarity_df[user_id]
    weighted_ratings = user_item_matrix.T.dot(similar_users) / similar_users.sum()
    
    # Map Item_IDs to indices in the item_similarity matrix
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_df['Item_ID'])}
    index_to_item_id = {idx: item_id for item_id, idx in item_id_to_index.items()}
    
    # Ensure all Item_IDs in user_item_matrix are in the mapping
    for item_id in user_item_matrix.columns:
        if item_id not in item_id_to_index:
            item_id_to_index[item_id] = len(item_id_to_index)
            index_to_item_id[len(index_to_item_id)] = item_id
    
    # Get indices of rated items
    rated_items = [item_id_to_index[item_id] for item_id in user_ratings[user_ratings > 0].index]
    
    # Calculate content-based scores
    content_scores = item_similarity[rated_items].mean(axis=0)
    
    # Combine collaborative and content-based scores
    hybrid_scores = weighted_ratings + pd.Series(content_scores, index=user_item_matrix.columns)
    recommendations = hybrid_scores[user_ratings == 0].sort_values(ascending=False).head(top_n)
    
    # Return recommended Item_IDs
    return [index_to_item_id[idx] for idx in recommendations.index]

# Step 6: Enhanced UI
def display_recommendations():
    try:
        user_id = int(user_dropdown.get())
        if user_id not in user_item_matrix.index:
            messagebox.showerror("Error", "User ID not found!")
            return
        recommendations = recommend_items(user_id, user_item_matrix, user_similarity_df, item_similarity)
        if not recommendations:
            messagebox.showinfo("Recommendations", "No recommendations available for this user.")
            return
        
        # Display recommended items with descriptions
        result_text = "Recommended Items:\n"
        for item_id in recommendations:
            item_desc = item_df.loc[item_df['Item_ID'] == item_id, 'Description'].values[0]
            result_text += f"- Item {item_id}: {item_desc}\n"
        messagebox.showinfo("Recommendations", result_text)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create a Tkinter UI
root = tk.Tk()
root.title("Enhanced Recommendation System")

# Dropdown for user selection
tk.Label(root, text="Select User ID:").grid(row=0, column=0, padx=10, pady=10)
user_dropdown = ttk.Combobox(root, values=list(user_item_matrix.index))
user_dropdown.grid(row=0, column=1, padx=10, pady=10)
user_dropdown.set("Select User")

# Button to get recommendations
recommend_button = tk.Button(root, text="Get Recommendations", command=display_recommendations)
recommend_button.grid(row=1, column=0, columnspan=2, pady=10)

root.mainloop()