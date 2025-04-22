import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Load and preprocess data
def load_data(user_filepath, product_filepath):
    """
    Load user-product interaction data and product metadata.
    """
    user_data = pd.read_csv(user_filepath)
    product_data = pd.read_csv(product_filepath)
    return user_data, product_data

# Normalize user-product interaction data
def normalize_data(data):
    """
    Normalize user-product interaction data for collaborative filtering.
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return pd.DataFrame(normalized_data, columns=data.columns, index=data.index)

# Content-based filtering
def content_based_recommendations(product_id, product_data, top_n=5):
    """
    Generate recommendations based on product similarity using content-based filtering.
    """
    # Use TF-IDF to compute similarity based on product descriptions
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(product_data['Description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get similarity scores for the given product
    product_index = product_data[product_data['ProductID'] == product_id].index[0]
    similarity_scores = list(enumerate(cosine_sim[product_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the top N most similar products
    similar_products = [product_data.iloc[i[0]]['ProductID'] for i in similarity_scores[1:top_n+1]]
    return similar_products

# Hybrid model
def hybrid_recommendations(user_id, user_data, product_data, top_n=5):
    """
    Combine collaborative filtering and content-based filtering for hybrid recommendations.
    """
    # Collaborative filtering
    similarity_matrix = cosine_similarity(user_data)
    similarity_df = pd.DataFrame(similarity_matrix, index=user_data.index, columns=user_data.index)
    similar_users = similarity_df[user_id].sort_values(ascending=False).index[1:]
    collaborative_recommendations = user_data.loc[similar_users].mean(axis=0).sort_values(ascending=False)

    # Content-based filtering
    content_recommendations = []
    for product_id in collaborative_recommendations.index[:top_n]:
        content_recommendations.extend(content_based_recommendations(product_id, product_data, top_n=1))

    # Combine and rank recommendations
    hybrid_recommendations = list(set(collaborative_recommendations.index[:top_n]).union(content_recommendations))
    return hybrid_recommendations[:top_n]

# Explainability
def explain_recommendations(user_id, recommendations, user_data, product_data):
    """
    Provide explanations for recommendations.
    """
    explanations = []
    for product_id in recommendations:
        similar_users = user_data[user_data[product_id] > 0].index.tolist()
        similar_products = content_based_recommendations(product_id, product_data, top_n=1)
        explanations.append({
            'ProductID': product_id,
            'Reason': f"Recommended because similar users ({similar_users}) interacted with it, "
                      f"and it is similar to {similar_products}."
        })
    return explanations

# Real-time recommendations
def real_time_recommendations(user_id, user_data, product_data, top_n=5):
    """
    Generate real-time recommendations for a user.
    """
    recommendations = hybrid_recommendations(user_id, user_data, product_data, top_n)
    explanations = explain_recommendations(user_id, recommendations, user_data, product_data)
    return recommendations, explanations

# Main workflow
def main():
    # Replace with the paths to your datasets
    user_filepath = 'user_data.csv'
    product_filepath = 'product_data.csv'

    # Load and preprocess data
    user_data, product_data = load_data(user_filepath, product_filepath)
    user_data = user_data.set_index('UserID')  # Set UserID as the index
    normalized_user_data = normalize_data(user_data)

    # Generate real-time recommendations for a specific user
    user_id = 1  # Replace with the ID of the user you want to recommend for
    recommendations, explanations = real_time_recommendations(user_id, normalized_user_data, product_data)

    print(f"Top Recommendations for User {user_id}:")
    for rec in recommendations:
        print(f"- ProductID: {rec}")

    print("\nExplanations:")
    for explanation in explanations:
        print(f"- {explanation}")

if __name__ == "__main__":
    main()