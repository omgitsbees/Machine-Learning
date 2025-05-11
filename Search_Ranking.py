import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression # Can be used as a simple pointwise ranker
from sklearn.model_selection import train_test_split
import pandas as pd

# Download necessary NLTK data (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- 1. Query Understanding ---

def preprocess_text(text):
    """
    Cleans and tokenizes text:
    - Lowercases
    - Removes punctuation
    - Removes stopwords
    - Tokenizes
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def analyze_query(query):
    """
    Performs basic query understanding.
    Returns processed tokens and potential intents/entities (simplified).
    """
    processed_tokens = preprocess_text(query)
    
    # Simple intent/entity extraction (example)
    intent = "information_retrieval"
    entities = []
    if "image" in processed_tokens or "picture" in processed_tokens:
        intent = "image_search"
    if "video" in processed_tokens or "movie" in processed_tokens:
        intent = "video_search"
    
    # Extract proper nouns as potential entities (very basic)
    # For more advanced NER, use libraries like SpaCy or NLTK's ne_chunk
    # This is a placeholder for simplicity
    for token in processed_tokens:
        if token.istitle() or token.isupper(): # Simplistic check for proper nouns
             if len(token) > 1: # Avoid single uppercase letters if not acronyms
                entities.append(token)

    return {
        "original_query": query,
        "processed_tokens": processed_tokens,
        "processed_query_str": " ".join(processed_tokens),
        "intent": intent, # Simplified intent
        "entities": list(set(entities)) # Unique entities
    }

# --- 2. Document Corpus & Initial Retrieval (Simplified) ---
# In a real system, this would be a search index (e.g., Elasticsearch, Solr)

documents_data = {
    "doc1": {"id": "doc1", "title": "Introduction to Machine Learning", "text": "Machine learning is a field of computer science that gives computers the ability to learn without being explicitly programmed. This article covers supervised and unsupervised learning.", "category": "education", "popularity": 150},
    "doc2": {"id": "doc2", "title": "Advanced Python Programming", "text": "Explore advanced Python topics like decorators, generators, and metaclasses. Useful for building complex applications.", "category": "technology", "popularity": 200},
    "doc3": {"id": "doc3", "title": "Healthy Cooking Recipes", "text": "Learn to cook healthy and delicious meals. Recipes for breakfast, lunch, and dinner. Focus on fresh ingredients and python.", "category": "lifestyle", "popularity": 300},
    "doc4": {"id": "doc4", "title": "Python for Data Analysis", "text": "This guide explains how to use Python and its libraries like Pandas and NumPy for data analysis and manipulation. Machine learning examples included.", "category": "technology", "popularity": 250},
    "doc5": {"id": "doc5", "title": "Understanding Deep Learning", "text": "Deep learning is a subfield of machine learning concerned with algorithms inspired by the structure and function of the brain called artificial neural networks.", "category": "education", "popularity": 180},
    "doc6": {"id": "doc6", "title": "Travel Guide to Paris", "text": "Discover the best sights in Paris, from the Eiffel Tower to the Louvre Museum. Tips for accommodation and dining.", "category": "travel", "popularity": 220},
    "doc7": {"id": "doc7", "title": "Basics of Python Programming", "text": "A beginner-friendly tutorial on Python programming, covering variables, data types, loops, and functions. Ideal for new programmers.", "category": "technology", "popularity": 100}
}

all_doc_texts = [doc['title'] + " " + doc['text'] for doc_id, doc in documents_data.items()]
all_doc_ids = list(documents_data.keys())

# Global TF-IDF Vectorizer for documents
corpus_vectorizer = TfidfVectorizer(preprocessor=lambda x: " ".join(preprocess_text(x)))
corpus_tfidf_matrix = corpus_vectorizer.fit_transform(all_doc_texts)

def initial_candidate_retrieval(processed_query_str, top_n=10):
    """
    Retrieves initial set of candidate documents based on TF-IDF cosine similarity.
    """
    if not processed_query_str.strip(): # Handle empty processed query
        return []
        
    query_tfidf = corpus_vectorizer.transform([processed_query_str])
    similarities = cosine_similarity(query_tfidf, corpus_tfidf_matrix).flatten()
    
    # Get top N document indices
    top_doc_indices = similarities.argsort()[-top_n:][::-1]
    
    candidate_docs = []
    for i in top_doc_indices:
        if similarities[i] > 0: # Only consider if there's some similarity
            candidate_docs.append({**documents_data[all_doc_ids[i]], "initial_score": similarities[i]})
    return candidate_docs

# --- 3. Feature Engineering for Ranking ---

def extract_features(query_info, document_info):
    """
    Extracts features for a (query, document) pair.
    """
    features = {}
    
    # a. Query-based features
    features["query_length"] = len(query_info["processed_tokens"])
    # features["query_intent"] = query_info["intent"] # Could be one-hot encoded if used

    # b. Document-based features
    features["doc_popularity"] = document_info.get("popularity", 0)
    # features["doc_category"] = document_info.get("category", "unknown") # Could be one-hot encoded

    # c. Query-Document interaction features
    # TF-IDF score (already calculated in initial retrieval, can be passed or recalculated)
    features["tfidf_score"] = document_info.get("initial_score", 0) # Use initial score as a feature

    # Title match: How many query terms are in the document title?
    title_tokens = preprocess_text(document_info["title"])
    common_in_title = len(set(query_info["processed_tokens"]) & set(title_tokens))
    features["title_match_count"] = common_in_title
    features["title_match_ratio"] = common_in_title / len(query_info["processed_tokens"]) if query_info["processed_tokens"] else 0

    # Text match: How many query terms are in the document text?
    # (using processed_query_str for simplicity here, could use tokens for more precision)
    # text_tokens = preprocess_text(document_info["text"]) # Could be pre-processed and stored
    # common_in_text = len(set(query_info["processed_tokens"]) & set(text_tokens))
    # features["text_match_count"] = common_in_text

    # Entity match in title or text
    entities_in_doc = 0
    doc_content_for_entity_check = (document_info["title"] + " " + document_info["text"]).lower()
    for entity in query_info["entities"]:
        if entity.lower() in doc_content_for_entity_check:
            entities_in_doc +=1
    features["entity_match_count"] = entities_in_doc
    
    # Exact phrase match (simplified)
    features["exact_phrase_in_title"] = 1 if query_info["processed_query_str"] in document_info["title"].lower() else 0
    features["exact_phrase_in_text"] = 1 if query_info["processed_query_str"] in document_info["text"].lower() else 0

    return features

# --- 4. Training Data Generation (Simulated) ---
# In a real scenario, this comes from human judgments or click logs.
# Format: (query_info, doc_info, relevance_label)
# Relevance: 0 = not relevant, 1 = somewhat relevant, 2 = highly relevant

simulated_training_data = [
    # Query: "machine learning basics"
    (analyze_query("machine learning basics"), documents_data["doc1"], 2), # Highly relevant
    (analyze_query("machine learning basics"), documents_data["doc5"], 2), # Highly relevant
    (analyze_query("machine learning basics"), documents_data["doc4"], 1), # Somewhat relevant (broader)
    (analyze_query("machine learning basics"), documents_data["doc2"], 0), # Not directly relevant
    (analyze_query("machine learning basics"), documents_data["doc3"], 0), # Not relevant

    # Query: "python programming"
    (analyze_query("python programming"), documents_data["doc2"], 2),
    (analyze_query("python programming"), documents_data["doc7"], 2),
    (analyze_query("python programming"), documents_data["doc4"], 1),
    (analyze_query("python programming"), documents_data["doc3"], 1), # "python" in text
    (analyze_query("python programming"), documents_data["doc1"], 0),

    # Query: "healthy food"
    (analyze_query("healthy food"), documents_data["doc3"], 2),
    (analyze_query("healthy food"), documents_data["doc6"], 0), # Travel guide

    # Query: "Paris travel"
    (analyze_query("Paris travel"), documents_data["doc6"], 2),
    (analyze_query("Paris travel"), documents_data["doc3"], 0),
]

# Create a DataFrame for training
feature_list = []
labels = []
query_ids = [] # For group-aware ranking if using XGBRanker, etc.

for i, (q_info, doc_info, rel) in enumerate(simulated_training_data):
    features = extract_features(q_info, doc_info)
    # Add initial score if not present (e.g. if doc wasn't from retrieval for this specific query)
    if 'tfidf_score' not in features:
        # Quick recalculation for this example
        processed_query_str = q_info["processed_query_str"]
        if processed_query_str.strip():
            doc_text = doc_info['title'] + " " + doc_info['text']
            query_tfidf_temp = corpus_vectorizer.transform([processed_query_str])
            doc_tfidf_temp = corpus_vectorizer.transform([doc_text])
            features['tfidf_score'] = cosine_similarity(query_tfidf_temp, doc_tfidf_temp)[0][0]
        else:
            features['tfidf_score'] = 0

    feature_list.append(features)
    labels.append(rel)
    query_ids.append(f"query_{i//5}") # Simple query grouping for this example data

df_train = pd.DataFrame(feature_list)
df_train['relevance'] = labels
# df_train['query_id'] = query_ids # For LTR models like XGBRanker

# Handle categorical features (if any were kept as strings) - for this example, all are numeric
# df_train = pd.get_dummies(df_train, columns=['doc_category', 'query_intent'], dummy_na=False)

# For simplicity, we'll use a pointwise approach with Logistic Regression.
# Relevance scores (0, 1, 2) can be treated as classes or mapped to binary for simpler models.
# Here, let's make it binary: relevant (1, 2) vs. not relevant (0)
df_train['relevance_binary'] = df_train['relevance'].apply(lambda x: 1 if x > 0 else 0)

# --- 5. Search Ranking Model Training (Pointwise Example) ---
# Pointwise: Predicts a relevance score for each document independently.
# Other approaches: Pairwise (predicts which of two docs is more relevant), Listwise (optimizes the entire list).

# Select features for the model
# Ensure all feature names are present after potential get_dummies
feature_columns = [col for col in df_train.columns if col not in ['relevance', 'relevance_binary', 'query_id']]

X = df_train[feature_columns]
y = df_train['relevance_binary'] # Using binary relevance for Logistic Regression

# Fill NaN values that might have occurred (e.g. if a query_info["processed_tokens"] was empty)
X = X.fillna(0)

if X.empty or y.empty:
    print("Training data is empty. Cannot train model.")
    ranking_model = None
else:
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        ranking_model = LogisticRegression(solver='liblinear', random_state=42) # Simple model
        ranking_model.fit(X_train, y_train)
        
        print("\n--- Ranking Model Training (Simplified Pointwise) ---")
        print(f"Training with features: {feature_columns}")
        if X_test.shape[0] > 0:
            accuracy = ranking_model.score(X_test, y_test)
            print(f"Model Accuracy on test set: {accuracy:.2f}")
        else:
            print("Test set is empty, cannot evaluate accuracy.")

    except ValueError as e:
        print(f"Error during model training: {e}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"X head:\n{X.head()}")
        ranking_model = None


# --- 6. Putting It All Together: Search and Rank ---

def search_and_rank(query_str, top_k=5):
    print(f"\n--- Performing Search for: '{query_str}' ---")
    
    # 1. Understand Query
    query_info = analyze_query(query_str)
    print(f"Query Analysis: {query_info}")
    
    if not query_info["processed_query_str"].strip():
        print("Processed query is empty. Cannot retrieve candidates.")
        return []

    # 2. Initial Candidate Retrieval (e.g., using TF-IDF from a larger index)
    candidate_documents = initial_candidate_retrieval(query_info["processed_query_str"], top_n=20) # Retrieve more to re-rank
    
    if not candidate_documents:
        print("No candidate documents found by initial retrieval.")
        return []
    
    print(f"\nRetrieved {len(candidate_documents)} candidates for re-ranking.")
    # for i, doc in enumerate(candidate_documents[:3]):
    #     print(f"  Top candidate {i+1}: {doc['id']} ({doc['title']}) - Initial Score: {doc.get('initial_score',0):.4f}")


    # 3. Feature Extraction for Candidates
    rerank_features_list = []
    valid_candidates_for_rerank = []
    for doc_data in candidate_documents:
        features = extract_features(query_info, doc_data)
        rerank_features_list.append(features)
        valid_candidates_for_rerank.append(doc_data)
        
    df_rerank = pd.DataFrame(rerank_features_list)
    df_rerank = df_rerank.fillna(0) # Ensure no NaNs

    # Align columns with training data (important!)
    for col in feature_columns: # feature_columns defined during training
        if col not in df_rerank.columns:
            df_rerank[col] = 0 # Add missing columns with default value
    df_rerank = df_rerank[feature_columns] # Ensure same order and feature set

    # 4. Re-ranking with the ML Model
    if ranking_model and not df_rerank.empty:
        # Predict probability of relevance (class 1)
        relevance_probabilities = ranking_model.predict_proba(df_rerank)[:, 1]
        
        for i, doc in enumerate(valid_candidates_for_rerank):
            doc['ml_score'] = relevance_probabilities[i]
            
        # Sort documents by the ML model's score
        ranked_documents = sorted(valid_candidates_for_rerank, key=lambda x: x['ml_score'], reverse=True)
        print("\nRe-ranked documents (top ML scores):")
    else:
        print("\nRanking model not available or no features to rank. Using initial TF-IDF scores.")
        # Fallback to initial TF-IDF sorting if model failed or no features
        ranked_documents = sorted(valid_candidates_for_rerank, key=lambda x: x.get('initial_score', 0), reverse=True)

    # 5. Return top K results
    final_results = ranked_documents[:top_k]
    
    print("\n--- Final Top Results ---")
    for i, doc in enumerate(final_results):
        print(f"{i+1}. ID: {doc['id']}, Title: '{doc['title']}'")
        print(f"   Initial TF-IDF Score: {doc.get('initial_score', 0):.4f}")
        if 'ml_score' in doc:
            print(f"   ML Relevance Score: {doc['ml_score']:.4f}")
        # print(f"   Category: {doc['category']}, Popularity: {doc['popularity']}")
        # print(f"   Extracted Features: {extract_features(query_info, doc)}") # For debugging
    
    return final_results


# --- Example Usage ---
if __name__ == "__main__":
    # Test Query Understanding
    print("--- Testing Query Understanding ---")
    test_q = "latest news on python machine learning algorithms"
    analyzed = analyze_query(test_q)
    print(f"Original: {analyzed['original_query']}")
    print(f"Processed Tokens: {analyzed['processed_tokens']}")
    print(f"Processed String: {analyzed['processed_query_str']}")
    print(f"Intent (simplified): {analyzed['intent']}")
    print(f"Entities (simplified): {analyzed['entities']}")
    
    print("\n--- Testing Feature Extraction (Example) ---")
    sample_doc_for_features = documents_data["doc1"]
    sample_query_info = analyze_query("machine learning introduction")
    features_ex = extract_features(sample_query_info, sample_doc_for_features)
    print(f"Features for query '{sample_query_info['original_query']}' and doc '{sample_doc_for_features['id']}':")
    for k, v in features_ex.items():
        print(f"  {k}: {v}")

    # Test Search and Rank
    if ranking_model is not None:
        search_results1 = search_and_rank("basics of machine learning", top_k=3)
        search_results2 = search_and_rank("python programming tutorial", top_k=3)
        search_results3 = search_and_rank("delicious healthy recipes for dinner", top_k=2)
        search_results4 = search_and_rank("Paris", top_k=2) # Test short query / entity
        search_results5 = search_and_rank("  ", top_k=2) # Test empty/whitespace query
    else:
        print("\nSkipping search_and_rank examples as the ranking model was not trained.")

    # Example of how features for training were created (for one instance)
    # q_info_train_ex, doc_info_train_ex, _ = simulated_training_data[0]
    # print(f"\nExample Training Features for query '{q_info_train_ex['original_query']}' and doc '{doc_info_train_ex['id']}':")
    # train_features_ex_df = pd.DataFrame([extract_features(q_info_train_ex, doc_info_train_ex)])
    # print(train_features_ex_df.T)

    # Note on XGBRanker (a more advanced LTR model):
    # If you were to use XGBoost's XGBRanker, you would typically:
    # 1. Prepare data with query_id (group) information.
    #    df_train['query_id'] = query_ids # As prepared earlier
    #    groups = df_train.groupby('query_id').size().to_numpy()
    # 2. Use relevance scores directly (0, 1, 2) instead of binary.
    #    y_ltr = df_train['relevance']
    # 3. Train the XGBRanker:
    #    import xgboost as xgb
    #    ltr_model = xgb.XGBRanker(
    #        objective='rank:ndcg', # or 'rank:map', 'rank:pairwise'
    #        learning_rate=0.1,
    #        n_estimators=100,
    #        random_state=42
    #    )
    #    ltr_model.fit(X_train_ltr, y_train_ltr, group=train_groups, eval_set=[(X_test_ltr, y_test_ltr)], eval_group=[test_groups])
    # 4. Predict scores:
    #    scores = ltr_model.predict(df_rerank_ltr) # df_rerank_ltr would be prepared similarly
    # This requires more careful data preparation for groups and handling of relevance scores.