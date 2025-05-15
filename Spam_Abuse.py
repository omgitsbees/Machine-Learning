import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Download necessary NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocesses the input text by performing the following steps:
    1.  Removes URLs.
    2.  Removes mentions (@username).
    3.  Removes hashtags.
    4.  Removes special characters and punctuation.
    5.  Converts text to lowercase.
    6.  Removes stop words.
    7.  Tokenizes the text.

    Args:
        text (str): The text to preprocess.

    Returns:
        list: A list of cleaned tokens.
    """
    if not isinstance(text, str):
        return []  # Return empty list for non-string input

    # 1. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 2. Remove mentions
    text = re.sub(r'@\w+', '', text)
    # 3. Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # 4. Remove special characters and punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 5. Convert to lowercase
    text = text.lower()
    # 6. Tokenize the text
    tokens = word_tokenize(text)
    # 7. Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def create_model(data_file='spam_data.csv', model_file='spam_detection_model.joblib'):
    """
    Creates and trains a machine learning model for spam detection.

    Args:
        data_file (str, optional): The path to the CSV data file. Defaults to 'spam_data.csv'.
        model_file (str, optional): The path to save the trained model. Defaults to 'spam_detection_model.joblib'.

    Returns:
        tuple: (vectorizer, model) if training is successful, (None, None) otherwise.
               vectorizer:  TfidfVectorizer instance.
               model: Trained LogisticRegression model.
    """
    import pandas as pd # Import pandas inside the function to avoid issues if it's not installed in some environments

    try:
        # Load the dataset
        data = pd.read_csv(data_file, encoding='latin-1')  # Handle potential encoding issues
        # Rename columns if necessary
        data = data.rename(columns={'v1': 'label', 'v2': 'text'}, errors='ignore')
        # Drop unnecessary columns
        data = data[['label', 'text']]
        # Preprocess the text data
        data['tokens'] = data['text'].apply(preprocess_text)
        data['text'] = data['tokens'].apply(lambda x: ' '.join(x)) #convert back to string

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            data['text'], data['label'], test_size=0.2, random_state=42
        )

        # Vectorize the text data using TF-IDF
        vectorizer = TfidfVectorizer()
        X_train_vectors = vectorizer.fit_transform(X_train)
        X_test_vectors = vectorizer.transform(X_test)

        # Create and train the Logistic Regression model
        model = LogisticRegression(solver='liblinear')  # Use 'liblinear' for smaller datasets
        model.fit(X_train_vectors, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test_vectors)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')
        print(classification_report(y_test, y_pred))
        print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')

        # Save the trained model and vectorizer
        joblib.dump((vectorizer, model), model_file)
        print(f'Model saved to {model_file}')
        return vectorizer, model

    except Exception as e:
        print(f'Error: {e}')
        return None, None

def load_model(model_file='spam_detection_model.joblib'):
    """
    Loads the trained machine learning model and vectorizer.

    Args:
        model_file (str, optional): The path to the saved model file. Defaults to 'spam_detection_model.joblib'.

    Returns:
        tuple: (vectorizer, model) if loading is successful, (None, None) otherwise.
               vectorizer:  TfidfVectorizer instance.
               model: Trained LogisticRegression model.
    """
    try:
        # Load the trained model and vectorizer
        vectorizer, model = joblib.load(model_file)
        print(f'Model loaded from {model_file}')
        return vectorizer, model
    except Exception as e:
        print(f'Error: {e}')
        return None, None

def predict_spam(text, vectorizer, model):
    """
    Predicts whether the given text is spam or not using the loaded model.

    Args:
        text (str): The text to classify.
        vectorizer:  TfidfVectorizer instance.
        model: Trained LogisticRegression model.

    Returns:
        str: 'spam' or 'ham' (not spam).  Returns "Error" on error.
    """
    if not isinstance(text, str):
        return "Error"

    if vectorizer is None or model is None:
        return "Error"  # Indicate that the model hasn't been loaded

    try:
        # Preprocess the input text
        tokens = preprocess_text(text)
        text_vector = vectorizer.transform([' '.join(tokens)])  # Vectorize the preprocessed text
        # Predict the class of the text
        prediction = model.predict(text_vector)[0]
        return prediction
    except Exception as e:
        print(f'Error: {e}')
        return "Error"

def main():
    """
    Main function to run the spam detection model.  This function:
    1.  Attempts to create a model.
    2.  If successful, loads the model and predicts on sample text.
    3.  If create model fails, it attempts to load a pre-existing model
    """
    # Create and train the model (or load a pre-trained one)
    vectorizer, model = create_model()

    if vectorizer is None or model is None:
        print("Failed to create model. Attempting to load...")
        vectorizer, model = load_model()
        if vectorizer is None or model is None:
            print("Failed to load model.  Exiting.")
            return

    if vectorizer and model:
        # Example usage:
        test_texts = [
            "WINNER!! As a valued network customer you have been selected to receive a å£900 prize reward! claim call 09061701461.",
            "Hi honey how r u? I'm doing fine and miss u",
            "Get a free iPhone 15 Pro Max now!!",
            "Please subscribe to my channel for more videos",
            "This is a great video!  Thanks for sharing.",
            "URGENT! You have won a 1 week FREE cruise!",
            " জমি বিক্রয় করা হবে, যোগাযোগ করুন : ৯৮XXXXXXX", # Bengali for "Land for sale, contact: 98XXXXXXX"
            "আপনার ভিডিওটি খুব ভালো লেগেছে।", # Bengali for "I liked your video very much."
            "超级优惠！点击链接领取你的专属红包！",  # Chinese: "Super discount! Click the link to claim your exclusive red envelope!"
            "关注我的频道，获取更多精彩内容！", # Chinese: "Follow my channel for more exciting content!"
            "Visit our website to learn more.",
            "Call this number for amazing offers!",
            "How to make money online fast!",
            "Check out this amazing new product!",
            "I'm giving away free stuff!",
            "Subscribe to my channel for more updates",
            "Nice video, thanks for sharing!",
            "Please like and subscribe!",
            "This is very helpful, thank you!",
            "I agree with you."
        ]

        for text in test_texts:
            prediction = predict_spam(text, vectorizer, model)
            print(f'Text: "{text}"\nPrediction: {prediction}\n')

if __name__ == '__main__':
    main()
