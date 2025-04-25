import random
from transformers import pipeline
from googletrans import Translator
from textblob import TextBlob
import pyttsx3
import speech_recognition as sr

# Load pre-trained NLP models
def load_nlp_models():
    """
    Load pre-trained models for intent recognition and sentiment analysis.
    """
    print("Loading NLP models...")
    intent_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_model = pipeline("sentiment-analysis")
    return intent_model, sentiment_model

# Define FAQ database
def get_faq_database():
    """
    Define a simple FAQ database with questions and answers.
    """
    return {
        "What is my account balance?": "You can check your account balance in the 'My Account' section of our app.",
        "How do I reset my password?": "To reset your password, click on 'Forgot Password' on the login page.",
        "What are your working hours?": "Our customer support is available 24/7.",
        "How do I apply for a loan?": "You can apply for a loan through our app or website under the 'Loans' section.",
        "What is the interest rate for savings accounts?": "Our savings accounts offer an interest rate of 3.5% per annum."
    }

# Match user query to FAQ
def match_query_to_faq(query, faq_database):
    """
    Match the user's query to the closest FAQ question.
    """
    for question, answer in faq_database.items():
        if query.lower() in question.lower():
            return answer
    return None

# Translate query for multi-language support
def translate_query(query, target_language="en"):
    """
    Translate the user's query to the target language (default: English).
    """
    translator = Translator()
    translated_query = translator.translate(query, dest=target_language).text
    return translated_query

# Detect sentiment of the query
def detect_sentiment(query, sentiment_model):
    """
    Detect the sentiment of the user's query.
    """
    sentiment = sentiment_model(query)[0]
    return sentiment['label'], sentiment['score']

# Generate response for complex queries
def generate_response(query, intent_model):
    """
    Generate a response for complex queries using the intent recognition model.
    """
    response = intent_model(query)
    return f"I'm not sure about that. Here's what I understood: {response[0]['label']}."

# Voice support: Convert text to speech
def text_to_speech(text):
    """
    Convert chatbot responses to speech.
    """
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Voice support: Convert speech to text
def speech_to_text():
    """
    Convert user speech to text.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        print(f"You said: {query}")
        return query
    except sr.UnknownValueError:
        return "Sorry, I didn't catch that."
    except sr.RequestError:
        return "Sorry, there was an issue with the speech recognition service."

# Context awareness
class ChatContext:
    """
    Maintain context for the chatbot to handle follow-up questions.
    """
    def __init__(self):
        self.last_query = None
        self.last_response = None

    def update_context(self, query, response):
        self.last_query = query
        self.last_response = response

# Chatbot logic
def chatbot():
    """
    Main chatbot logic for handling user queries.
    """
    print("Welcome to the Customer Support Chatbot!")
    print("Type 'exit' to end the chat.\n")

    intent_model, sentiment_model = load_nlp_models()
    faq_database = get_faq_database()
    context = ChatContext()

    while True:
        # Voice or text input
        mode = input("Type 'voice' for voice input or 'text' for text input: ").strip().lower()
        if mode == "voice":
            user_query = speech_to_text()
        else:
            user_query = input("You: ")

        if user_query.lower() == "exit":
            print("Chatbot: Thank you for using our service. Goodbye!")
            break

        # Translate query to English for processing
        translated_query = translate_query(user_query)

        # Check if the query matches an FAQ
        faq_response = match_query_to_faq(translated_query, faq_database)
        if faq_response:
            response = faq_response
        else:
            # Generate a response for complex queries
            response = generate_response(translated_query, intent_model)

        # Detect sentiment
        sentiment, confidence = detect_sentiment(translated_query, sentiment_model)
        print(f"Chatbot: Sentiment detected as {sentiment} (confidence: {confidence:.2f})")

        # Provide response
        print(f"Chatbot: {response}")
        text_to_speech(response)

        # Update context
        context.update_context(user_query, response)

# Main workflow
if __name__ == "__main__":
    chatbot()