import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download("punkt")
# Load dataset (FAQ dataset as CSV)h
faq_data = {
    "Question": [
        "What is AI?",
        "What is machine learning?",
        "What is deep learning?",
        "What is NLP?",
        "Who created Python?"
        
        "What is the capital of France?",
        "How does a chatbot work?"
        "What is Overfitting, and How Can You Avoid It?"
    ],
    "Answer": [
        "AI stands for Artificial Intelligence.",
        "Machine Learning is a subset of AI that enables computers to learn from data.",
        "Deep Learning is a specialized form of ML using neural networks.",
        "NLP stands for Natural Language Processing, a field of AI that helps machines understand human language.",
        "Python was created by Guido van Rossum in 1991.",
        "The capital of France is Paris.",
        "A chatbot uses NLP and AI to understand user input and respond accordingly."
        "Regularization. It involves a cost term for the features involved with the objective function , Cross-validation methods like k-folds can also be used"
    ]
}
df = pd.DataFrame(faq_data)
# Convert questions and answers to lists
questions = df["Question"].tolist()
answers = df["Answer"].tolist()
# Precompute TF-IDF matrix for the dataset
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)
# Preprocessing function to clean input text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    return text
# Function to get the best response
def get_response(user_input):
    user_input = preprocess_text(user_input)
    user_vector = vectorizer.transform([user_input])  # Convert input to TF-IDF vector
    scores = cosine_similarity(user_vector, tfidf_matrix).flatten()  # Compute similarity scores

    best_match_index = scores.argmax()
    if scores[best_match_index] > 0.3:  # Threshold for similarity
        return answers[best_match_index]

    return "Sorry, I don't have an answer for that."
# Chat function
def chat():
    print("Chatbot: Hello! Type 'exit' to stop chatting.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        response = get_response(user_input)
        print("Chatbot:", response)
# Run chatbot
chat()