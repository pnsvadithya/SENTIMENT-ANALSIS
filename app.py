import streamlit as st
import joblib
import numpy as np
from preprocess import preprocess_text

# Load your trained model and vectorizer
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

def predict_sentiment(text):
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    return prediction, probabilities, processed_text, text_vector

# Streamlit app
st.title('Sentiment Analysis')

# Text input
user_input = st.text_area("Enter your text here:")

if st.button('Analyze Sentiment'):
    if user_input:
        sentiment, probabilities, processed_text, text_vector = predict_sentiment(user_input)
        st.write(f"Original Text: {user_input}")
        st.write(f"Processed Text: {processed_text}")
        st.write(f"Predicted Sentiment: {sentiment}")
        st.write("Class Probabilities:")
        for i, prob in enumerate(probabilities):
            st.write(f"  Class {i}: {prob:.4f}")
    else:
        st.write("Please enter some text to analyze.")

# Display model information
st.write("Model Information:")
st.write(f"Model type: {type(model).__name__}")
st.write(f"Number of classes: {len(model.classes_)}")
st.write(f"Classes: {model.classes_}")

