import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib
from preprocess import preprocess_text

def load_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            text, label = line.strip().split(';')
            data.append(preprocess_text(text))
            labels.append(label)
    return data, labels


train_data, train_labels = load_data(r"C:\Users\chand\Documents\SENtiment Analysis\archive\train.txt")
val_data, val_labels = load_data(r"C:\Users\chand\Documents\SENtiment Analysis\archive\val.txt")
test_data, test_labels = load_data(r"C:\Users\chand\Documents\SENtiment Analysis\archive\test.txt")

all_data = train_data + val_data + test_data
all_labels = train_labels + val_labels + test_labels


X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)


joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("Model and vectorizer saved successfully.")