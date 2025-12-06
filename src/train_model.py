# src/train_model.py
import os
import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)

# Load dataset (replace with real dataset)
fake_path = os.path.join("..", "data", "Fake.csv")
true_path = os.path.join("..", "data", "True.csv")

fake = pd.read_csv(fake_path)
true = pd.read_csv(true_path)

# Combine text columns if needed
def get_text(df):
    if "text" in df.columns:
        return df["text"].astype(str)
    elif "title" in df.columns and "text" in df.columns:
        return df["title"].astype(str) + " " + df["text"].astype(str)
    else:
        return df.astype(str).agg(" ".join, axis=1)

fake_texts = get_text(fake)
true_texts = get_text(true)

# Create DataFrame
df = pd.DataFrame({
    "text": list(fake_texts) + list(true_texts),
    "label": ["FAKE"]*len(fake_texts) + ["REAL"]*len(true_texts)
})

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Clean text
df["clean"] = df["text"].apply(clean_text)

X = df["clean"]
y = df["label"]

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF + Logistic Regression pipeline
vectorizer = TfidfVectorizer(stop_words="english", max_features=30000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

clf = LogisticRegression(max_iter=2000)
clf.fit(X_train_vec, y_train)

# Evaluate
y_pred = clf.predict(X_val_vec)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

# Save pipeline
os.makedirs(os.path.join("..","models"), exist_ok=True)
pickle.dump((vectorizer, clf), open(os.path.join("..","models","model.pkl"), "wb"))
print("Saved pipeline to ../models/model.pkl")
