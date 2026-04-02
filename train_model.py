import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake['label'] = 'FAKE'
true['label'] = 'REAL'

# Combine datasets
data = pd.concat([fake, true])

# Combine text columns
data['content'] = data['title'] + " " + data['text']

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

data['content'] = data['content'].apply(clean_text)

# Features & labels
X = data['content']
y = data['label']

# Convert text → numbers
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

# Save files
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model created successfully!")