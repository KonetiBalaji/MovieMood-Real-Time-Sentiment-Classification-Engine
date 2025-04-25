# scripts/infer.py - Step 5: Inference Pipeline

import joblib
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths
VECTORIZER_PATH = "models/vectorizer.pkl"
MODEL_PATH = "models/randomforest_model.pkl"

# Load model and vectorizer
if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MODEL_PATH):
    print("❌ Model or vectorizer not found. Run train.py first.")
    sys.exit(1)

print("📦 Loading model and vectorizer...")
vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

# Input review (manual or CLI)
if len(sys.argv) > 1:
    text_input = " ".join(sys.argv[1:])
else:
    text_input = input("📝 Enter a movie review: ")

# Vectorize
X_input = vectorizer.transform([text_input])

# Predict
pred = model.predict(X_input)[0]
pred_proba = model.predict_proba(X_input)[0]

label = "Positive" if pred == 1 else "Negative"
print(f"\n📣 Sentiment: {label}")
print(f"🔍 Confidence - Negative: {pred_proba[0]:.4f}, Positive: {pred_proba[1]:.4f}")
