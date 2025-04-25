# scripts/train.py - Step 4.5 Elite: RandomForest + SMOTETomek + Class-Specific F1 & PR

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, f1_score, precision_recall_curve
from imblearn.combine import SMOTETomek

# Paths
DATA_PATH = "data/processed/preprocessed.csv"
VECTORIZER_PATH = "models/vectorizer.pkl"
MODEL_PATH = "models/randomforest_model.pkl"
METRICS_PATH = "outputs/metrics.txt"
CONF_MATRIX_PATH = "outputs/confusion_matrix.png"
PREC_RECALL_PATH = "outputs/precision_recall_curve_neg.png"

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load Data
df = pd.read_csv(DATA_PATH)
X = df["text"]
y = df["sentiment"].map({"pos": 1, "neg": 0})

# Split
df_train, df_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF Vectorizer
print("🔄 Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english')
X_train = vectorizer.fit_transform(df_train)
X_test = vectorizer.transform(df_test)

# Resample: SMOTETomek
print("🔁 Applying SMOTETomek to balance classes...")
resampler = SMOTETomek(random_state=42)
X_train, y_train = resampler.fit_resample(X_train, y_train)

# Model: Random Forest
print("🌲 Training Random Forest Classifier with class_weight={0: 10, 1: 1}...")
model = RandomForestClassifier(n_estimators=100, class_weight={0: 10, 1: 1}, random_state=42)
model.fit(X_train, y_train)

# Predict
print("📊 Evaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Metrics
report = classification_report(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)
f1_neg = f1_score(y_test, y_pred, pos_label=0)
conf_mat = confusion_matrix(y_test, y_pred)

# Save Metrics
with open(METRICS_PATH, "w") as f:
    f.write(report)
    f.write(f"\nF2 Score: {f2:.4f}\n")
    f.write(f"F1 Score (Negative Class): {f1_neg:.4f}\n")
print("✅ Metrics + F2 + F1-neg saved to", METRICS_PATH)

# Confusion Matrix Plot
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["neg", "pos"], yticklabels=["neg", "pos"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH)
print("✅ Confusion matrix saved to", CONF_MATRIX_PATH)

# Precision-Recall Curve (Negative class focus)
probs_neg = y_pred_proba[:, 0]
precision_neg, recall_neg, _ = precision_recall_curve(y_test, probs_neg, pos_label=0)
plt.figure()
plt.plot(recall_neg, precision_neg, marker='.')
plt.xlabel("Recall (neg)")
plt.ylabel("Precision (neg)")
plt.title("Precision-Recall Curve - Negative Class")
plt.savefig(PREC_RECALL_PATH)
print("✅ PR curve for negative class saved to", PREC_RECALL_PATH)

# Save model/vectorizer
joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(model, MODEL_PATH)
print("💾 RandomForest model and vectorizer saved.")