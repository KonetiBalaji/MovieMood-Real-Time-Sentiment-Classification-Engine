# scripts/train_bert.py - Step 6: Fine-Tuning DistilBERT for Sentiment Classification

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import torch

# Paths
DATA_PATH = "data/processed/preprocessed.csv"
MODEL_DIR = "models/bert_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "sentiment"])
df = df[df["sentiment"].isin(["pos", "neg"])]
df["label"] = df["sentiment"].map({"neg": 0, "pos": 1})

# Split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# Convert to Hugging Face Dataset
train_ds = Dataset.from_pandas(df_train[["text", "label"]])
test_ds = Dataset.from_pandas(df_test[["text", "label"]])

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

train_ds = train_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

# Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training args
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"{MODEL_DIR}/logs",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer
def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions), axis=1)
    labels = torch.tensor(p.label_ids)
    acc = (preds == labels).float().mean().item()
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save final model
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print("✅ DistilBERT model and tokenizer saved to", MODEL_DIR)
