import os
import pandas as pd
import re
from tqdm import tqdm

RAW_RT_PATH = "data/raw/rt/rotten_tomatoes_reviews.csv"
RT_DIR = "data/raw/rt"
IMDB_DIR = "data/raw/imdb"
FINAL_OUTPUT = "data/processed/preprocessed.csv"


def ensure_dirs():
    os.makedirs(os.path.join(RT_DIR, "pos"), exist_ok=True)
    os.makedirs(os.path.join(RT_DIR, "neg"), exist_ok=True)
    os.makedirs(os.path.join(IMDB_DIR, "pos"), exist_ok=True)
    os.makedirs(os.path.join(IMDB_DIR, "neg"), exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


def label_sentiment(rating):
    if rating >= 4.0:
        return "pos"
    elif rating <= 2.0:
        return "neg"
    return None


def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return text


def load_rt_reviews():
    print("\n Loading Rotten Tomatoes dataset...")
    df = pd.read_csv(RAW_RT_PATH)
    if "Review" not in df.columns or "Score" not in df.columns:
        raise ValueError("CSV must contain 'Review' and 'Score' columns from Kaggle version.")

    df["sentiment"] = df["Score"].apply(label_sentiment)
    df = df[df["sentiment"].notnull()].dropna(subset=["Review"])
    df = df[["Review", "sentiment"]].rename(columns={"Review": "text"})
    df["source"] = "rt"
    print(f" Rotten Tomatoes: {len(df)} reviews")
    return df


def load_imdb_reviews():
    print("\n Loading IMDb dataset...")
    data = []
    for sentiment in ["pos", "neg"]:
        path = os.path.join(IMDB_DIR, sentiment)
        label = sentiment
        files = os.listdir(path)
        print(f"Processing {len(files)} {sentiment} files...")
        for file in tqdm(files, desc=f"IMDb {sentiment}"):
            full_path = os.path.join(path, file)
            with open(full_path, "r", encoding="utf-8") as f:
                review = f.read()
                data.append({"text": review, "sentiment": label, "source": "imdb"})
    df = pd.DataFrame(data)
    print(f" IMDb: {len(df)} reviews")
    return df


def combine_and_save():
    df_rt = load_rt_reviews()
    df_imdb = load_imdb_reviews()
    df_all = pd.concat([df_rt, df_imdb], ignore_index=True)
    df_all["text"] = df_all["text"].apply(clean_text)
    df_all.to_csv(FINAL_OUTPUT, index=False)
    print(f"\n Saved combined dataset to {FINAL_OUTPUT} ({len(df_all)} total reviews)")


if __name__ == "__main__":
    ensure_dirs()
    combine_and_save()
