import ast
import os
import numpy as np
import pandas as pd
import gensim.downloader as api
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from tqdm import tqdm

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "love", "loved",
    "best", "fantastic", "beautiful", "superb", "perfect", "enjoy", "enjoyed",
    "brilliant", "awesome", "favorite", "fun", "happy", "nice",
}
NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "worst", "hate", "hated", "boring", "poor",
    "waste", "worse", "horrible", "disappointing", "stupid", "dull",
    "annoying", "ridiculous", "fail", "failed", "ugly", "lame",
}
NEGATION_WORDS = {"not", "no", "never", "nor", "neither"}

GLOVE_CACHE_PATH = "glove_cache.kv"


def load_glove():
    if os.path.exists(GLOVE_CACHE_PATH):
        print(f"Loading GloVe from cache ({GLOVE_CACHE_PATH})...")
        return KeyedVectors.load(GLOVE_CACHE_PATH)

    print("First run -- downloading GloVe (one time only)...")
    model = api.load("glove-wiki-gigaword-300")
    model.save(GLOVE_CACHE_PATH)
    print(f"Saved to '{GLOVE_CACHE_PATH}' -- future runs will be fast.")
    return model


def sentence_to_vector(tokens, model, dim=300):
    """
    Average GloVe vector. Words right after a negation get their
    vector flipped so 'not bad' ≈ 'good'.
    """
    vectors, weights = [], []
    after_neg = False
    for token in tokens:
        if token in NEGATION_WORDS:
            after_neg = True
            continue
        if token in model:
            vectors.append(-model[token] if after_neg else model[token])
            weights.append(1.0)
        after_neg = False
    if not vectors:
        return np.zeros(dim, dtype=np.float32)
    return np.average(np.asarray(vectors), axis=0,
                      weights=np.asarray(weights)).astype(np.float32)


def extra_features(tokens):
    """Eight cheap, hand-crafted sentiment features."""
    n = max(len(tokens), 1)
    pos  = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg  = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    negs = sum(1 for t in tokens if t in NEGATION_WORDS)
    neg_pos = neg_neg = 0
    flipped = False
    for t in tokens:
        if t in NEGATION_WORDS:
            flipped = True
            continue
        if flipped and t in POSITIVE_WORDS: neg_pos += 1
        if flipped and t in NEGATIVE_WORDS: neg_neg += 1
        flipped = False
    return np.array([
        np.log1p(n),
        pos / n,
        neg / n,
        negs / n,
        neg_pos / n,
        neg_neg / n,
        (pos - neg + neg_neg - neg_pos) / n,
        (pos + neg) / n,
    ], dtype=np.float32)


def sentence_to_full_vector(tokens, model, dim=300):
    """Concatenate avg-GloVe (300) and hand-crafted features (8) -> 308."""
    return np.concatenate([
        sentence_to_vector(tokens, model, dim=dim),
        extra_features(tokens),
    ])


if __name__ == "__main__":
    print("Loading preprocessed dataset...")
    df = pd.read_csv("cleand date.csv")
    df["not_stopwords"] = df["not_stopwords"].apply(ast.literal_eval)

    embedding_model = load_glove()

    print("Building 308-D embeddings...")
    tqdm.pandas(desc="Embed")
    X = np.stack(df["not_stopwords"].progress_apply(
        lambda t: sentence_to_full_vector(t, embedding_model)
    ).values)
    y = np.array(
        [1 if str(s).lower() == "positive" else 0 for s in df["sentiment"]],
        dtype=np.int64,
    )

    print(f"Shapes: X={X.shape}, y={y.shape}")

    print("Splitting train/test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Saving .npy files...")
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    np.save("X_embeddings.npy", X)
    np.save("y_labels.npy", y)
    print("Done!")