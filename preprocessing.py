import ast
import os
import re
import unicodedata
import numpy as np
import pandas as pd
import gensim.downloader as api
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

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

# Keep negation words in the vocabulary / out of stopwords
stop_words = stop_words - NEGATION_WORDS

GLOVE_CACHE_PATH = "glove_cache.kv"


# ─── Robust Preprocessing Pipeline ─────────────────────────────────
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    
    # 1) Clean text
    cleaned = text.lower()
    cleaned = unicodedata.normalize("NFKD", cleaned)
    cleaned = cleaned.encode("ascii", errors="ignore").decode()
    cleaned = re.sub(r"http\S+", " ", cleaned)  # Remove URLs
    cleaned = re.sub(r"<.*?>", " ", cleaned)     # Remove HTML tags
    cleaned = re.sub(r"\bbr\b", " ", cleaned)    # Remove <br> leftover string
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", cleaned) # Keep only letters and spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    # 2) Tokenize
    raw_tokens = cleaned.split()
    
    # 3) Filter stopwords and short words, and track negations
    tokens = []
    removed_stopwords = 0
    removed_short_words = 0
    negation_pairs = []
    
    after_neg = False
    last_neg_word = None
    
    for token in raw_tokens:
        # Check if it is a stopword or short word (but keep negation words!)
        if token in NEGATION_WORDS:
            tokens.append(token)
            after_neg = True
            last_neg_word = token
            continue
            
        if token in stop_words:
            removed_stopwords += 1
            # Stopword shouldn't break or satisfy a negation pair, just skip it
            continue
            
        if len(token) <= 2:
            removed_short_words += 1
            # Short word skipped
            continue
            
        # Valid token
        tokens.append(token)
        if after_neg:
            negation_pairs.append([last_neg_word, token])
            after_neg = False
            last_neg_word = None
            
    return {
        "tokens": tokens,
        "negation_pairs": negation_pairs,
        "removed_stopwords": removed_stopwords,
        "removed_short_words": removed_short_words
    }



# ─── GloVe loader with cache ──────────────────────────────────────
def load_glove():
    """
    First run: download from internet and save to disk.
    After that: load from disk in seconds.
    """
    if os.path.exists(GLOVE_CACHE_PATH):
        print(f"Loading GloVe from cache ({GLOVE_CACHE_PATH})...")
        return KeyedVectors.load(GLOVE_CACHE_PATH)

    print("First run -- downloading GloVe (one time only)...")
    model = api.load("glove-wiki-gigaword-300")
    model.save(GLOVE_CACHE_PATH)
    print(f"Saved to '{GLOVE_CACHE_PATH}' -- future runs will be fast.")
    return model



# ─── Bigram builder ───────────────────────────────────────────────
def apply_bigrams(tokens):
    """
    يدمج كل كلمة نفي مع الكلمة اللي بعدها كـ bigram واحد.
    مثال: ["not", "bad", "movie"] → ["not_bad", "movie"]
    """
    result = []
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token in NEGATION_WORDS and i + 1 < len(tokens):
            result.append(f"{token}_{tokens[i + 1]}")
            skip_next = True
        else:
            result.append(token)
    return result


# ─── Weighted vector builder ──────────────────────────────────────
def sentence_to_vector(tokens, model, dim=300):
    """
    Weighted average GloVe:
    - الكلمات الإيجابية/السلبية تاخد وزن 2.0
    - باقي الكلمات تاخد وزن 1.0
    بيمنع الكلمات المحايدة من تخفيف الإشارة العاطفية.
    """
    vectors, weights = [], []
    for token in tokens:
        if token in model:
            w = 2.0 if (token in POSITIVE_WORDS or token in NEGATIVE_WORDS) else 1.0
            vectors.append(model[token])
            weights.append(w)
    if not vectors:
        return np.zeros(dim, dtype=np.float32)
    return np.average(
        np.asarray(vectors), axis=0, weights=np.asarray(weights)
    ).astype(np.float32)


# ─── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv("cleand date.csv")
    df["not_stopwords"] = df["not_stopwords"].apply(ast.literal_eval)

    df["tokens_bigrammed"] = df["not_stopwords"].apply(apply_bigrams)

    embedding_model = load_glove()

    print("Building embeddings (weighted)...")
    tqdm.pandas(desc="Embed")
    X = np.stack(
        df["tokens_bigrammed"].progress_apply(
            lambda t: sentence_to_vector(t, embedding_model)
        ).values
    )

    y = np.array(
        [1 if str(s).lower() == "positive" else 0 for s in df["sentiment"]],
        dtype=np.int64,
    )

    print(f"Shapes: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    np.save("X_embeddings.npy", X)
    np.save("y_labels.npy", y)
    print("Done!")