import os
import re
import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
import gensim.downloader as api

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


# =========================================
# Model
# =========================================
class SentimentNN(nn.Module):
    def __init__(self, input_dim):
        super(SentimentNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.dropout1(out)
        out = self.relu(self.bn2(self.fc2(out)))
        out = self.dropout2(out)
        out = self.relu(self.bn3(self.fc3(out)))
        out = self.dropout3(out)
        out = self.sigmoid(self.fc4(out))
        return out


# =========================================
# GloVe loader with cache
# =========================================
def load_glove():
    if os.path.exists(GLOVE_CACHE_PATH):
        print(f"Loading GloVe from cache ({GLOVE_CACHE_PATH})...")
        return KeyedVectors.load(GLOVE_CACHE_PATH)

    print("First run -- downloading GloVe (one time only)...")
    model = api.load("glove-wiki-gigaword-300")
    model.save(GLOVE_CACHE_PATH)
    print(f"Saved to '{GLOVE_CACHE_PATH}' -- future runs will be fast.")
    return model


# =========================================
# Pipeline (matches EMB.py exactly)
# =========================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()


def sentence_to_vector(tokens, model, dim=300):
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
    """300 GloVe + 8 hand-crafted features = 308."""
    return np.concatenate([
        sentence_to_vector(tokens, model, dim=dim),
        extra_features(tokens),
    ])


# =========================================
# Load model
# =========================================
def load_model(model_path="sentiment_model.pth"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at '{model_path}'. Train first.")

    if os.path.exists("X_train.npy"):
        input_dim = np.load("X_train.npy").shape[1]
        print(f"Input dim from X_train.npy: {input_dim}")
    else:
        input_dim = 308
        print(f"X_train.npy not found -- using input_dim={input_dim}")

    model = SentimentNN(input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"Model loaded from '{model_path}'")
    return model


# =========================================
# Predict
# =========================================
def predict(sentence, model, embedding_model):
    tokens = preprocess_text(sentence)
    if not tokens:
        return None, None

    vec = sentence_to_full_vector(tokens, embedding_model, dim=300)
    vec_tensor = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        probability = model(vec_tensor).item()

    sentiment = "Positive" if probability > 0.5 else "Negative"
    confidence = probability if probability > 0.5 else 1 - probability
    return sentiment, confidence


# =========================================
# Tests
# =========================================
def run_tests(model, embedding_model):
    test_sentences = [
    # Clear positive
    "This movie was absolutely fantastic and beautifully crafted.",
    "The acting was superb and the story was touching.",
    "I loved every second of it, a true masterpiece.",
    "One of the best films I have ever watched, highly recommend.",
    "The characters were brilliant and the plot was perfect.",

    # Clear negative
    "Terrible experience, I hated every moment of it.",
    "Worst movie I have ever seen, total waste of time.",
    "Awful acting and a boring story, completely disappointing.",
    "I could not even finish it, absolutely horrible.",
    "The film was ugly, stupid and a complete failure.",

    # Negation
    "The film was not good at all.",
    "I did not hate the movie, it was actually pretty fun.",
    "It was not bad, I really enjoyed it.",
    "The acting was not great, and the story was boring.",
    "Not the worst movie but definitely not enjoyable.",
    "I did not find anything amazing about this film.",

    # Mixed
    "The start was amazing but the ending was completely disappointing.",
    "Great acting but the story was awful and boring.",
    "The visuals were beautiful but the plot was a total waste.",
    "Some scenes were fantastic but most of the film was terrible.",
]

    print("\n" + "=" * 65)
    for sentence in test_sentences:
        sentiment, confidence = predict(sentence, model, embedding_model)
        if sentiment is None:
            print(f"Sentence : '{sentence}'")
            print("Result   : No valid tokens found.")
        else:
            icon = "+" if sentiment == "Positive" else "-"
            print(f"Sentence : '{sentence}'")
            print(f"Result   : [{icon}] {sentiment}  |  Confidence: {confidence * 100:.1f}%")
        print("-" * 65)


# =========================================
# Main
# =========================================
if __name__ == "__main__":
    embedding_model = load_glove()
    model = load_model("sentiment_model.pth")

    run_tests(model, embedding_model)

    print("\n" + "=" * 65)
    print("Interactive Mode  (type 'exit' to quit)")
    print("=" * 65)

    while True:
        user_input = input("\nEnter sentence: ").strip()
        if user_input.lower() in ("exit", "quit", "q"):
            break
        if not user_input:
            continue

        sentiment, confidence = predict(user_input, model, embedding_model)
        if sentiment is None:
            print("No valid tokens found.")
        else:
            icon = "+" if sentiment == "Positive" else "-"
            print(f"Result: [{icon}] {sentiment}  |  Confidence: {confidence * 100:.1f}%")