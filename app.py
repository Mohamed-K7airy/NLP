"""
Flask Web Application for Sentiment Analysis
Uses the trained PyTorch model + GloVe embeddings to classify user sentences.
"""

from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import os
from EMB import sentence_to_full_vector, NEGATION_WORDS
from preprocessing import preprocess_text, load_glove

# =========================================
# 1) Model Definition (imported from main.py)
# =========================================
from main import SentimentNN

# =========================================
# 2) Load Resources at Startup
# =========================================
print("Loading GloVe model...")
embedding_model = load_glove()
print("GloVe loaded!")


# Load saved model
MODEL_PATH = "sentiment_model.pth"

if not os.path.exists(MODEL_PATH):
    print("ERROR: {} not found! Run main.py first to train the model.".format(MODEL_PATH))
    exit(1)

print("Loading saved model weights...")
model = SentimentNN(308)  # 300 GloVe + 8 sentiment features
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print("Model loaded from file!")

# =========================================
# 3) Flask App
# =========================================
app = Flask(__name__, static_folder="static")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sentence = data.get("sentence", "")

    if not sentence.strip():
        return jsonify({"error": "Please enter a sentence."}), 400

    # Use the robust preprocessing pipeline
    prep_data = preprocess_text(sentence)
    tokens = prep_data["tokens"]

    if len(tokens) == 0:
        return jsonify({"error": "No valid English words were found in the sentence."}), 400

    # Convert to vector (300 GloVe + 8 sentiment features = 308)
    vec = sentence_to_full_vector(tokens, embedding_model, dim=300)
    vec_tensor = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(vec_tensor)
        probability = output.item()
        sentiment = "positive" if probability > 0.5 else "negative"
        confidence = probability if sentiment == "positive" else 1 - probability

    # Calculate token details matching EMB.py's weighting logic for the UI
    token_details = []
    after_neg = False
    for token in tokens:
        if token in NEGATION_WORDS:
            after_neg = True
            continue
        if token in embedding_model:
            weight = 0.25 if after_neg else 1.0
            token_details.append({
                "token": token,
                "weight": weight,
                "negated": after_neg
            })
        after_neg = False
        
    top_tokens = sorted(token_details, key=lambda x: x["weight"], reverse=True)[:10]

    return jsonify({
        "sentence": sentence,
        "sentiment": sentiment,
        "confidence": round(confidence * 100, 2),
        "probability": round(probability, 4),
        "tokens_used": len(tokens),
        "tokens_sample": tokens[:12],
        "top_tokens": top_tokens,
        "negation_pairs": prep_data.get("negation_pairs", []),
        "removed_stopwords": prep_data.get("removed_stopwords", 0),
        "removed_short_words": prep_data.get("removed_short_words", 0),
        "input_quality": "good" if len(tokens) >= 3 else "short"
    })

if __name__ == "__main__":
    app.run(debug=False, port=5000)
