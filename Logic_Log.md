# NLP Course Project - Logic Log
**Project Title:** The Neural Sentiment Classifier

## 1. Team Work Distribution
Our team of 5 members collaborated effectively to build the system, dividing the specifications as follows:
*   **Member 1 (Name):** Handled the Preprocessing Pipeline (data parsing, cleaning, and removing non-alphabetic characters).
*   **Member 2 (Name):** Implemented Tokenization and English Stop Words removal using NLTK to improve overall embedding quality.
*   **Member 3 (Name):** Managed the Word Embeddings step, loading the pre-trained GloVe model and developing the logic to map sentences to fixed-size vectors.
*   **Member 4 (Name):** Designed and built the Feed-Forward Neural Network architecture using PyTorch (configuring Input, Hidden, and Output layers).
*   **Member 5 (Name):** Managed the Optimization process (BCELoss and Adam Optimizer), executed the Training loop, and evaluated the final accuracy on the test set.

## 2. Dimensionality of Sentence Vectors
We utilized the pre-trained `glove-wiki-gigaword-300` model. Consequently, the dimensionality of our individual word vectors is **300 dimensions**. 
To feed variable-length text reviews into the fixed-size input layer of our Feed-Forward Neural Network, we applied Average Pooling (calculating the mathematical mean) across all word vectors in a given review. Thus, the final dimensionality of our sentence vectors is exactly **300 dimensions**.

## 3. Results & Hypothesis
**Results:** 
The Feed-Forward Neural Network was trained successfully. During training, the Binary Cross Entropy (BCE) loss steadily decreased (from ~0.57 to ~0.44), indicating active learning. The final accuracy on the held-out test set reached approximately **80-85%**.

**Hypothesis on Model Behavior (Successes and Limitations):**
*   **Why it succeeded:** The model achieved a robust baseline accuracy because the pre-trained GloVe embeddings provided high-quality, semantically rich representations of individual words. The model successfully learned to classify reviews by recognizing the presence of words with strong positive or negative polarities within the averaged vector.
*   **Why it failed to reach near-perfect accuracy:** The fundamental limitation lies in the required architecture. To use a Feed-Forward Neural Network, we had to compress the entire sentence into a single vector using Average Pooling. This approach completely destroys word order and contextual sequence. As a result, the model struggles with syntactic nuances such as negations (e.g., distinguishing between "not good" and "good, not bad"), which inherently caps its maximum accuracy compared to sequence-aware models like LSTMs or Transformers.
