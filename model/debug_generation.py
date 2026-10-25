import os
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ================================
# LOAD MODEL
# ================================

MODEL_PATH = "model/text_summarization_final.keras"
ARTICLE_TOKENIZER_PATH = "model/article_tokenizer.pkl"
SUMMARY_TOKENIZER_PATH = "model/summary_tokenizer.pkl"

model = tf.keras.models.load_model(MODEL_PATH)

with open(ARTICLE_TOKENIZER_PATH, "rb") as f:
    article_tokenizer = pickle.load(f)

with open(SUMMARY_TOKENIZER_PATH, "rb") as f:
    summary_tokenizer = pickle.load(f)

print("Model loaded!")

START_TOKEN = summary_tokenizer.word_index["<start>"]
END_TOKEN = summary_tokenizer.word_index["<end>"]

print("START TOKEN:", START_TOKEN)
print("END TOKEN:", END_TOKEN)

# ================================
# TEST ARTICLE
# ================================

article = """
The Indian government has announced a new initiative to promote
renewable energy across the country. The program will support solar
and wind power projects and provide financial assistance to companies
developing clean energy technologies. Officials said the initiative
will help reduce carbon emissions and increase India's renewable
energy capacity over the coming years.
"""

# ================================
# PREPROCESS ARTICLE
# ================================

clean_article = article.lower()
clean_article = re.sub(r"[^a-zA-Z0-9\s]", "", clean_article)
clean_article = re.sub(r"\s+", " ", clean_article).strip()

article_sequence = article_tokenizer.texts_to_sequences([clean_article])

article_padded = pad_sequences(
    article_sequence,
    maxlen=400,
    padding="post",
    truncating="post"
)

# ================================
# DEBUG GENERATION
# ================================

print("\n================================")
print("DEBUG GENERATION")
print("================================")

decoder_sequence = [START_TOKEN]

reverse_word_index = {
    value: key
    for key, value in summary_tokenizer.word_index.items()
}

for step in range(20):

    decoder_padded = pad_sequences(
        [decoder_sequence],
        maxlen=99,
        padding="post",
        truncating="post"
    )

    predictions = model.predict(
        [article_padded, decoder_padded],
        verbose=0
    )

    # Prediction at the current decoder position
    position = len(decoder_sequence) - 1

    predicted_id = np.argmax(
        predictions[0, position, :]
    )

    predicted_word = reverse_word_index.get(
        predicted_id,
        "<UNKNOWN>"
    )

    print(
        f"Step {step + 1}: "
        f"ID = {predicted_id}, "
        f"Word = {predicted_word}"
    )

    # Stop if END token predicted
    if predicted_id == END_TOKEN:
        print("\nModel predicted <end>.")
        break

    decoder_sequence.append(predicted_id)

print("\n================================")
print("TOKEN SEQUENCE")
print("================================")
print(decoder_sequence)

print("\n================================")
print("DECODED WORDS")
print("================================")

words = []

for token_id in decoder_sequence:
    word = reverse_word_index.get(token_id, "<UNKNOWN>")

    if word not in ["<start>", "<end>"]:
        words.append(word)

print(" ".join(words))
