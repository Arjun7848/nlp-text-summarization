import pandas as pd
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


print("================================")
print("LOADING TRAINED MODEL")
print("================================")


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

model = load_model(
    "model/text_summarization_final.keras"
)

print("Model loaded successfully!")


# ==========================================
# 2. LOAD TOKENIZERS
# ==========================================

with open(
    "model/article_tokenizer.pkl",
    "rb"
) as file:
    article_tokenizer = pickle.load(file)


with open(
    "model/summary_tokenizer.pkl",
    "rb"
) as file:
    summary_tokenizer = pickle.load(file)


print("Tokenizers loaded successfully!")


# ==========================================
# 3. CHECK SPECIAL TOKENS
# ==========================================

start_token = summary_tokenizer.word_index.get("<start>")
end_token = summary_tokenizer.word_index.get("<end>")


print("\nSpecial tokens:")
print("<start> ID:", start_token)
print("<end> ID:", end_token)


if start_token is None or end_token is None:
    raise ValueError(
        "ERROR: <start> or <end> token not found!"
    )


# ==========================================
# 4. LOAD TEST DATA
# ==========================================

test_data = pd.read_csv(
    "dataset/test_cleaned.csv"
)

print(
    "\nTest records:",
    len(test_data)
)


# ==========================================
# 5. SELECT ONE ARTICLE
# ==========================================

article_text = str(
    test_data.iloc[0]["clean_article"]
)

reference_summary = str(
    test_data.iloc[0]["clean_summary"]
)


print("\n================================")
print("ORIGINAL ARTICLE")
print("================================")

print(article_text)


print("\n================================")
print("REFERENCE SUMMARY")
print("================================")

print(reference_summary)


# ==========================================
# 6. CONVERT ARTICLE TO SEQUENCE
# ==========================================

article_sequence = article_tokenizer.texts_to_sequences(
    [article_text]
)


article_padded = pad_sequences(
    article_sequence,
    maxlen=400,
    padding="post"
)


# ==========================================
# 7. GENERATE SUMMARY
# ==========================================

print("\n================================")
print("GENERATING SUMMARY...")
print("================================")


generated_tokens = [start_token]


# Maximum summary length
max_summary_length = 99


for step in range(max_summary_length):

    decoder_input = pad_sequences(
        [generated_tokens],
        maxlen=99,
        padding="post"
    )

    prediction = model.predict(
        [article_padded, decoder_input],
        verbose=0
    )


    # Prediction at the last generated token
    current_position = len(generated_tokens) - 1

    next_token = np.argmax(
        prediction[0, current_position, :]
    )


    # Stop if END token is generated
    if next_token == end_token:
        break


    # Avoid invalid token
    if next_token == 0:
        break


    generated_tokens.append(
        int(next_token)
    )


# ==========================================
# 8. CONVERT TOKENS TO WORDS
# ==========================================

reverse_word_index = {
    value: key
    for key, value in summary_tokenizer.word_index.items()
}


generated_words = []

for token_id in generated_tokens:

    word = reverse_word_index.get(
        token_id,
        ""
    )

    if word in ["<start>", "<end>", "<OOV>", ""]:
        continue

    generated_words.append(word)


generated_summary = " ".join(
    generated_words
)


# ==========================================
# 9. DISPLAY RESULT
# ==========================================

print("\n================================")
print("GENERATED SUMMARY")
print("================================")

print(generated_summary)


# ==========================================
# 10. BASIC STATISTICS
# ==========================================

original_words = len(
    article_text.split()
)

summary_words = len(
    generated_summary.split()
)


if original_words > 0:

    compression_ratio = (
        1 -
        (summary_words / original_words)
    ) * 100

else:

    compression_ratio = 0


print("\n================================")
print("SUMMARY STATISTICS")
print("================================")

print(
    "Original words:",
    original_words
)

print(
    "Generated summary words:",
    summary_words
)

print(
    "Compression ratio:",
    round(compression_ratio, 2),
    "%"
)

print("\n================================")
print("SUMMARY GENERATION COMPLETED")
print("================================")