import pickle
import numpy as np
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from rouge_score import rouge_scorer


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading trained model...")

model = load_model(
    "model/text_summarization_final.keras"
)

print("Model loaded!")


# ==========================================
# LOAD TOKENIZERS
# ==========================================

with open("model/article_tokenizer.pkl", "rb") as f:
    article_tokenizer = pickle.load(f)

with open("model/summary_tokenizer.pkl", "rb") as f:
    summary_tokenizer = pickle.load(f)


# ==========================================
# SPECIAL TOKENS
# ==========================================

start_token = summary_tokenizer.word_index["<start>"]
end_token = summary_tokenizer.word_index["<end>"]

print("START TOKEN:", start_token)
print("END TOKEN:", end_token)


# ==========================================
# ORIGINAL ARTICLE
# ==========================================

article = """
The Indian government has announced a new initiative to promote
renewable energy across the country. The program will support solar
and wind power projects and provide financial assistance to companies
developing clean energy technologies. Officials said the initiative
will help reduce carbon emissions and increase India's renewable
energy capacity over the coming years.
"""


# ==========================================
# REFERENCE SUMMARY
# ==========================================

reference_summary = """
The Indian government launched an initiative to promote solar and
wind energy, support clean technology companies, reduce carbon
emissions, and increase renewable energy capacity.
"""


# ==========================================
# DISPLAY ORIGINAL ARTICLE
# ==========================================

print("\n================================")
print("ORIGINAL ARTICLE")
print("================================")

print(article.strip())


# ==========================================
# DISPLAY REFERENCE
# ==========================================

print("\n================================")
print("REFERENCE SUMMARY")
print("================================")

print(reference_summary.strip())


# ==========================================
# PREPROCESS ARTICLE
# ==========================================

clean_article = article.lower()

clean_article = re.sub(
    r"[^a-z0-9\s]",
    "",
    clean_article
)

sequence = article_tokenizer.texts_to_sequences(
    [clean_article]
)

article_padded = pad_sequences(
    sequence,
    maxlen=400,
    padding="post"
)


# ==========================================
# GENERATE SUMMARY
# ==========================================

print("\n================================")
print("GENERATED SUMMARY")
print("================================")


generated = [start_token]

for i in range(50):

    decoder_input = pad_sequences(
        [generated],
        maxlen=99,
        padding="post"
    )

    prediction = model.predict(
        [article_padded, decoder_input],
        verbose=0
    )

    position = len(generated) - 1

    next_token = int(
        np.argmax(
            prediction[0, position, :]
        )
    )

    if next_token == end_token:
        break

    if next_token == 0:
        break

    generated.append(next_token)


# ==========================================
# TOKEN → WORD
# ==========================================

reverse_word_index = {
    value: key
    for key, value in summary_tokenizer.word_index.items()
}

generated_words = []

for token_id in generated:

    word = reverse_word_index.get(
        token_id,
        ""
    )

    if word not in [
        "<start>",
        "<end>",
        "<OOV>",
        ""
    ]:
        generated_words.append(word)


generated_summary = " ".join(
    generated_words
)

print(generated_summary)


# ==========================================
# ROUGE
# ==========================================

print("\n================================")
print("ROUGE RESULTS")
print("================================")


scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)

scores = scorer.score(
    reference_summary.strip(),
    generated_summary
)


print(
    "ROUGE-1:",
    round(scores["rouge1"].fmeasure, 4)
)

print(
    "ROUGE-2:",
    round(scores["rouge2"].fmeasure, 4)
)

print(
    "ROUGE-L:",
    round(scores["rougeL"].fmeasure, 4)
)


# ==========================================
# COMPRESSION RATIO
# ==========================================

original_words = len(
    article.split()
)

generated_words_count = len(
    generated_summary.split()
)

if original_words > 0:

    compression_ratio = (
        1 -
        generated_words_count / original_words
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
    "Generated words:",
    generated_words_count
)

print(
    "Compression ratio:",
    round(compression_ratio, 2),
    "%"
)


print("\n================================")
print("TEST COMPLETED")
print("================================")