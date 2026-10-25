import pandas as pd
import pickle
import numpy as np

from rouge_score import rouge_scorer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load model
model = load_model("model/text_summarization_final.keras")

# Load tokenizers
with open("model/article_tokenizer.pkl", "rb") as f:
    article_tokenizer = pickle.load(f)

with open("model/summary_tokenizer.pkl", "rb") as f:
    summary_tokenizer = pickle.load(f)

# Load test data
df = pd.read_csv("dataset/test_cleaned.csv")

# ROUGE
scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)

scores = []

# Evaluate first 20 test articles
for i in range(20):

    article = str(df.iloc[i]["clean_article"])
    reference = str(df.iloc[i]["clean_summary"])

    sequence = article_tokenizer.texts_to_sequences([article])

    padded = pad_sequences(
        sequence,
        maxlen=400,
        padding="post"
    )

    start_token = summary_tokenizer.word_index["<start>"]
    end_token = summary_tokenizer.word_index["<end>"]

    generated = [start_token]

    for _ in range(50):

        decoder_input = pad_sequences(
            [generated],
            maxlen=99,
            padding="post"
        )

        prediction = model.predict(
            [padded, decoder_input],
            verbose=0
        )

        position = len(generated) - 1

        next_token = int(
            np.argmax(prediction[0, position, :])
        )

        if next_token == end_token or next_token == 0:
            break

        generated.append(next_token)

    reverse_words = {
        value: key
        for key, value in summary_tokenizer.word_index.items()
    }

    words = []

    for token in generated:

        word = reverse_words.get(token, "")

        if word not in [
            "<start>",
            "<end>",
            "<OOV>",
            ""
        ]:
            words.append(word)

    generated_summary = " ".join(words)

    result = scorer.score(
        reference,
        generated_summary
    )

    scores.append(result)


# Average scores
rouge1 = np.mean([
    x["rouge1"].fmeasure for x in scores
])

rouge2 = np.mean([
    x["rouge2"].fmeasure for x in scores
])

rougeL = np.mean([
    x["rougeL"].fmeasure for x in scores
])


print("\n================================")
print("FINAL ROUGE RESULTS")
print("================================")

print("ROUGE-1:", round(rouge1, 4))
print("ROUGE-2:", round(rouge2, 4))
print("ROUGE-L:", round(rougeL, 4))

print("\nEvaluation completed on 20 test documents.")
