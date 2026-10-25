import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Load datasets
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

# Create preprocessing tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    words = text.split()

    # Stopword removal + lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


print("Processing training data...")

train_df["clean_article"] = train_df["article"].apply(preprocess_text)
train_df["clean_summary"] = train_df["summary"].apply(preprocess_text)

print("Training preprocessing completed!")


print("Processing testing data...")

test_df["clean_article"] = test_df["article"].apply(preprocess_text)
test_df["clean_summary"] = test_df["summary"].apply(preprocess_text)

print("Testing preprocessing completed!")


# Save cleaned datasets
train_df.to_csv("dataset/train_cleaned.csv", index=False)
test_df.to_csv("dataset/test_cleaned.csv", index=False)

print("\n================================")
print("PREPROCESSING COMPLETED")
print("================================")
print("Training records:", len(train_df))
print("Testing records:", len(test_df))

print("\nFiles created:")
print("dataset/train_cleaned.csv")
print("dataset/test_cleaned.csv")