import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense


print("TensorFlow version:", tf.__version__)


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("dataset/train_cleaned.csv")

# First 100 records for testing
df = df.head(100)


# ==========================================
# 2. ADD START AND END TOKENS
# ==========================================

df["clean_summary"] = (
    "<start> " +
    df["clean_summary"].astype(str) +
    " <end>"
)


# ==========================================
# 3. TOKENIZERS
# ==========================================

article_tokenizer = Tokenizer(
    num_words=10000,
    oov_token="<OOV>"
)

summary_tokenizer = Tokenizer(
    num_words=5000,
    oov_token="<OOV>"
)


article_tokenizer.fit_on_texts(
    df["clean_article"]
)

summary_tokenizer.fit_on_texts(
    df["clean_summary"]
)


# ==========================================
# 4. TEXT → SEQUENCES
# ==========================================

article_sequences = article_tokenizer.texts_to_sequences(
    df["clean_article"]
)

summary_sequences = summary_tokenizer.texts_to_sequences(
    df["clean_summary"]
)


# ==========================================
# 5. PADDING
# ==========================================

article_padded = pad_sequences(
    article_sequences,
    maxlen=400,
    padding="post"
)

summary_padded = pad_sequences(
    summary_sequences,
    maxlen=100,
    padding="post"
)


# ==========================================
# 6. DECODER INPUT & TARGET
# ==========================================

decoder_input = summary_padded[:, :-1]

decoder_target = summary_padded[:, 1:]


print("\nData preparation completed!")

print("Article shape:", article_padded.shape)
print("Summary shape:", summary_padded.shape)

print("Decoder input shape:", decoder_input.shape)
print("Decoder target shape:", decoder_target.shape)


# ==========================================
# 7. VOCABULARY SIZE
# ==========================================

article_vocab_size = min(
    10000,
    len(article_tokenizer.word_index) + 1
)

summary_vocab_size = min(
    5000,
    len(summary_tokenizer.word_index) + 1
)

print("\nArticle vocabulary size:", article_vocab_size)
print("Summary vocabulary size:", summary_vocab_size)


# ==========================================
# 8. ENCODER
# ==========================================

encoder_inputs = Input(
    shape=(400,),
    name="encoder_input"
)

encoder_embedding = Embedding(
    input_dim=article_vocab_size,
    output_dim=128,
    mask_zero=True,
    name="encoder_embedding"
)(encoder_inputs)

encoder_lstm = LSTM(
    128,
    return_state=True,
    name="encoder_lstm"
)

encoder_outputs, state_h, state_c = encoder_lstm(
    encoder_embedding
)

encoder_states = [state_h, state_c]


# ==========================================
# 9. DECODER
# ==========================================

decoder_inputs = Input(
    shape=(99,),
    name="decoder_input"
)

decoder_embedding = Embedding(
    input_dim=summary_vocab_size,
    output_dim=128,
    mask_zero=True,
    name="decoder_embedding"
)(decoder_inputs)

decoder_lstm = LSTM(
    128,
    return_sequences=True,
    return_state=True,
    name="decoder_lstm"
)

decoder_outputs, _, _ = decoder_lstm(
    decoder_embedding,
    initial_state=encoder_states
)


# ==========================================
# 10. OUTPUT LAYER
# ==========================================

decoder_dense = Dense(
    summary_vocab_size,
    activation="softmax",
    name="output_layer"
)

decoder_outputs = decoder_dense(
    decoder_outputs
)


# ==========================================
# 11. CREATE MODEL
# ==========================================

model = Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs
)


# ==========================================
# 12. COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nSeq2Seq model created successfully!")


# ==========================================
# 13. MODEL SUMMARY
# ==========================================

model.summary()


# ==========================================
# 14. PREPARE TARGET
# ==========================================

# Add one dimension because sparse categorical
# crossentropy expects integer class labels.

decoder_target = decoder_target[..., None]


print("\nStarting training...")


# ==========================================
# 15. TRAIN MODEL
# ==========================================

history = model.fit(
    [article_padded, decoder_input],
    decoder_target,
    batch_size=16,
    epochs=2,
    validation_split=0.1
)


print("\nTraining completed successfully!")
# ==========================================
# SAVE TEST MODEL
# ==========================================

model.save("model/text_summarization_test.keras")

print("\nModel saved successfully!")
print("File: model/text_summarization_test.keras")