import pandas as pd
import pickle
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint


print("TensorFlow version:", tf.__version__)


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("dataset/train_cleaned.csv")

df["clean_summary"] = (
    "<start> "
    + df["clean_summary"].astype(str)
    + " <end>"
)

print("Training records:", len(df))


# ==========================================
# 2. TOKENIZERS
# ==========================================

article_tokenizer = Tokenizer(
    num_words=10000,
    oov_token="<OOV>"
)

summary_tokenizer = Tokenizer(
    num_words=5000,
    oov_token="<OOV>"
)

article_tokenizer.fit_on_texts(df["clean_article"])
summary_tokenizer.fit_on_texts(df["clean_summary"])


# ==========================================
# 3. TEXT → SEQUENCES
# ==========================================

article_sequences = article_tokenizer.texts_to_sequences(
    df["clean_article"]
)

summary_sequences = summary_tokenizer.texts_to_sequences(
    df["clean_summary"]
)


# ==========================================
# 4. PADDING
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
# 5. DECODER INPUT & TARGET
# ==========================================

decoder_input = summary_padded[:, :-1]
decoder_target = summary_padded[:, 1:]

# Required for sparse categorical crossentropy
decoder_target = decoder_target[..., None]


print("\nData prepared!")
print("Article:", article_padded.shape)
print("Decoder input:", decoder_input.shape)
print("Decoder target:", decoder_target.shape)


# ==========================================
# 6. VOCABULARY
# ==========================================

article_vocab_size = min(
    10000,
    len(article_tokenizer.word_index) + 1
)

summary_vocab_size = min(
    5000,
    len(summary_tokenizer.word_index) + 1
)


print("\nArticle vocabulary used:", article_vocab_size)
print("Summary vocabulary used:", summary_vocab_size)


# ==========================================
# 7. ENCODER
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
# 8. DECODER
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
# 9. OUTPUT LAYER
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
# 10. CREATE MODEL
# ==========================================

model = Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs
)


# ==========================================
# 11. COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nModel created successfully!")

model.summary()


# ==========================================
# 12. CHECKPOINT
# ==========================================

checkpoint = ModelCheckpoint(
    "model/seq2seq_best.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# ==========================================
# 13. SAVE TOKENIZERS
# ==========================================

with open(
    "model/article_tokenizer.pkl",
    "wb"
) as file:
    pickle.dump(article_tokenizer, file)

with open(
    "model/summary_tokenizer.pkl",
    "wb"
) as file:
    pickle.dump(summary_tokenizer, file)


# ==========================================
# 14. TRAIN
# ==========================================

print("\n================================")
print("STARTING FULL MODEL TRAINING")
print("================================")

history = model.fit(
    [article_padded, decoder_input],
    decoder_target,
    batch_size=16,
    epochs=5,
    validation_split=0.1,
    callbacks=[checkpoint]
)


# ==========================================
# 15. SAVE FINAL MODEL
# ==========================================

model.save(
    "model/text_summarization_final.keras"
)

print("\n================================")
print("TRAINING COMPLETED")
print("================================")

print("Final model saved:")
print("model/text_summarization_final.keras")

print("Best model saved:")
print("model/seq2seq_best.keras")